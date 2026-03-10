#!/usr/bin/env python3
"""Build daily adjustment factors (hfq/qfq) from local zip datasets.

We store:
- raw (不复权) bars separately
- adj factors separately
Then hfq/qfq prices can be derived as: price_adj = price_raw * factor

Input
- A股数据_zip/daily.zip
- A股数据_zip/daily_hfq.zip
- A股数据_zip/daily_qfq.zip (optional but recommended)

Output (Hive partitions)
- data/lake/adj_factor_daily/year=YYYY/month=MM/*.parquet

Factor construction (per symbol, per date)
- hfq_factor = close_hfq / close_raw
- qfq_factor = close_qfq / close_raw

Notes
- Factors are derived from CLOSE to avoid needing a corporate action calendar.
- Small rounding noise may exist; for most research/selection it is fine.
"""

from __future__ import annotations

import argparse
import io
import shutil
import zipfile
from contextlib import ExitStack
from pathlib import Path

import duckdb
import pandas as pd

from prepare_akquant_data import load_stock_map, normalize_stock_code

CSV_ENCODINGS = ("utf-8", "utf-8-sig", "gb18030", "gbk")

RAW_MAP = {"日期": "date", "收盘": "close_raw"}
HFQ_MAP = {"日期": "date", "收盘": "close_hfq"}
QFQ_MAP = {"日期": "date", "收盘": "close_qfq"}


def read_csv_bytes(raw: bytes) -> pd.DataFrame:
    for enc in CSV_ENCODINGS:
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(io.BytesIO(raw))


def build_member_map(zf: zipfile.ZipFile) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for name in zf.namelist():
        if not name.lower().endswith(".csv"):
            continue
        stem = Path(name).stem
        try:
            code = normalize_stock_code(stem.split("_")[0])
        except Exception:
            continue
        mapping[code] = name
    return mapping


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build hfq/qfq adjustment factors parquet lake")
    p.add_argument("--raw-zip", type=Path, default=Path("A股数据_zip/daily.zip"))
    p.add_argument("--hfq-zip", type=Path, default=Path("A股数据_zip/daily_hfq.zip"))
    p.add_argument("--qfq-zip", type=Path, default=Path("A股数据_zip/daily_qfq.zip"))
    p.add_argument("--no-qfq", action="store_true", help="不计算 qfq_factor")

    p.add_argument("--stock-list", type=Path, default=Path("A股数据_zip/股票列表.csv"))
    p.add_argument("--out-dir", type=Path, default=Path("data/lake/adj_factor_daily"))
    p.add_argument(
        "--tmp-db",
        type=Path,
        default=Path("data/duckdb/_ingest_adj_factor_daily.duckdb"),
        help="临时 DuckDB 文件，用于 ingest 后 COPY 输出 parquet",
    )
    p.add_argument("--limit", type=int, default=None, help="仅处理前 N 只股票（自检用）")
    p.add_argument("--overwrite", action="store_true", help="覆盖 out-dir 和 tmp-db")
    p.add_argument("--progress-every", type=int, default=200, help="进度打印间隔")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    for zp in [args.raw_zip, args.hfq_zip]:
        if not zp.exists():
            raise FileNotFoundError(f"missing zip: {zp}")
    if not args.no_qfq and not args.qfq_zip.exists():
        raise FileNotFoundError(f"missing qfq zip: {args.qfq_zip}")
    if not args.stock_list.exists():
        raise FileNotFoundError(f"missing stock list: {args.stock_list}")

    if args.overwrite:
        if args.out_dir.exists():
            shutil.rmtree(args.out_dir)
        if args.tmp_db.exists():
            args.tmp_db.unlink()
    else:
        if args.out_dir.exists() and any(args.out_dir.rglob("*.parquet")):
            raise FileExistsError(f"out-dir not empty: {args.out_dir} (use --overwrite)")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.tmp_db.parent.mkdir(parents=True, exist_ok=True)

    stock_map = load_stock_map(args.stock_list)

    con = duckdb.connect(args.tmp_db.as_posix())
    con.execute("PRAGMA threads=8")
    con.execute(
        """
        CREATE TABLE adj_factor_daily (
          date DATE,
          symbol VARCHAR,
          hfq_factor DOUBLE,
          qfq_factor DOUBLE,
          year INTEGER,
          month INTEGER
        );
        """
    )

    inserted_symbols = 0
    skipped = 0

    with ExitStack() as stack:
        raw_zf = stack.enter_context(zipfile.ZipFile(args.raw_zip))
        hfq_zf = stack.enter_context(zipfile.ZipFile(args.hfq_zip))
        qfq_zf = None
        if not args.no_qfq:
            qfq_zf = stack.enter_context(zipfile.ZipFile(args.qfq_zip))

        raw_map = build_member_map(raw_zf)
        hfq_map = build_member_map(hfq_zf)
        qfq_map = build_member_map(qfq_zf) if qfq_zf else {}

        codes = sorted(set(raw_map) & set(hfq_map))
        if args.limit is not None:
            codes = codes[: max(0, int(args.limit))]

        for idx, code in enumerate(codes, start=1):
            symbol = stock_map.get(code)
            if not symbol:
                skipped += 1
                continue

            raw_df = read_csv_bytes(raw_zf.read(raw_map[code]))
            hfq_df = read_csv_bytes(hfq_zf.read(hfq_map[code]))

            if raw_df is None or raw_df.empty or hfq_df is None or hfq_df.empty:
                skipped += 1
                continue

            if not set(RAW_MAP).issubset(raw_df.columns) or not set(HFQ_MAP).issubset(hfq_df.columns):
                skipped += 1
                continue

            raw_w = raw_df.rename(columns=RAW_MAP)[list(RAW_MAP.values())].copy()
            hfq_w = hfq_df.rename(columns=HFQ_MAP)[list(HFQ_MAP.values())].copy()

            raw_w["date"] = pd.to_datetime(raw_w["date"], errors="coerce")
            hfq_w["date"] = pd.to_datetime(hfq_w["date"], errors="coerce")
            raw_w["close_raw"] = pd.to_numeric(raw_w["close_raw"], errors="coerce")
            hfq_w["close_hfq"] = pd.to_numeric(hfq_w["close_hfq"], errors="coerce")

            merged = raw_w.merge(hfq_w, on="date", how="inner")
            merged = merged.dropna(subset=["date", "close_raw", "close_hfq"])
            merged = merged.loc[merged["close_raw"] > 0].copy()
            if merged.empty:
                skipped += 1
                continue

            merged["hfq_factor"] = merged["close_hfq"] / merged["close_raw"]
            merged["qfq_factor"] = pd.NA

            if qfq_zf and code in qfq_map and not args.no_qfq:
                qfq_df = read_csv_bytes(qfq_zf.read(qfq_map[code]))
                if qfq_df is not None and not qfq_df.empty and set(QFQ_MAP).issubset(qfq_df.columns):
                    qfq_w = qfq_df.rename(columns=QFQ_MAP)[list(QFQ_MAP.values())].copy()
                    qfq_w["date"] = pd.to_datetime(qfq_w["date"], errors="coerce")
                    qfq_w["close_qfq"] = pd.to_numeric(qfq_w["close_qfq"], errors="coerce")
                    merged = merged.merge(qfq_w, on="date", how="left")
                    merged["qfq_factor"] = merged["close_qfq"] / merged["close_raw"]

            out = merged[["date", "hfq_factor", "qfq_factor"]].copy()
            out["symbol"] = symbol
            out["year"] = out["date"].dt.year.astype("int32")
            out["month"] = out["date"].dt.month.astype("int8")

            # reduce rounding noise a bit (optional)
            out["hfq_factor"] = pd.to_numeric(out["hfq_factor"], errors="coerce").round(10)
            out["qfq_factor"] = pd.to_numeric(out["qfq_factor"], errors="coerce").round(10)

            con.register("tmp_df", out)
            con.execute(
                """
                INSERT INTO adj_factor_daily
                SELECT
                  date::DATE as date,
                  symbol,
                  hfq_factor,
                  qfq_factor,
                  year,
                  month
                FROM tmp_df;
                """
            )
            con.unregister("tmp_df")
            inserted_symbols += 1

            if idx % int(args.progress_every) == 0:
                print(f"[INFO] processed {idx}/{len(codes)}; inserted_symbols={inserted_symbols}; skipped={skipped}")

    rows = con.execute("SELECT COUNT(*) FROM adj_factor_daily").fetchone()[0]
    print(f"[INFO] ingest done: symbols={inserted_symbols}, skipped={skipped}, rows={rows}")

    out_dir = args.out_dir.resolve().as_posix()
    con.execute(
        f"""
        COPY (
          SELECT date, symbol, hfq_factor, qfq_factor, year, month
          FROM adj_factor_daily
        )
        TO '{out_dir}'
        (FORMAT PARQUET, PARTITION_BY (year, month), COMPRESSION ZSTD);
        """
    )

    print(f"[DONE] wrote parquet factors -> {args.out_dir}")
    con.close()


if __name__ == "__main__":
    main()
