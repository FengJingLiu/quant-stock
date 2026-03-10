#!/usr/bin/env python3
"""Build Parquet lake for A-share daily bars (unadjusted / 不复权).

Input
- A股数据_zip/daily.zip (per-symbol CSV inside zip)

Output (Hive partitions)
- data/lake/fact_bar_daily/adjust=none/year=YYYY/month=MM/*.parquet

Notes
- Uses DuckDB as an ingestion engine (fast, streaming-ish) to produce partitioned Parquet.
- This script intentionally does NOT include any indicators; it focuses on the core OHLCV+amount.
"""

from __future__ import annotations

import argparse
import io
import shutil
import zipfile
from pathlib import Path

import duckdb
import pandas as pd

# Reuse stock list mapping utilities
from prepare_akquant_data import load_stock_map, normalize_stock_code

CSV_ENCODINGS = ("utf-8", "utf-8-sig", "gb18030", "gbk")

PRICE_COL_MAP = {
    "日期": "date",
    "开盘": "open",
    "最高": "high",
    "最低": "low",
    "收盘": "close",
    "成交量": "volume",
    "成交额": "amount",
}


def read_csv_bytes(raw: bytes) -> pd.DataFrame:
    for enc in CSV_ENCODINGS:
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(io.BytesIO(raw))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build unadjusted daily bar Parquet lake (A股不复权)")
    p.add_argument(
        "--price-zip",
        type=Path,
        default=Path("A股数据_zip/daily.zip"),
        help="不复权日线 zip (default: A股数据_zip/daily.zip)",
    )
    p.add_argument(
        "--stock-list",
        type=Path,
        default=Path("A股数据_zip/股票列表.csv"),
        help="股票列表(含 TS代码) 用于 code->symbol 映射",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/lake/fact_bar_daily/adjust=none"),
        help="输出目录 (Hive partitioned parquet)",
    )
    p.add_argument(
        "--tmp-db",
        type=Path,
        default=Path("data/duckdb/_ingest_raw_daily.duckdb"),
        help="临时 DuckDB 文件，用于 ingest 后 COPY 输出 parquet",
    )
    p.add_argument("--limit", type=int, default=None, help="仅处理前 N 只股票（自检用）")
    p.add_argument("--overwrite", action="store_true", help="覆盖 out-dir 和 tmp-db")
    p.add_argument("--progress-every", type=int, default=200, help="进度打印间隔")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.price_zip.exists():
        raise FileNotFoundError(f"missing zip: {args.price_zip}")
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
        CREATE TABLE raw_daily (
          date DATE,
          symbol VARCHAR,
          open DOUBLE,
          high DOUBLE,
          low DOUBLE,
          close DOUBLE,
          volume DOUBLE,
          amount DOUBLE,
          year INTEGER,
          month INTEGER
        );
        """
    )

    inserted_symbols = 0
    skipped = 0

    with zipfile.ZipFile(args.price_zip) as zf:
        members = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        members = sorted(members)
        if args.limit is not None:
            members = members[: max(0, int(args.limit))]

        for idx, member in enumerate(members, start=1):
            # e.g. 000001_daily.csv
            stem = Path(member).stem
            try:
                code = normalize_stock_code(stem.split("_")[0])
            except Exception:
                skipped += 1
                continue

            symbol = stock_map.get(code)
            if not symbol:
                skipped += 1
                continue

            raw = zf.read(member)
            df = read_csv_bytes(raw)
            if df is None or df.empty:
                skipped += 1
                continue

            if not set(PRICE_COL_MAP).issubset(df.columns):
                skipped += 1
                continue

            out = df.rename(columns=PRICE_COL_MAP).copy()
            out = out[list(PRICE_COL_MAP.values())]
            out["date"] = pd.to_datetime(out["date"], errors="coerce")
            for c in ["open", "high", "low", "close", "volume", "amount"]:
                out[c] = pd.to_numeric(out[c], errors="coerce")
            out = out.dropna(subset=["date", "open", "high", "low", "close"])
            if out.empty:
                skipped += 1
                continue

            out["symbol"] = symbol
            out["year"] = out["date"].dt.year.astype("int32")
            out["month"] = out["date"].dt.month.astype("int8")

            con.register("tmp_df", out)
            con.execute(
                """
                INSERT INTO raw_daily
                SELECT
                  date::DATE as date,
                  symbol,
                  open, high, low, close,
                  volume, amount,
                  year, month
                FROM tmp_df;
                """
            )
            con.unregister("tmp_df")
            inserted_symbols += 1

            if idx % int(args.progress_every) == 0:
                print(f"[INFO] processed {idx}/{len(members)}; inserted_symbols={inserted_symbols}; skipped={skipped}")

    rows = con.execute("SELECT COUNT(*) FROM raw_daily").fetchone()[0]
    print(f"[INFO] ingest done: symbols={inserted_symbols}, skipped={skipped}, rows={rows}")

    out_dir = args.out_dir.resolve().as_posix()
    con.execute(
        f"""
        COPY (
          SELECT date, symbol, open, high, low, close, volume, amount, year, month
          FROM raw_daily
        )
        TO '{out_dir}'
        (FORMAT PARQUET, PARTITION_BY (year, month), COMPRESSION ZSTD);
        """
    )

    print(f"[DONE] wrote parquet lake -> {args.out_dir}")
    con.close()


if __name__ == "__main__":
    main()
