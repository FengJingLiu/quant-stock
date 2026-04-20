#!/usr/bin/env python3
"""Build parquet lake for A-share daily indicators (excluding OHLC/volume/pct_chg).

Input
- A股数据_每日指标/每日指标.zip
- A股数据_每日指标/每日指标_退市.zip (optional; enabled by default)

Output (Hive partitions)
- data/lake/fact_indicator_daily/year=YYYY/month=MM/*.parquet
"""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

import duckdb
import pandas as pd

try:
    from prepare_akquant_data import INDICATOR_COLUMN_MAP, clean_indicator_frame, read_csv_bytes
except ModuleNotFoundError:  # pragma: no cover
    from scripts.prepare_akquant_data import (  # type: ignore
        INDICATOR_COLUMN_MAP,
        clean_indicator_frame,
        read_csv_bytes,
    )

INDICATOR_VALUE_COLUMNS = [
    col
    for col in INDICATOR_COLUMN_MAP.values()
    if col not in {"symbol", "trade_date", "pct_chg"}
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build daily indicator parquet lake")
    p.add_argument(
        "--indicator-zip",
        type=Path,
        default=Path("A股数据_每日指标/每日指标.zip"),
        help="每日指标 zip",
    )
    p.add_argument(
        "--delisted-zip",
        type=Path,
        default=Path("A股数据_每日指标/每日指标_退市.zip"),
        help="退市每日指标 zip",
    )
    p.add_argument(
        "--include-delisted",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否同时 ingest 退市股票数据（默认开启）",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/lake/fact_indicator_daily"),
        help="输出目录 (Hive partitioned parquet)",
    )
    p.add_argument(
        "--tmp-db",
        type=Path,
        default=Path("data/duckdb/_ingest_indicator_daily.duckdb"),
        help="临时 DuckDB 文件",
    )
    p.add_argument("--limit", type=int, default=None, help="仅处理前 N 个成员（自检用）")
    p.add_argument("--overwrite", action="store_true", help="覆盖 out-dir 和 tmp-db")
    p.add_argument("--progress-every", type=int, default=200, help="进度打印间隔")
    return p.parse_args()


def parse_symbol_from_member(member: str) -> str:
    stem = Path(member).stem.strip()
    if not stem:
        raise ValueError(f"invalid member name: {member}")
    return stem.upper()


def _prepare_output(out_dir: Path, tmp_db: Path, overwrite: bool) -> None:
    if overwrite:
        if out_dir.exists():
            shutil.rmtree(out_dir)
        if tmp_db.exists():
            tmp_db.unlink()
    else:
        if out_dir.exists() and any(out_dir.rglob("*.parquet")):
            raise FileExistsError(f"out-dir not empty: {out_dir} (use --overwrite)")

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_db.parent.mkdir(parents=True, exist_ok=True)


def build_indicator_daily_parquet(
    indicator_zip: Path,
    delisted_zip: Path,
    out_dir: Path,
    tmp_db: Path,
    include_delisted: bool,
    limit: int | None,
    overwrite: bool,
    progress_every: int,
) -> dict[str, int]:
    if not indicator_zip.exists():
        raise FileNotFoundError(f"missing indicator zip: {indicator_zip}")

    source_zips: list[tuple[Path, int, str]] = [(indicator_zip, 1, "active")]
    if include_delisted:
        if delisted_zip.exists():
            source_zips.append((delisted_zip, 2, "delisted"))
        else:
            print(f"[WARN] delisted zip not found, skip: {delisted_zip}")

    _prepare_output(out_dir, tmp_db, overwrite)

    indicator_schema_sql = ",\n          ".join(f"{c} DOUBLE" for c in INDICATOR_VALUE_COLUMNS)
    indicator_insert_cols_sql = ", ".join(INDICATOR_VALUE_COLUMNS)

    con = duckdb.connect(tmp_db.as_posix())
    con.execute("PRAGMA threads=8")
    con.execute(
        f"""
        CREATE TABLE indicator_daily (
          date DATE,
          symbol VARCHAR,
          {indicator_schema_sql},
          year INTEGER,
          month INTEGER,
          source_rank INTEGER
        );
        """
    )

    stats = {
        "processed_members": 0,
        "inserted_members": 0,
        "inserted_rows": 0,
        "skipped_empty": 0,
        "skipped_missing_cols": 0,
        "skipped_parse_fail": 0,
    }

    limit_n = None if limit is None else max(0, int(limit))
    should_stop = False

    try:
        for zip_path, source_rank, source_name in source_zips:
            with zipfile.ZipFile(zip_path) as zf:
                members = sorted(n for n in zf.namelist() if n.lower().endswith(".csv"))

                for member in members:
                    if limit_n is not None and stats["processed_members"] >= limit_n:
                        should_stop = True
                        break

                    stats["processed_members"] += 1

                    try:
                        raw_df = read_csv_bytes(zf.read(member))
                        if raw_df is None or raw_df.empty:
                            stats["skipped_empty"] += 1
                            continue

                        symbol = parse_symbol_from_member(member)
                        cleaned = clean_indicator_frame(raw_df, force_symbol=symbol)
                    except Exception as exc:  # noqa: BLE001
                        stats["skipped_parse_fail"] += 1
                        print(f"[WARN] skip member parse fail: {source_name}/{member} -> {exc}")
                        continue

                    if cleaned.empty:
                        stats["skipped_empty"] += 1
                        continue

                    missing_cols = [c for c in INDICATOR_VALUE_COLUMNS if c not in cleaned.columns]
                    if missing_cols:
                        stats["skipped_missing_cols"] += 1
                        print(
                            f"[WARN] skip member missing cols: {source_name}/{member} "
                            f"missing={len(missing_cols)}"
                        )
                        continue

                    out = cleaned[["timestamp", "symbol", *INDICATOR_VALUE_COLUMNS]].copy()
                    ts = pd.to_datetime(out["timestamp"], errors="coerce")
                    out["date"] = ts.dt.date
                    out["year"] = ts.dt.year.astype("Int32")
                    out["month"] = ts.dt.month.astype("Int8")
                    out["source_rank"] = int(source_rank)
                    out = out.drop(columns=["timestamp"]).dropna(subset=["date", "symbol", "year", "month"])

                    if out.empty:
                        stats["skipped_empty"] += 1
                        continue

                    con.register("tmp_df", out)
                    con.execute(
                        f"""
                        INSERT INTO indicator_daily
                        SELECT
                          date::DATE AS date,
                          symbol,
                          {indicator_insert_cols_sql},
                          year,
                          month,
                          source_rank
                        FROM tmp_df;
                        """
                    )
                    con.unregister("tmp_df")

                    stats["inserted_members"] += 1
                    stats["inserted_rows"] += int(len(out))

                    pe = max(1, int(progress_every))
                    if stats["processed_members"] % pe == 0:
                        print(
                            "[INFO] processed="
                            f"{stats['processed_members']}, "
                            f"inserted_members={stats['inserted_members']}, "
                            f"skipped_empty={stats['skipped_empty']}, "
                            f"skipped_missing_cols={stats['skipped_missing_cols']}, "
                            f"skipped_parse_fail={stats['skipped_parse_fail']}"
                        )

            if should_stop:
                break

        rows_ingested = con.execute("SELECT COUNT(*) FROM indicator_daily").fetchone()[0]
        print(
            "[INFO] ingest done: "
            f"members={stats['processed_members']}, "
            f"inserted_members={stats['inserted_members']}, "
            f"rows={rows_ingested}, "
            f"skipped_empty={stats['skipped_empty']}, "
            f"skipped_missing_cols={stats['skipped_missing_cols']}, "
            f"skipped_parse_fail={stats['skipped_parse_fail']}"
        )

        out_dir_posix = out_dir.resolve().as_posix()
        copy_cols = ", ".join(["date", "symbol", *INDICATOR_VALUE_COLUMNS, "year", "month"])
        con.execute(
            f"""
            COPY (
              SELECT {copy_cols}
              FROM (
                SELECT *,
                       ROW_NUMBER() OVER (
                         PARTITION BY date, symbol
                         ORDER BY source_rank DESC
                       ) AS rn
                FROM indicator_daily
              ) t
              WHERE rn = 1
            )
            TO '{out_dir_posix}'
            (FORMAT PARQUET, PARTITION_BY (year, month), COMPRESSION ZSTD);
            """
        )

        stats["output_rows"] = con.execute(
            """
            SELECT COUNT(*)
            FROM (
              SELECT date, symbol
              FROM indicator_daily
              QUALIFY ROW_NUMBER() OVER (PARTITION BY date, symbol ORDER BY source_rank DESC) = 1
            )
            """
        ).fetchone()[0]
        print(f"[DONE] wrote parquet indicator lake -> {out_dir}")
        return stats
    finally:
        con.close()


def main() -> None:
    args = parse_args()
    build_indicator_daily_parquet(
        indicator_zip=args.indicator_zip,
        delisted_zip=args.delisted_zip,
        out_dir=args.out_dir,
        tmp_db=args.tmp_db,
        include_delisted=args.include_delisted,
        limit=args.limit,
        overwrite=args.overwrite,
        progress_every=args.progress_every,
    )


if __name__ == "__main__":
    main()
