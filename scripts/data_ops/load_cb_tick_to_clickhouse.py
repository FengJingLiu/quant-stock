#!/usr/bin/env python3
"""
导入可转债/正股 tick 到 ClickHouse。

规则:
1) data/bondtick: 全量导入可转债 tick。
2) data/atick: 仅导入可转债基础信息中「正股代码」对应的 A 股 tick。

优化:
1) 每个 zip 先合并后一次批量 insert。
2) 按 zip 多进程并行导入，默认进程数=CPU 核心数。

用法:
  .venv/bin/python scripts/load_cb_tick_to_clickhouse.py
  .venv/bin/python scripts/load_cb_tick_to_clickhouse.py --init-only
  .venv/bin/python scripts/load_cb_tick_to_clickhouse.py --load-only --months 2024-01,2024-02
  .venv/bin/python scripts/load_cb_tick_to_clickhouse.py --truncate --recreate-table
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import io
import logging
import os
import sys
import zipfile
from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import CH_DATABASE
from src.data_clients import create_clickhouse_http_client

BOND_ROOT = Path("data/bondtick")
STOCK_ROOT = Path("data/atick")
CB_INFO_CSV = BOND_ROOT / "可转债基础信息列表.csv"

TABLE_CB = f"{CH_DATABASE}.tick_cb"
TABLE_STOCK = f"{CH_DATABASE}.tick_stock_cb_underlying"

INSERT_COLUMNS = [
    "trade_date",
    "ts",
    "symbol",
    "price",
    "volume",
    "side",
    "side_raw",
    "source_zip",
    "source_file",
]

SIDE_MAP = {
    "买入": 1,
    "买盘": 1,
    "卖出": -1,
    "卖盘": -1,
    "中性盘": 0,
}

_WORKER_STOCK_MAP: dict[str, str] = {}


def setup_logger() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
    )
    return logging.getLogger("load_cb_tick")


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).replace("\ufeff", "").strip() for c in out.columns]
    return out


def _read_csv_auto(buf: io.BytesIO) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            buf.seek(0)
            return pd.read_csv(buf, encoding=enc)
        except UnicodeDecodeError:
            continue
    buf.seek(0)
    return pd.read_csv(buf)


def _normalize_tick_df(
    df: pd.DataFrame,
    *,
    symbol: str,
    source_zip: str,
    source_file: str,
) -> pd.DataFrame:
    raw = _normalize_columns(df)
    time_col = "时间"
    price_col = "成交价"
    volume_col = "成交量" if "成交量" in raw.columns else "手数"
    side_col = "买卖方向"

    # 优先保留量价数据：方向列允许缺失，缺失时按中性盘处理。
    if not {time_col, price_col, volume_col}.issubset(raw.columns):
        return pd.DataFrame(columns=INSERT_COLUMNS)

    ts = pd.to_datetime(raw[time_col], errors="coerce")
    ts = ts.dt.tz_localize("Asia/Shanghai", ambiguous="NaT", nonexistent="shift_forward")
    price = pd.to_numeric(raw[price_col], errors="coerce")
    volume = pd.to_numeric(raw[volume_col], errors="coerce")
    if side_col in raw.columns:
        side_raw = raw[side_col].fillna("").astype(str).str.strip()
    else:
        side_raw = pd.Series([""] * len(raw), index=raw.index, dtype="string")
    side_raw = side_raw.replace("", "中性盘")

    out = pd.DataFrame(
        {
            "trade_date": ts.dt.date,
            "ts": ts,
            "symbol": symbol,
            "price": price.astype("float64"),
            "volume": volume.fillna(0).clip(lower=0).astype("int64"),
            "side": side_raw.map(SIDE_MAP).fillna(0).astype("int8"),
            "side_raw": side_raw,
            "source_zip": source_zip,
            "source_file": source_file,
        }
    )
    out = out.dropna(subset=["ts", "price"]).reset_index(drop=True)
    out = out[out["price"] > 0].reset_index(drop=True)
    out["volume"] = out["volume"].clip(upper=4_294_967_295).astype("uint32")
    return out


def create_tables(client: object, *, truncate: bool, recreate_table: bool) -> None:
    ddl_cb = f"""
    CREATE TABLE IF NOT EXISTS {TABLE_CB}
    (
        trade_date Date,
        ts DateTime64(3),
        symbol String,
        price Float64,
        volume UInt32,
        side Int8,
        side_raw LowCardinality(String),
        source_zip String,
        source_file String,
        ingest_time DateTime DEFAULT now()
    )
    ENGINE = MergeTree
    PARTITION BY toYYYYMM(trade_date)
    ORDER BY (symbol, ts)
    """
    ddl_stock = f"""
    CREATE TABLE IF NOT EXISTS {TABLE_STOCK}
    (
        trade_date Date,
        ts DateTime64(3),
        symbol String,
        price Float64,
        volume UInt32,
        side Int8,
        side_raw LowCardinality(String),
        source_zip String,
        source_file String,
        ingest_time DateTime DEFAULT now()
    )
    ENGINE = MergeTree
    PARTITION BY toYYYYMM(trade_date)
    ORDER BY (symbol, ts)
    """

    if recreate_table:
        client.command(f"DROP TABLE IF EXISTS {TABLE_CB}")
        client.command(f"DROP TABLE IF EXISTS {TABLE_STOCK}")

    client.command(ddl_cb)
    client.command(ddl_stock)

    if truncate:
        client.command(f"TRUNCATE TABLE {TABLE_CB}")
        client.command(f"TRUNCATE TABLE {TABLE_STOCK}")


def _iter_month_dirs(root: Path, months: set[str] | None) -> list[Path]:
    dirs = [p for p in root.iterdir() if p.is_dir() and p.name[:4].isdigit() and "-" in p.name]
    dirs = sorted(dirs)
    if months is None:
        return dirs
    return [p for p in dirs if p.name in months]


def load_underlying_symbol_map() -> tuple[dict[str, str], list[str]]:
    df = pd.read_csv(CB_INFO_CSV, dtype=str, usecols=["正股代码"])
    series = df["正股代码"].fillna("").astype(str).str.strip().str.upper()
    series = series[series.str.contains(r"^\d{6}\.(?:SH|SZ|BJ)$", regex=True)]

    map6: dict[str, str] = {}
    conflicts: list[str] = []
    for full_symbol in sorted(set(series.tolist())):
        code6 = full_symbol.split(".", 1)[0]
        old = map6.get(code6)
        if old is None:
            map6[code6] = full_symbol
        elif old != full_symbol:
            conflicts.append(code6)
    return map6, sorted(set(conflicts))


def _insert_df(client: object, table: str, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    client.insert_df(table, df[INSERT_COLUMNS])
    return len(df)


def _iter_zip_files(month_dirs: Iterable[Path], max_zips: int | None) -> list[Path]:
    zips: list[Path] = []
    for mdir in month_dirs:
        zips.extend(sorted(mdir.glob("*.zip")))
    zips = sorted(zips)
    if max_zips is not None and max_zips >= 0:
        return zips[:max_zips]
    return zips


def _resolve_workers(workers: int, total_jobs: int) -> int:
    auto = os.cpu_count() or 1
    requested = auto if workers <= 0 else workers
    if total_jobs <= 0:
        return 1
    return max(1, min(requested, total_jobs))


def _read_zip_df_for_bond(zpath: Path) -> tuple[pd.DataFrame, int]:
    frames: list[pd.DataFrame] = []
    csv_count = 0
    with zipfile.ZipFile(zpath, "r") as zf:
        for name in zf.namelist():
            if not name.endswith(".csv"):
                continue
            csv_count += 1
            symbol = Path(name).stem.upper()
            with zf.open(name) as fp:
                raw = _read_csv_auto(io.BytesIO(fp.read()))
            out = _normalize_tick_df(
                raw,
                symbol=symbol,
                source_zip=zpath.name,
                source_file=name,
            )
            if not out.empty:
                frames.append(out)

    if not frames:
        return pd.DataFrame(columns=INSERT_COLUMNS), csv_count
    return pd.concat(frames, ignore_index=True), csv_count


def _read_zip_df_for_stock(zpath: Path) -> tuple[pd.DataFrame, int]:
    frames: list[pd.DataFrame] = []
    matched_files = 0
    with zipfile.ZipFile(zpath, "r") as zf:
        for name in zf.namelist():
            if not name.endswith(".csv"):
                continue
            code6 = Path(name).stem[:6]
            full_symbol = _WORKER_STOCK_MAP.get(code6)
            if full_symbol is None:
                continue
            matched_files += 1
            with zf.open(name) as fp:
                raw = _read_csv_auto(io.BytesIO(fp.read()))
            out = _normalize_tick_df(
                raw,
                symbol=full_symbol,
                source_zip=zpath.name,
                source_file=name,
            )
            if not out.empty:
                frames.append(out)

    if not frames:
        return pd.DataFrame(columns=INSERT_COLUMNS), matched_files
    return pd.concat(frames, ignore_index=True), matched_files


def _worker_init_stock_map(stock_map: dict[str, str]) -> None:
    global _WORKER_STOCK_MAP
    _WORKER_STOCK_MAP = stock_map


def _process_bond_zip(zpath_str: str) -> tuple[str, int, int, str | None]:
    zpath = Path(zpath_str)
    try:
        df, csv_count = _read_zip_df_for_bond(zpath)
        if df.empty:
            return zpath.name, 0, csv_count, None
        client = create_clickhouse_http_client()
        inserted = _insert_df(client, TABLE_CB, df)
        return zpath.name, inserted, csv_count, None
    except Exception as exc:
        return zpath.name, 0, 0, str(exc)


def _process_stock_zip(zpath_str: str) -> tuple[str, int, int, str | None]:
    zpath = Path(zpath_str)
    try:
        df, matched_files = _read_zip_df_for_stock(zpath)
        if df.empty:
            return zpath.name, 0, matched_files, None
        client = create_clickhouse_http_client()
        inserted = _insert_df(client, TABLE_STOCK, df)
        return zpath.name, inserted, matched_files, None
    except Exception as exc:
        return zpath.name, 0, 0, str(exc)


def load_bond_ticks(
    log: logging.Logger,
    *,
    months: set[str] | None,
    max_zips: int | None,
    workers: int,
) -> int:
    month_dirs = _iter_month_dirs(BOND_ROOT, months)
    zips = _iter_zip_files(month_dirs, max_zips)
    inserted = 0
    worker_count = _resolve_workers(workers, len(zips))

    log.info("[bond] months=%d zips=%d workers=%d", len(month_dirs), len(zips), worker_count)
    if not zips:
        return 0

    with cf.ProcessPoolExecutor(max_workers=worker_count) as pool:
        futures = [pool.submit(_process_bond_zip, str(zpath)) for zpath in zips]
        for i, fut in enumerate(cf.as_completed(futures), 1):
            zip_name, day_rows, csv_count, err = fut.result()
            if err:
                log.warning("[bond] failed %s: %s", zip_name, err)
                continue
            inserted += day_rows
            log.info("[bond] %4d/%d %-16s rows=%d files=%d", i, len(zips), zip_name, day_rows, csv_count)
    return inserted


def load_stock_ticks(
    log: logging.Logger,
    *,
    stock_map: dict[str, str],
    months: set[str] | None,
    max_zips: int | None,
    workers: int,
) -> tuple[int, int]:
    month_dirs = _iter_month_dirs(STOCK_ROOT, months)
    zips = _iter_zip_files(month_dirs, max_zips)
    inserted = 0
    matched_files = 0
    worker_count = _resolve_workers(workers, len(zips))

    log.info(
        "[stock] months=%d zips=%d underlyings=%d workers=%d",
        len(month_dirs),
        len(zips),
        len(stock_map),
        worker_count,
    )
    if not zips:
        return 0, 0

    with cf.ProcessPoolExecutor(
        max_workers=worker_count,
        initializer=_worker_init_stock_map,
        initargs=(stock_map,),
    ) as pool:
        futures = [pool.submit(_process_stock_zip, str(zpath)) for zpath in zips]
        for i, fut in enumerate(cf.as_completed(futures), 1):
            zip_name, day_rows, day_matched, err = fut.result()
            if err:
                log.warning("[stock] failed %s: %s", zip_name, err)
                continue
            inserted += day_rows
            matched_files += day_matched
            log.info("[stock] %4d/%d %-16s rows=%d files=%d", i, len(zips), zip_name, day_rows, day_matched)
    return inserted, matched_files


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Load convertible-bond related tick data to ClickHouse")
    ap.add_argument("--init-only", action="store_true", help="Only create tables")
    ap.add_argument("--load-only", action="store_true", help="Only load data")
    ap.add_argument("--truncate", action="store_true", help="Truncate both tables before loading")
    ap.add_argument("--recreate-table", action="store_true", help="Drop and recreate both tables")
    ap.add_argument("--skip-bond", action="store_true", help="Skip loading bond ticks")
    ap.add_argument("--skip-stock", action="store_true", help="Skip loading stock ticks")
    ap.add_argument("--months", type=str, default="", help="Comma-separated months, e.g. 2024-01,2024-02")
    ap.add_argument("--max-zips", type=int, default=None, help="Only process first N zip files for smoke test")
    ap.add_argument("--workers", type=int, default=0, help="并行进程数；0=CPU 核心数")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    log = setup_logger()
    client = create_clickhouse_http_client()

    months = {m.strip() for m in args.months.split(",") if m.strip()} or None
    stock_map, conflicts = load_underlying_symbol_map()

    if conflicts:
        log.warning("underlying code conflicts (ignored first map wins): %s", ",".join(conflicts[:20]))
    log.info("loaded underlying symbols: %d", len(stock_map))

    need_init = not args.load_only
    need_load = not args.init_only

    if need_init:
        create_tables(client, truncate=args.truncate, recreate_table=args.recreate_table)
        log.info("tables ready: %s, %s", TABLE_CB, TABLE_STOCK)

    if not need_load:
        return

    bond_rows = 0
    stock_rows = 0
    matched_files = 0

    if not args.skip_bond:
        bond_rows = load_bond_ticks(
            log,
            months=months,
            max_zips=args.max_zips,
            workers=args.workers,
        )

    if not args.skip_stock:
        stock_rows, matched_files = load_stock_ticks(
            log,
            stock_map=stock_map,
            months=months,
            max_zips=args.max_zips,
            workers=args.workers,
        )

    cnt_cb = client.query(f"SELECT count() FROM {TABLE_CB}").result_rows[0][0]
    cnt_st = client.query(f"SELECT count() FROM {TABLE_STOCK}").result_rows[0][0]
    dt_cb = client.query(f"SELECT min(ts), max(ts) FROM {TABLE_CB}").result_rows[0]
    dt_st = client.query(f"SELECT min(ts), max(ts) FROM {TABLE_STOCK}").result_rows[0]

    log.info("inserted rows: bond=%d, stock=%d", bond_rows, stock_rows)
    if not args.skip_stock:
        log.info("[stock] matched_csv_files=%d", matched_files)
    log.info("table stats: %s rows=%d range=[%s, %s]", TABLE_CB, cnt_cb, dt_cb[0], dt_cb[1])
    log.info("table stats: %s rows=%d range=[%s, %s]", TABLE_STOCK, cnt_st, dt_st[0], dt_st[1])


if __name__ == "__main__":
    main()
