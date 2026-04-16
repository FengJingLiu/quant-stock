#!/usr/bin/env python3
"""
股票 1分钟K线写入 ClickHouse — astock.klines_1m_stock

数据源:
  历史 2000-2025:
    data/沪深/1分钟_按年汇总/2000_1min.zip … 2025_1min.zip (26 个年包)
    每 zip 约 5181 只股票，文件名：sh600000_2025.csv
  增量 2026:
    data/沪深/2026-01/20260105_1min.zip 等（按月归档的日级 zip）
    文件名：sh600000.csv

CSV 列:
  时间,代码,名称,开盘价,收盘价,最高价,最低价,成交量,成交额,涨幅,振幅

代码格式:
  sh600000 / sz000001（带交易所前缀，直接写入，不做转换）

时间格式:
  "2025-01-02 09:30:00"  (历史年包)
  "2026/01/05 09:30"     (2026 日包，无秒)

用法:
  python scripts/load_stock_1min_to_clickhouse.py           # 全量
  python scripts/load_stock_1min_to_clickhouse.py --hist    # 仅历史年包
  python scripts/load_stock_1min_to_clickhouse.py --inc     # 仅 2026 增量
  python scripts/load_stock_1min_to_clickhouse.py --hist --year 2025  # 单年测试
"""

import argparse
import io
import logging
import sys
import zipfile
from pathlib import Path

import pandas as pd
from clickhouse_driver import Client

# ── 路径 ─────────────────────────────────────────────────────────────────
BASE      = Path("/home/autumn/quant/stock/data/沪深")
HIST_BASE = BASE / "1分钟_按年汇总"

# ── ClickHouse ────────────────────────────────────────────────────────────
CH_KW = dict(
    host="localhost",
    port=9000,
    user="default",
    password="***CH_PASSWORD***",
    compression="lz4",
    settings={"max_insert_block_size": 500_000},
)
DB    = "astock"
TABLE = f"{DB}.klines_1m_stock"

CHUNK = 200_000
BATCH = 500_000

# ── 日志 ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

COLS = ["symbol", "trade_date", "datetime",
        "open", "high", "low", "close",
        "volume", "amount", "vwap"]


# ── 工具 ─────────────────────────────────────────────────────────────────
def _read_zip_bytes(zf: zipfile.ZipFile, name: str) -> io.BytesIO:
    return io.BytesIO(zf.read(name))


def _df_from_bytes(buf: io.BytesIO, chunksize: "int | None" = None):
    try:
        return pd.read_csv(buf, chunksize=chunksize, low_memory=False)
    except UnicodeDecodeError:
        buf.seek(0)
        return pd.read_csv(buf, encoding="gbk", chunksize=chunksize, low_memory=False)


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    原始股票 CSV → 标准插入格式。

    兼容时间格式:
      "2025-01-02 09:30:00"  (历史年包)
      "2026/01/05 09:30"     (2026 日包，无秒)

    代码: sh600000 / sz000001，原样保留。
    """
    df = df.copy()
    df["_dt"] = pd.to_datetime(df["时间"], errors="coerce")
    df = df.dropna(subset=["_dt", "开盘价", "收盘价", "最高价", "最低价"]).reset_index(drop=True)

    vol = df["成交量"].astype("float64")
    amt = df["成交额"].astype("float64")
    vwap = (amt / vol.where(vol > 0)).fillna(df["收盘价"].astype("float64"))

    out = pd.DataFrame({
        "symbol":     df["代码"].astype(str),
        "trade_date": df["_dt"].dt.date,
        "datetime":   df["_dt"].tolist(),
        "open":       df["开盘价"].astype("float32"),
        "high":       df["最高价"].astype("float32"),
        "low":        df["最低价"].astype("float32"),
        "close":      df["收盘价"].astype("float32"),
        "volume":     vol,
        "amount":     amt,
        "vwap":       vwap.astype("float32"),
    })
    return out[out["close"] > 0].reset_index(drop=True)


def _do_insert(client: Client, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    data = [df[c].tolist() for c in COLS]
    client.execute(
        f"INSERT INTO {TABLE} ({', '.join(COLS)}) VALUES",
        data,
        columnar=True,
        types_check=False,
    )
    return len(df)


def insert_batched(client: Client, df: pd.DataFrame) -> int:
    total = 0
    for start in range(0, len(df), BATCH):
        total += _do_insert(client, df.iloc[start : start + BATCH])
    return total


# ── 历史年包 ──────────────────────────────────────────────────────────────
def load_hist(client: Client, only_year: "int | None" = None) -> None:
    if only_year is not None:
        year_zips = sorted(HIST_BASE.glob(f"{only_year}_1min.zip"))
    else:
        year_zips = sorted(HIST_BASE.glob("*_1min.zip"))

    if not year_zips:
        log.warning("[HIST] 未找到年包: %s", HIST_BASE)
        return
    log.info("[HIST] 共 %d 个年包", len(year_zips))

    for zp in year_zips:
        log.info("[HIST] 开始 %s （预计每只股票约 1-7 秒）", zp.name)
        with zipfile.ZipFile(zp, "r") as zf:
            names = sorted(n for n in zf.namelist() if n.endswith(".csv"))
            log.info("[HIST]   %d 只股票", len(names))

            for i, name in enumerate(names, 1):
                # sh600000_2025.csv → sh600000
                symbol_hint = name.rsplit("_", 1)[0] if "_" in name else name.replace(".csv", "")
                try:
                    buf = _read_zip_bytes(zf, name)
                    total_rows = 0
                    for chunk in _df_from_bytes(buf, chunksize=CHUNK):
                        tdf = transform(chunk)
                        total_rows += insert_batched(client, tdf)
                    log.info("[HIST]   (%d/%d) %-12s  %d 行", i, len(names), symbol_hint, total_rows)
                except Exception as exc:
                    log.warning("[HIST]   (%d/%d) %-12s  失败: %s", i, len(names), symbol_hint, exc)

        log.info("[HIST] 完成 %s", zp.name)


# ── 2026 增量 ─────────────────────────────────────────────────────────────
def load_inc(client: Client) -> None:
    month_dirs = sorted(BASE.glob("2026-*"))
    if not month_dirs:
        log.warning("[INC] 未找到 2026-* 目录: %s", BASE)
        return

    log.info("[INC] 发现 %d 个月目录", len(month_dirs))

    for mdir in month_dirs:
        day_zips = sorted(mdir.glob("*.zip"))
        log.info("[INC] %s: %d 个日级 zip", mdir.name, len(day_zips))

        for dzip in day_zips:
            try:
                frames: list[pd.DataFrame] = []
                with zipfile.ZipFile(dzip, "r") as zf:
                    for name in zf.namelist():
                        if not name.endswith(".csv"):
                            continue
                        buf = _read_zip_bytes(zf, name)
                        df = _df_from_bytes(buf)
                        if isinstance(df, pd.DataFrame) and not df.empty:
                            frames.append(df)

                if frames:
                    big = pd.concat(frames, ignore_index=True)
                    tdf = transform(big)
                    rows = insert_batched(client, tdf)
                    log.info("[INC]  %-28s → %d 行", dzip.name, rows)
                else:
                    log.warning("[INC]  %s 无有效数据", dzip.name)
            except Exception as exc:
                log.warning("[INC]  %s 失败: %s", dzip.name, exc)

    log.info("[INC] 2026 增量写入完毕")


# ── 主入口 ────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="股票 1分钟K线写入 ClickHouse")
    ap.add_argument("--hist", action="store_true", help="仅加载历史年包 (2000-2025)")
    ap.add_argument("--inc",  action="store_true", help="仅加载 2026 增量")
    ap.add_argument("--year", type=int, default=None,
                    help="配合 --hist 使用，只处理指定年份，用于测试（如 --hist --year 2025）")
    args = ap.parse_args()

    client = Client(**CH_KW)

    try:
        ver = client.execute("SELECT version()")[0][0]
        log.info("ClickHouse 连通: v%s", ver)
    except Exception as exc:
        log.error("无法连接 ClickHouse: %s", exc)
        sys.exit(1)

    run_hist = args.hist or not (args.hist or args.inc)
    run_inc  = args.inc  or not (args.hist or args.inc)

    if run_hist:
        load_hist(client, only_year=args.year)

    if run_inc:
        load_inc(client)

    cnt = client.execute(f"SELECT count() FROM {TABLE}")[0][0]
    log.info("完成！%s 共 {:,} 行".format(cnt), TABLE)


if __name__ == "__main__":
    main()
