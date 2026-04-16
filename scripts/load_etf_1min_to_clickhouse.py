#!/usr/bin/env python3
"""
ETF 1分钟K线写入 ClickHouse — astock 数据库

表结构:
  astock.klines_1m_etf    ETF 分钟数据
  astock.klines_1m_index  指数 分钟数据（预留，数据另行导入）
  astock.klines_1m_stock  股票 分钟数据（预留，数据另行导入）

数据源:
  历史 2005-2025:
    data/基金分钟数据/ETF分钟数据_汇总/ETF_1min_2005_2022.zip
    data/基金分钟数据/ETF分钟数据_汇总/ETF_1min_2023_2025.zip
  增量 2026:
    data/基金分钟数据/ETF_分钟数据/1分钟_按月归档/2026-*/日级 zip

用法:
  python scripts/load_etf_1min_to_clickhouse.py           # 全量（默认）
  python scripts/load_etf_1min_to_clickhouse.py --init    # 仅建库建表
  python scripts/load_etf_1min_to_clickhouse.py --hist    # 仅历史大包
  python scripts/load_etf_1min_to_clickhouse.py --inc     # 仅 2026 增量
"""

import argparse
import io
import logging
import sys
import zipfile
from pathlib import Path

import pandas as pd
from clickhouse_driver import Client

# ── 路径配置 ─────────────────────────────────────────────────────────────
BASE     = Path("/home/autumn/quant/stock/data/基金分钟数据")
HIST_ZIPS = [
    BASE / "ETF分钟数据_汇总" / "ETF_1min_2005_2022.zip",
    BASE / "ETF分钟数据_汇总" / "ETF_1min_2023_2025.zip",
]
INC_BASE = BASE / "ETF_分钟数据" / "1分钟_按月归档"

# ── ClickHouse 连接 ───────────────────────────────────────────────────────
CH_KW = dict(
    host="localhost",
    port=9000,
    user="default",
    password="***CH_PASSWORD***",
    compression="lz4",
    settings={"max_insert_block_size": 500_000},
)
DB = "astock"

# ── 批次大小 ──────────────────────────────────────────────────────────────
CHUNK = 200_000   # 读大 CSV 每块行数
BATCH = 500_000   # 每次 INSERT 行数

# ── 日志 ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ── DDL ──────────────────────────────────────────────────────────────────
_BODY = """\
(
    symbol     String   COMMENT '代码，如 510300.SH / 000300.SH / 000001.SZ',
    trade_date Date     COMMENT '交易日期，按月分区',
    datetime   DateTime COMMENT '1分钟K线时间戳',
    open       Float32,
    high       Float32,
    low        Float32,
    close      Float32,
    volume     Float64  COMMENT '成交量',
    amount     Float64  COMMENT '成交额（元）',
    vwap       Float32  COMMENT '量加权均价 amount/volume'
)
ENGINE = ReplacingMergeTree()
PARTITION BY toYYYYMM(trade_date)
ORDER BY (trade_date, datetime, symbol)
SETTINGS index_granularity = 8192"""

DDL_DB    = f"CREATE DATABASE IF NOT EXISTS {DB}"
DDL_ETF   = f"CREATE TABLE IF NOT EXISTS {DB}.klines_1m_etf   {_BODY}"
DDL_INDEX = f"CREATE TABLE IF NOT EXISTS {DB}.klines_1m_index {_BODY}"
DDL_STOCK = f"CREATE TABLE IF NOT EXISTS {DB}.klines_1m_stock {_BODY}"

# 插入列顺序（与 DDL 一致）
COLS = ["symbol", "trade_date", "datetime",
        "open", "high", "low", "close",
        "volume", "amount", "vwap"]


# ── 工具函数 ──────────────────────────────────────────────────────────────
def create_schema(client: Client) -> None:
    for ddl in (DDL_DB, DDL_ETF, DDL_INDEX, DDL_STOCK):
        client.execute(ddl)
    log.info("Schema OK: %s.klines_1m_{etf,index,stock}", DB)


def _read_zip_bytes(zf: zipfile.ZipFile, name: str) -> io.BytesIO:
    """一次性读取 zip 内文件字节，返回可重复读取的 BytesIO。"""
    return io.BytesIO(zf.read(name))


def _df_from_bytes(buf: io.BytesIO, chunksize: int | None = None):
    """从 BytesIO 解析 CSV，自动处理 UTF-8 / GBK 编码。"""
    try:
        return pd.read_csv(buf, chunksize=chunksize, low_memory=False)
    except UnicodeDecodeError:
        buf.seek(0)
        return pd.read_csv(buf, encoding="gbk", chunksize=chunksize, low_memory=False)


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """原始 CSV DataFrame → 标准插入格式。

    兼容两种时间格式:
      "2005-02-23 09:30:00"  (历史大包)
      "2026/01/05 09:30"     (2026 日级包，无秒)
    """
    # 先将时间解析到 df 本身，方便统一过滤
    df = df.copy()
    df["_dt"] = pd.to_datetime(df["时间"], errors="coerce")
    # 丢弃无法解析的行及价格缺失行
    df = df.dropna(subset=["_dt", "开盘价", "收盘价", "最高价", "最低价"]).reset_index(drop=True)

    vol = df["成交量"].astype("float64")
    amt = df["成交额"].astype("float64")
    # volume=0 时 vwap 用 close 填充
    vwap = (amt / vol.where(vol > 0)).fillna(df["收盘价"].astype("float64"))

    out = pd.DataFrame({
        "symbol":     df["代码"].astype(str),
        "trade_date": df["_dt"].dt.date,
        # tolist() 返回 pd.Timestamp（datetime 子类），不会出现 float nan
        "datetime":   df["_dt"].tolist(),
        "open":       df["开盘价"].astype("float32"),
        "high":       df["最高价"].astype("float32"),
        "low":        df["最低价"].astype("float32"),
        "close":      df["收盘价"].astype("float32"),
        "volume":     vol,
        "amount":     amt,
        "vwap":       vwap.astype("float32"),
    })
    # 去掉价格全零（停牌/异常行）
    return out[out["close"] > 0].reset_index(drop=True)


def _do_insert(client: Client, df: pd.DataFrame, table: str) -> int:
    if df.empty:
        return 0
    # columnar 格式插入性能最优
    data = [df[c].tolist() for c in COLS]
    client.execute(
        f"INSERT INTO {table} ({', '.join(COLS)}) VALUES",
        data,
        columnar=True,
        types_check=False,
    )
    return len(df)


def insert_batched(client: Client, df: pd.DataFrame, table: str) -> int:
    """按 BATCH 行分批写入，返回总行数。"""
    total = 0
    for start in range(0, len(df), BATCH):
        total += _do_insert(client, df.iloc[start : start + BATCH], table)
    return total


# ── 历史大包加载（2005-2025） ──────────────────────────────────────────────
def load_hist_zip(client: Client, zip_path: Path, table: str) -> None:
    """加载一个历史大 zip（每个内部 CSV = 一只 ETF 全历史数据）。"""
    log.info("[HIST] 开始 %s", zip_path.name)
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = sorted(n for n in zf.namelist() if n.endswith(".csv"))
        log.info("[HIST] 共 %d 只 ETF", len(names))

        for i, name in enumerate(names, 1):
            symbol = name.replace(".csv", "")
            try:
                # 读入内存再逐块解析（避免 ZipExtFile 不可 seek 的问题）
                buf = _read_zip_bytes(zf, name)
                total_rows = 0
                for chunk in _df_from_bytes(buf, chunksize=CHUNK):
                    tdf = transform(chunk)
                    total_rows += insert_batched(client, tdf, table)
                log.info("[HIST]  (%d/%d) %-16s  %d 行", i, len(names), symbol, total_rows)
            except Exception as exc:
                log.warning("[HIST]  (%d/%d) %-16s  失败: %s", i, len(names), symbol, exc)

    log.info("[HIST] 完成 %s", zip_path.name)


# ── 2026 增量加载 ─────────────────────────────────────────────────────────
def load_inc_2026(client: Client, table: str) -> None:
    """加载 2026 按月归档（每个日级 zip = 当天全量 ETF 分钟数据）。"""
    month_dirs = sorted(INC_BASE.glob("2026-*"))
    if not month_dirs:
        log.warning("[INC] 未找到 2026-* 目录: %s", INC_BASE)
        return

    log.info("[INC] 发现 %d 个月目录: %s", len(month_dirs),
             ", ".join(d.name for d in month_dirs))

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
                    rows = insert_batched(client, tdf, table)
                    log.info("[INC]  %-28s → %d 行", dzip.name, rows)
                else:
                    log.warning("[INC]  %s 无有效数据", dzip.name)

            except Exception as exc:
                log.warning("[INC]  %s 失败: %s", dzip.name, exc)

    log.info("[INC] 2026 增量写入完毕")


# ── 主入口 ────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="ETF 1分钟K线写入 ClickHouse")
    ap.add_argument("--init", action="store_true", help="仅建库建表，不写数据")
    ap.add_argument("--hist", action="store_true", help="仅加载历史大包 (2005-2025)")
    ap.add_argument("--inc",  action="store_true", help="仅加载 2026 增量")
    args = ap.parse_args()

    client = Client(**CH_KW)

    # 连通性检查
    try:
        ver = client.execute("SELECT version()")[0][0]
        log.info("ClickHouse 连通: v%s", ver)
    except Exception as exc:
        log.error("无法连接 ClickHouse: %s", exc)
        sys.exit(1)

    create_schema(client)

    if args.init:
        log.info("--init 模式，建表完毕，退出")
        return

    # 未指定 --hist / --inc 时两者都跑
    run_hist = args.hist or not (args.hist or args.inc)
    run_inc  = args.inc  or not (args.hist or args.inc)

    table = f"{DB}.klines_1m_etf"

    if run_hist:
        for zp in HIST_ZIPS:
            if not zp.exists():
                log.warning("[HIST] 文件不存在，跳过: %s", zp)
                continue
            load_hist_zip(client, zp, table)

    if run_inc:
        load_inc_2026(client, table)

    # 最终统计
    cnt = client.execute(f"SELECT count() FROM {table}")[0][0]
    log.info("完成！%s 共 {:,} 行".format(cnt), table)


if __name__ == "__main__":
    main()
