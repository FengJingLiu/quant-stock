#!/usr/bin/env python3
"""
复权因子写入 ClickHouse — astock.adj_factor

数据源:
  data/基金分钟数据/基金_复权因子/
    ETF_复权因子_前复权.zip  (1480 只 ETF)
    ETF_复权因子_后复权.zip  (1480 只 ETF)
    LOF_复权因子_前复权.zip  ( 404 只 LOF)
    LOF_复权因子_后复权.zip  ( 404 只 LOF)

  每个 zip 内部：每只基金一个 CSV，列：基金代码/股票代码, 交易日期(YYYYMMDD), 复权因子

表结构:
  astock.adj_factor
    symbol       String    — 基金/LOF 代码
    trade_date   Date      — 交易日期（分区键）
    adj_type     LowCardinality(String)  — 'qfq' 前复权 | 'hfq' 后复权
    fund_type    LowCardinality(String)  — 'etf' | 'lof'
    factor       Float64   — 复权因子

用法:
  python scripts/load_adj_factor_to_clickhouse.py
"""

import io
import logging
import os
import sys
import zipfile
from pathlib import Path

import pandas as pd
from clickhouse_driver import Client

import _load_env  # noqa: F401

# ── 路径 ─────────────────────────────────────────────────────────────────
BASE = Path("/home/autumn/quant/stock/data/基金分钟数据/基金_复权因子")
SOURCES = [
    (BASE / "ETF_复权因子_前复权.zip", "etf", "qfq"),
    (BASE / "ETF_复权因子_后复权.zip", "etf", "hfq"),
    (BASE / "LOF_复权因子_前复权.zip", "lof", "qfq"),
    (BASE / "LOF_复权因子_后复权.zip", "lof", "hfq"),
]

# ── ClickHouse ────────────────────────────────────────────────────────────
CH_KW = dict(
    host="localhost",
    port=9000,
    user="default",
    password=os.environ["CH_PASSWORD"],
    compression="lz4",
    settings={"max_insert_block_size": 500_000},
)
DB    = "astock"
TABLE = f"{DB}.adj_factor"

BATCH = 500_000

# ── 日志 ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ── DDL ──────────────────────────────────────────────────────────────────
DDL = f"""
CREATE TABLE IF NOT EXISTS {TABLE}
(
    symbol     String                   COMMENT '基金/LOF 代码，如 510050.SH',
    trade_date Date                     COMMENT '交易日期',
    adj_type   LowCardinality(String)   COMMENT '复权类型: qfq=前复权, hfq=后复权',
    fund_type  LowCardinality(String)   COMMENT '品种类型: etf | lof',
    factor     Float64                  COMMENT '复权因子'
)
ENGINE = ReplacingMergeTree()
ORDER BY (trade_date, symbol, adj_type)
SETTINGS index_granularity = 8192
"""

COLS = ["symbol", "trade_date", "adj_type", "fund_type", "factor"]


# ── 工具 ─────────────────────────────────────────────────────────────────
def create_table(client: Client) -> None:
    client.execute(f"CREATE DATABASE IF NOT EXISTS {DB}")
    client.execute(DDL)
    log.info("Table OK: %s", TABLE)


def _read_single_csv(zf: zipfile.ZipFile, name: str) -> pd.DataFrame | None:
    """读取 zip 内单只基金的 CSV，返回 DataFrame 或 None。"""
    try:
        buf = io.BytesIO(zf.read(name))
        try:
            df = pd.read_csv(buf)
        except UnicodeDecodeError:
            buf.seek(0)
            df = pd.read_csv(buf, encoding="gbk")
        return df
    except Exception as exc:
        log.warning("  读取失败 %s: %s", name, exc)
        return None


def load_zip(client: Client, zip_path: Path, fund_type: str, adj_type: str) -> int:
    """读取一个复权因子 zip，整体拼接后批量写入。"""
    log.info("[%s/%s] 开始 %s", fund_type.upper(), adj_type, zip_path.name)

    frames: list[pd.DataFrame] = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = sorted(n for n in zf.namelist() if n.endswith(".csv"))
        log.info("  共 %d 只", len(names))
        for name in names:
            df = _read_single_csv(zf, name)
            if df is None or df.empty:
                continue
            frames.append(df)

    if not frames:
        log.warning("  无有效数据，跳过")
        return 0

    big = pd.concat(frames, ignore_index=True)

    # 列名兼容：ETF 用"基金代码"，LOF 用"股票代码"
    code_col = "基金代码" if "基金代码" in big.columns else "股票代码"
    big = big.rename(columns={code_col: "symbol", "交易日期": "trade_date_raw", "复权因子": "factor"})

    big["trade_date"] = pd.to_datetime(big["trade_date_raw"].astype(str), format="%Y%m%d", errors="coerce").dt.date
    big["adj_type"]  = adj_type
    big["fund_type"] = fund_type
    big["factor"]    = big["factor"].astype("float64")
    big = big.dropna(subset=["trade_date", "factor"])
    # 按日期排序，确保每批次跨越的分区数尽量少
    big = big.sort_values("trade_date").reset_index(drop=True)

    # 分批写入
    total = 0
    for start in range(0, len(big), BATCH):
        chunk = big.iloc[start : start + BATCH]
        data  = [chunk[c].tolist() for c in COLS]
        client.execute(
            f"INSERT INTO {TABLE} ({', '.join(COLS)}) VALUES",
            data,
            columnar=True,
            types_check=False,
        )
        total += len(chunk)

    log.info("  写入 {:,} 行".format(total))
    return total


# ── 主入口 ────────────────────────────────────────────────────────────────
def main() -> None:
    client = Client(**CH_KW)

    try:
        ver = client.execute("SELECT version()")[0][0]
        log.info("ClickHouse 连通: v%s", ver)
    except Exception as exc:
        log.error("无法连接 ClickHouse: %s", exc)
        sys.exit(1)

    create_table(client)

    grand_total = 0
    for zip_path, fund_type, adj_type in SOURCES:
        if not zip_path.exists():
            log.warning("文件不存在，跳过: %s", zip_path)
            continue
        grand_total += load_zip(client, zip_path, fund_type, adj_type)

    cnt = client.execute(f"SELECT count() FROM {TABLE}")[0][0]
    log.info("完成！%s 共 {:,} 行（本次写入 {:,} 行）".format(cnt, grand_total), TABLE)

    # 简单校验
    sample = client.execute(
        f"SELECT symbol, trade_date, adj_type, fund_type, factor FROM {TABLE} "
        f"WHERE symbol='510050.SH' ORDER BY trade_date LIMIT 3"
    )
    if sample:
        log.info("样本数据 (510050.SH):")
        for row in sample:
            log.info("  %s", row)


if __name__ == "__main__":
    main()
