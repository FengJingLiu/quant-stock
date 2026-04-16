#!/usr/bin/env python3
"""
股票复权因子写入 ClickHouse — astock.adj_factor (fund_type='stock')

数据源:
  data/复权因子/
    复权因子_前复权.zip  (5805 只股票)
    复权因子_后复权.zip  (5824 只股票)
  每个 zip 内：每只股票一个 CSV，列：股票代码, 交易日期(YYYYMMDD 整数), 复权因子

注: 前复权因子值为负数（加法偏移），直接写入 Float64，无需处理。

用法:
  python scripts/load_stock_adj_factor_to_clickhouse.py
"""

import io
import logging
import sys
import zipfile
from pathlib import Path

import pandas as pd
from clickhouse_driver import Client

# ── 路径 ─────────────────────────────────────────────────────────────────
BASE = Path("/home/autumn/quant/stock/data/复权因子")
SOURCES = [
    (BASE / "复权因子_前复权.zip", "stock", "qfq"),
    (BASE / "复权因子_后复权.zip", "stock", "hfq"),
]

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
TABLE = f"{DB}.adj_factor"

BATCH       = 500_000   # 每次 INSERT 行数
FLUSH_EVERY = BATCH     # 缓冲多少行后触发写入

# ── 日志 ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

COLS = ["symbol", "trade_date", "adj_type", "fund_type", "factor"]


# ── 工具 ─────────────────────────────────────────────────────────────────
def _read_single_csv(zf: zipfile.ZipFile, name: str) -> "pd.DataFrame | None":
    try:
        buf = io.BytesIO(zf.read(name))
        try:
            return pd.read_csv(buf)
        except UnicodeDecodeError:
            buf.seek(0)
            return pd.read_csv(buf, encoding="gbk")
    except Exception as exc:
        log.warning("  读取失败 %s: %s", name, exc)
        return None


def _transform_and_insert(client: Client, frames: list, adj_type: str, fund_type: str) -> int:
    """合并 frames，转换列，批量写入 ClickHouse，返回写入行数。"""
    if not frames:
        return 0

    big = pd.concat(frames, ignore_index=True)

    # 统一列名：股票代码 → symbol，交易日期 → trade_date_raw，复权因子 → factor
    big = big.rename(columns={
        "股票代码":  "symbol",
        "交易日期":  "trade_date_raw",
        "复权因子":  "factor",
    })

    big["trade_date"] = pd.to_datetime(
        big["trade_date_raw"].astype(str), format="%Y%m%d", errors="coerce"
    ).dt.date
    big["adj_type"]  = adj_type
    big["fund_type"] = fund_type
    big["factor"]    = big["factor"].astype("float64")
    big = big.dropna(subset=["trade_date", "factor"]).sort_values("trade_date").reset_index(drop=True)

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
    return total


def load_zip(client: Client, zip_path: Path, fund_type: str, adj_type: str) -> int:
    """流式读取复权因子 zip，每积累 FLUSH_EVERY 行触发一次写入，减少内存峰值。"""
    log.info("[%s/%s] 开始 %s", fund_type.upper(), adj_type, zip_path.name)

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = sorted(n for n in zf.namelist() if n.endswith(".csv"))
        log.info("  共 %d 只股票", len(names))

        frames: list[pd.DataFrame] = []
        buf_rows = 0
        grand_total = 0

        for name in names:
            df = _read_single_csv(zf, name)
            if df is None or df.empty:
                continue
            frames.append(df)
            buf_rows += len(df)

            if buf_rows >= FLUSH_EVERY:
                written = _transform_and_insert(client, frames, adj_type, fund_type)
                grand_total += written
                log.info("  中途写入 {:,} 行 (累计 {:,})".format(written, grand_total))
                frames = []
                buf_rows = 0

        # 残余
        if frames:
            written = _transform_and_insert(client, frames, adj_type, fund_type)
            grand_total += written

    log.info("[%s/%s] 写入 {:,} 行".format(grand_total), fund_type.upper(), adj_type)
    return grand_total


# ── 主入口 ────────────────────────────────────────────────────────────────
def main() -> None:
    client = Client(**CH_KW)

    try:
        ver = client.execute("SELECT version()")[0][0]
        log.info("ClickHouse 连通: v%s", ver)
    except Exception as exc:
        log.error("无法连接 ClickHouse: %s", exc)
        sys.exit(1)

    # 确认表存在
    cnt_before = client.execute(f"SELECT count() FROM {TABLE}")[0][0]
    log.info("写入前 %s 共 {:,} 行".format(cnt_before), TABLE)

    grand_total = 0
    for zip_path, fund_type, adj_type in SOURCES:
        if not zip_path.exists():
            log.warning("文件不存在，跳过: %s", zip_path)
            continue
        grand_total += load_zip(client, zip_path, fund_type, adj_type)

    cnt_after = client.execute(f"SELECT count() FROM {TABLE}")[0][0]
    log.info("完成！%s 共 {:,} 行（本次写入 {:,} 行）".format(cnt_after, grand_total), TABLE)

    # 简单校验：查一只股票
    sample = client.execute(
        f"SELECT symbol, trade_date, adj_type, fund_type, factor FROM {TABLE} "
        f"WHERE symbol='000001.SZ' ORDER BY trade_date LIMIT 3"
    )
    if sample:
        log.info("样本数据 (000001.SZ):")
        for row in sample:
            log.info("  %s", row)


if __name__ == "__main__":
    main()
