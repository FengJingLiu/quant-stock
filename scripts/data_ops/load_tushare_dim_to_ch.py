"""
从 Tushare 加载维度表到 ClickHouse

生成表:
  astock.dim_stock_info       — 股票基础信息 (板块→涨跌停幅度)
  astock.dim_namechange       — 股票更名/ST 历史 (PIT)
  astock.dim_suspend_daily    — 日度停牌记录
  astock.dim_stk_limit_daily  — 日度涨跌停价 (按年批量拉取)
  astock.dim_index_weights    — 指数成分权重 (PIT)

用法:
  python scripts/load_tushare_dim_to_ch.py                    # 全部加载
  python scripts/load_tushare_dim_to_ch.py --table stock_info # 单表
  python scripts/load_tushare_dim_to_ch.py --table stk_limit --start-year 2020  # stk_limit 指定起始年
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date, datetime
from pathlib import Path

import pandas as pd

# ── 项目路径 ──────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data_clients import create_clickhouse_http_client, get_tushare_pro


def get_ch():
    return create_clickhouse_http_client()


# ── 工具函数 ──────────────────────────────────────────────────

def _ts_call(func, max_retry=3, sleep_between=0.3, **kwargs):
    """带重试的 Tushare API 调用。"""
    for attempt in range(max_retry):
        try:
            df = func(**kwargs)
            time.sleep(sleep_between)
            return df
        except Exception as e:
            if attempt == max_retry - 1:
                raise
            print(f"  retry {attempt+1}/{max_retry}: {e}")
            time.sleep(2 ** attempt)
    return pd.DataFrame()


def _insert_df(ch, table: str, df: pd.DataFrame):
    """将 pandas DataFrame 写入 ClickHouse (自动处理列顺序)。"""
    if df.empty:
        return
    ch.insert_df(table, df)


# ══════════════════════════════════════════════════════════════
#  1. dim_stock_info  —  股票基础信息 + 涨跌停幅度
# ══════════════════════════════════════════════════════════════

DDL_STOCK_INFO = """
CREATE TABLE IF NOT EXISTS astock.dim_stock_info (
    ts_code    String,
    symbol     String,
    name       String,
    market     String      COMMENT '主板/创业板/科创板/北交所',
    exchange   String      COMMENT 'SSE/SZSE/BSE',
    list_date  Nullable(Date),
    delist_date Nullable(Date),
    is_hs      String      COMMENT 'H/S/N',
    industry   String,
    area       String,
    limit_pct  Float32     COMMENT '涨跌停基准幅度 0.10/0.20/0.30'
) ENGINE = ReplacingMergeTree()
ORDER BY ts_code
"""

MARKET_LIMIT = {"主板": 0.10, "创业板": 0.20, "科创板": 0.20, "北交所": 0.30}


def _derive_exchange(ts_code: str) -> str:
    if ts_code.endswith(".SH"):
        return "SSE"
    elif ts_code.endswith(".SZ"):
        return "SZSE"
    elif ts_code.endswith(".BJ"):
        return "BSE"
    return ""


def load_stock_info():
    print("=" * 60)
    print("[1/5] dim_stock_info — 股票基础信息")
    pro = get_tushare_pro()
    ch = get_ch()

    ch.command(DDL_STOCK_INFO)

    frames = []
    for status, label in [("L", "上市"), ("D", "退市"), ("P", "暂停上市")]:
        df = _ts_call(
            pro.stock_basic,
            exchange="",
            list_status=status,
            fields="ts_code,symbol,name,area,industry,market,list_date,delist_date,is_hs",
        )
        print(f"  {label}: {len(df)} rows")
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)

    # 派生字段
    df["exchange"] = df["ts_code"].apply(_derive_exchange)
    df["limit_pct"] = df["market"].map(MARKET_LIMIT).fillna(0.10).astype("float32")
    df["list_date"] = pd.to_datetime(df["list_date"], format="%Y%m%d", errors="coerce")
    df["delist_date"] = pd.to_datetime(df["delist_date"], format="%Y%m%d", errors="coerce")
    df["is_hs"] = df["is_hs"].fillna("N")
    df["industry"] = df["industry"].fillna("")
    df["area"] = df["area"].fillna("")

    # 清空后写入
    ch.command("TRUNCATE TABLE astock.dim_stock_info")
    _insert_df(ch, "astock.dim_stock_info", df[
        ["ts_code", "symbol", "name", "market", "exchange",
         "list_date", "delist_date", "is_hs", "industry", "area", "limit_pct"]
    ])
    print(f"  → dim_stock_info: {len(df)} rows written")
    return len(df)


# ══════════════════════════════════════════════════════════════
#  2. dim_namechange  —  股票更名/ST历史 (PIT)
# ══════════════════════════════════════════════════════════════

DDL_NAMECHANGE = """
CREATE TABLE IF NOT EXISTS astock.dim_namechange (
    ts_code       String,
    name          String,
    start_date    Date,
    end_date      Nullable(Date),
    change_reason String
) ENGINE = ReplacingMergeTree()
ORDER BY (ts_code, start_date)
"""


def load_namechange():
    print("=" * 60)
    print("[2/5] dim_namechange — 股票更名/ST历史")
    pro = get_tushare_pro()
    ch = get_ch()

    ch.command(DDL_NAMECHANGE)

    # Tushare namechange 默认 limit=10000, 需要翻页
    frames = []
    offset = 0
    batch_size = 10000
    while True:
        df = _ts_call(
            pro.namechange,
            fields="ts_code,name,start_date,end_date,change_reason",
            limit=batch_size,
            offset=offset,
        )
        if df.empty:
            break
        frames.append(df)
        print(f"  offset={offset}: {len(df)} rows")
        offset += batch_size
        if len(df) < batch_size:
            break

    df = pd.concat(frames, ignore_index=True)
    df["start_date"] = pd.to_datetime(df["start_date"], format="%Y%m%d", errors="coerce")
    df["end_date"] = pd.to_datetime(df["end_date"], format="%Y%m%d", errors="coerce")
    df["change_reason"] = df["change_reason"].fillna("")

    # 去重
    df = df.drop_duplicates(subset=["ts_code", "start_date"], keep="first")

    ch.command("TRUNCATE TABLE astock.dim_namechange")
    _insert_df(ch, "astock.dim_namechange", df[
        ["ts_code", "name", "start_date", "end_date", "change_reason"]
    ])
    print(f"  → dim_namechange: {len(df)} rows written")
    return len(df)


# ══════════════════════════════════════════════════════════════
#  3. dim_suspend_daily  —  日度停牌记录
# ══════════════════════════════════════════════════════════════

DDL_SUSPEND = """
CREATE TABLE IF NOT EXISTS astock.dim_suspend_daily (
    ts_code        String,
    trade_date     Date,
    suspend_timing Nullable(String),
    suspend_type   Nullable(String)
) ENGINE = ReplacingMergeTree()
ORDER BY (trade_date, ts_code)
"""


def load_suspend_daily(start_year: int = 2010):
    print("=" * 60)
    print("[3/5] dim_suspend_daily — 停牌记录")
    pro = get_tushare_pro()
    ch = get_ch()

    ch.command(DDL_SUSPEND)

    today = date.today()
    total_rows = 0

    for year in range(start_year, today.year + 1):
        for month in range(1, 13):
            sd = f"{year}{month:02d}01"
            if month == 12:
                ed = f"{year}1231"
            else:
                ed = f"{year}{month+1:02d}01"

            if int(sd) > int(today.strftime("%Y%m%d")):
                break

            # suspend_d 支持按日期范围查 (ts_code 留空 = 全市场)
            df = _ts_call(
                pro.suspend_d,
                start_date=sd,
                end_date=ed,
                fields="ts_code,trade_date,suspend_timing,suspend_type",
                sleep_between=0.15,
            )
            if df.empty:
                continue

            df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
            df["suspend_timing"] = df["suspend_timing"].fillna("")
            df["suspend_type"] = df["suspend_type"].fillna("")

            _insert_df(ch, "astock.dim_suspend_daily", df)
            total_rows += len(df)

        print(f"  {year}: cumulative {total_rows} rows")

    print(f"  → dim_suspend_daily: {total_rows} rows written")
    return total_rows


# ══════════════════════════════════════════════════════════════
#  4. dim_stk_limit_daily  —  日度涨跌停价
# ══════════════════════════════════════════════════════════════

DDL_STK_LIMIT = """
CREATE TABLE IF NOT EXISTS astock.dim_stk_limit_daily (
    ts_code    String,
    trade_date Date,
    up_limit   Float64,
    down_limit Float64
) ENGINE = ReplacingMergeTree()
ORDER BY (trade_date, ts_code)
"""


def load_stk_limit_daily(start_year: int = 2010):
    print("=" * 60)
    print("[4/5] dim_stk_limit_daily — 涨跌停价")
    print(f"  范围: {start_year}-01-01 ~ today (每日~7500行，预计较慢)")
    pro = get_tushare_pro()
    ch = get_ch()

    ch.command(DDL_STK_LIMIT)

    # 获取交易日历 （用 trade_cal 更高效）
    today_str = date.today().strftime("%Y%m%d")
    cal = _ts_call(
        pro.trade_cal,
        exchange="SSE",
        start_date=f"{start_year}0101",
        end_date=today_str,
        fields="cal_date,is_open",
    )
    trade_dates = sorted(cal[cal["is_open"] == 1]["cal_date"].tolist())
    print(f"  交易日: {len(trade_dates)} days")

    # 检查已有数据，跳过已加载的日期
    existing = ch.query("SELECT DISTINCT toString(trade_date) FROM astock.dim_stk_limit_daily")
    existing_dates = set()
    for row in existing.result_rows:
        existing_dates.add(row[0].replace("-", ""))
    print(f"  已有: {len(existing_dates)} days, 需加载: {len(trade_dates) - len(existing_dates)} days")

    total_rows = 0
    loaded_days = 0
    for i, td in enumerate(trade_dates):
        if td in existing_dates:
            continue

        df = _ts_call(
            pro.stk_limit,
            trade_date=td,
            fields="ts_code,trade_date,up_limit,down_limit",
            sleep_between=0.12,
        )
        if df.empty:
            continue

        df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
        _insert_df(ch, "astock.dim_stk_limit_daily", df)
        total_rows += len(df)
        loaded_days += 1

        if loaded_days % 50 == 0:
            print(f"  进度: {loaded_days} days loaded, {total_rows} rows, "
                  f"current={td}")

    print(f"  → dim_stk_limit_daily: {loaded_days} days, {total_rows} rows written")
    return total_rows


# ══════════════════════════════════════════════════════════════
#  5. dim_index_weights  —  指数成分权重 (PIT)
# ══════════════════════════════════════════════════════════════

DDL_INDEX_WEIGHTS = """
CREATE TABLE IF NOT EXISTS astock.dim_index_weights (
    index_code String,
    con_code   String,
    trade_date Date,
    weight     Float64
) ENGINE = ReplacingMergeTree()
ORDER BY (index_code, trade_date, con_code)
"""

# 需要跟踪的核心指数
TARGET_INDICES = [
    ("399300.SZ", "沪深300"),
    ("000016.SH", "上证50"),
    ("000905.SH", "中证500"),
    ("000852.SH", "中证1000"),
]


def load_index_weights(start_year: int = 2005):
    print("=" * 60)
    print("[5/5] dim_index_weights — 指数成分权重")
    pro = get_tushare_pro()
    ch = get_ch()

    ch.command(DDL_INDEX_WEIGHTS)

    today = date.today()
    total_rows = 0

    for idx_code, idx_name in TARGET_INDICES:
        print(f"  {idx_name} ({idx_code}):")
        idx_rows = 0

        # 按半年切片, 每次最多拿 6000 行 (300成分×20日期)
        for year in range(start_year, today.year + 1):
            for half in [(f"{year}0101", f"{year}0630"), (f"{year}0701", f"{year}1231")]:
                sd, ed = half
                if int(sd) > int(today.strftime("%Y%m%d")):
                    break

                offset = 0
                while True:
                    df = _ts_call(
                        pro.index_weight,
                        index_code=idx_code,
                        start_date=sd,
                        end_date=ed,
                        limit=6000,
                        offset=offset,
                        sleep_between=0.2,
                    )
                    if df.empty:
                        break

                    df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
                    df = df.rename(columns={"index_code": "index_code", "con_code": "con_code"})

                    _insert_df(ch, "astock.dim_index_weights", df[
                        ["index_code", "con_code", "trade_date", "weight"]
                    ])
                    idx_rows += len(df)
                    total_rows += len(df)

                    if len(df) < 6000:
                        break
                    offset += 6000

        print(f"    → {idx_rows} rows")

    print(f"  → dim_index_weights: {total_rows} rows written (4 indices)")
    return total_rows


# ══════════════════════════════════════════════════════════════
#  main
# ══════════════════════════════════════════════════════════════

TABLE_LOADERS = {
    "stock_info": load_stock_info,
    "namechange": load_namechange,
    "suspend":    load_suspend_daily,
    "stk_limit":  load_stk_limit_daily,
    "weights":    load_index_weights,
}


def main():
    parser = argparse.ArgumentParser(description="Load Tushare dim tables → ClickHouse")
    parser.add_argument("--table", choices=list(TABLE_LOADERS.keys()),
                        help="只加载指定表 (默认全部)")
    parser.add_argument("--start-year", type=int, default=2010,
                        help="stk_limit/suspend 起始年份 (默认2010)")
    args = parser.parse_args()

    started = datetime.now()
    print(f"开始时间: {started:%Y-%m-%d %H:%M:%S}")
    print()

    results = {}

    if args.table:
        loader = TABLE_LOADERS[args.table]
        if args.table in ("stk_limit", "suspend"):
            results[args.table] = loader(start_year=args.start_year)
        elif args.table == "weights":
            results[args.table] = loader(start_year=max(2005, args.start_year))
        else:
            results[args.table] = loader()
    else:
        # 按顺序全部加载: 小表先, 大表后
        results["stock_info"] = load_stock_info()
        results["namechange"] = load_namechange()
        results["suspend"] = load_suspend_daily(start_year=args.start_year)
        results["weights"] = load_index_weights(start_year=max(2005, args.start_year))
        # stk_limit 最慢, 放最后
        results["stk_limit"] = load_stk_limit_daily(start_year=args.start_year)

    elapsed = datetime.now() - started
    print()
    print("=" * 60)
    print("完成摘要:")
    for name, count in results.items():
        print(f"  {name}: {count:,} rows")
    print(f"总耗时: {elapsed}")


if __name__ == "__main__":
    main()
