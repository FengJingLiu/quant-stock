#!/usr/bin/env python3
"""将 ETF基础信息列表.csv 写入 ClickHouse astock.dim_etf_info 表。"""

import pandas as pd
from datetime import date as D
from clickhouse_driver import Client

CSV = "data/基金分钟数据/ETF基础信息列表.csv"

CH_KW = dict(
    host="localhost", port=9000,
    user="default", password="***CH_PASSWORD***",
    compression="lz4",
)

DDL = """
CREATE TABLE astock.dim_etf_info (
    symbol            String                  COMMENT 'ETF代码',
    short_name        String                  COMMENT 'ETF简称',
    ext_short_name    String                  COMMENT '扩位简称',
    full_name         String                  COMMENT '基金全称',
    track_index_code  String                  COMMENT '跟踪指数代码',
    track_index_name  String                  COMMENT '跟踪指数名称',
    setup_date        Nullable(Date)          COMMENT '设立日期',
    list_date         Nullable(Date)          COMMENT '上市日期',
    list_status       LowCardinality(String)  COMMENT '上市状态 L/P/D',
    exchange          LowCardinality(String)  COMMENT '交易所 SZ/SH',
    manager           String                  COMMENT '基金管理人',
    custodian         String                  COMMENT '基金托管人',
    mgmt_fee_rate     Nullable(Float32)       COMMENT '管理费率',
    etf_type          LowCardinality(String)  COMMENT 'ETF类型'
) ENGINE = ReplacingMergeTree()
ORDER BY symbol
SETTINGS index_granularity = 8192
"""


def parse_date(s):
    if not s:
        return None
    try:
        p = s.split("-")
        return D(int(p[0]), int(p[1]), int(p[2]))
    except Exception:
        return None


def parse_float(s):
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None


def main():
    df = pd.read_csv(CSV, dtype=str).fillna("")
    df.columns = [
        "symbol", "short_name", "ext_short_name", "full_name",
        "track_index_code", "track_index_name",
        "setup_date", "list_date", "list_status", "exchange",
        "manager", "custodian", "mgmt_fee_rate", "etf_type",
    ]
    df["setup_date"] = df["setup_date"].apply(parse_date)
    df["list_date"] = df["list_date"].apply(parse_date)
    df["mgmt_fee_rate"] = df["mgmt_fee_rate"].apply(parse_float)

    c = Client(**CH_KW)
    c.execute("DROP TABLE IF EXISTS astock.dim_etf_info")
    c.execute(DDL)

    cols = list(df.columns)
    data = [df[col].tolist() for col in cols]
    c.execute(
        f"INSERT INTO astock.dim_etf_info ({', '.join(cols)}) VALUES",
        data, columnar=True, types_check=False,
    )

    cnt = c.execute("SELECT count() FROM astock.dim_etf_info")[0][0]
    print(f"astock.dim_etf_info: {cnt:,} rows")
    for r in c.execute(
        "SELECT symbol, short_name, track_index_name, list_date, etf_type "
        "FROM astock.dim_etf_info LIMIT 5"
    ):
        print(r)


if __name__ == "__main__":
    main()
