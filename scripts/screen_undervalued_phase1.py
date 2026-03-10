#!/usr/bin/env python3
"""
A股低估值股票筛选器 - 第一阶段：基础指标筛选
使用本地DuckDB数据获取全市场股票估值指标
"""

import sys
import os
from pathlib import Path
import duckdb
import pandas as pd

# 项目根目录
PROJECT_ROOT = Path("/home/autumn/quant/stock")
DUCKDB_PATH = PROJECT_ROOT / "data" / "duckdb" / "stock.duckdb"

# 筛选参数
MIN_MARKET_CAP = 50  # 最小市值50亿（排除过小市值股票）
MAX_PE = 50  # 最大PE 50倍（排除过高估值）
MAX_PB = 5  # 最大PB 5倍
MIN_TURNOVER = 0.5  # 最小日换手率0.5%（确保流动性）


def get_stock_universe():
    """从本地DuckDB获取全市场股票基础指标"""
    os.chdir(PROJECT_ROOT)
    conn = duckdb.connect(str(DUCKDB_PATH), read_only=True)

    # 获取最新日期的股票数据，包含基础估值指标
    query = """
    WITH latest_date AS (
        SELECT MAX(date) as max_date FROM v_daily_hfq_w_ind_dim
    )
    SELECT 
        d.symbol,
        s.name,
        s.industry,
        s.exchange,
        d.close as price,
        d.pe_ttm,
        d.pb,
        d.turnover_rate,
        d.total_mv_10k as market_cap_10k,
        d.circ_mv_10k as circulating_cap_10k,
        d.dividend_yield_ttm,
        d.date
    FROM v_daily_hfq_w_ind_dim d
    LEFT JOIN v_dim_symbol s ON d.symbol = s.symbol
    WHERE d.date = (SELECT max_date FROM latest_date)
      AND s.market_type NOT IN ('科创板', '创业板', '北交所')  -- 先聚焦主板
      AND d.pe_ttm > 0  -- 排除亏损公司
      AND d.pe_ttm < 100  -- 排除极端高估值
      AND d.total_mv_10k >= 500000  -- 市值 >= 50亿
      AND d.turnover_rate >= 0.3  -- 流动性要求
    ORDER BY d.total_mv_10k DESC
    """

    df = conn.execute(query).fetchdf()
    conn.close()

    # 转换市值单位为亿元
    df["market_cap"] = df["market_cap_10k"] / 10000  # 转换为亿元
    df["circulating_cap"] = df["circulating_cap_10k"] / 10000

    return df


def get_industry_medians():
    """计算各行业的PE、PB中位数"""
    os.chdir(PROJECT_ROOT)
    conn = duckdb.connect(str(DUCKDB_PATH), read_only=True)

    query = """
    WITH latest_date AS (
        SELECT MAX(date) as max_date FROM v_daily_hfq_w_ind_dim
    )
    SELECT 
        s.industry,
        median(d.pe_ttm) as median_pe,
        median(d.pb) as median_pb,
        count(*) as stock_count
    FROM v_daily_hfq_w_ind_dim d
    LEFT JOIN v_dim_symbol s ON d.symbol = s.symbol
    WHERE d.date = (SELECT max_date FROM latest_date)
      AND d.pe_ttm > 0 AND d.pe_ttm < 100
      AND d.pb > 0 AND d.pb < 10
      AND s.market_type NOT IN ('科创板', '创业板', '北交所')
    GROUP BY s.industry
    HAVING count(*) >= 5
    """

    df = conn.execute(query).fetchdf()
    conn.close()

    return df


def apply_initial_screening(stock_df, industry_df):
    """应用初步筛选条件"""
    # 合并行业数据
    merged = stock_df.merge(
        industry_df[["industry", "median_pe", "median_pb"]], on="industry", how="left"
    )

    # 筛选条件：
    # 1. PE低于行业中位数
    # 2. PB低于行业中位数 或 PB < 1.5
    # 3. 市值 >= 50亿

    screened = merged[
        (merged["pe_ttm"] < merged["median_pe"])  # PE低于行业中位数
        & (
            (merged["pb"] < merged["median_pb"]) | (merged["pb"] < 1.5)
        )  # PB低于行业中位数或绝对低
        & (merged["market_cap"] >= 50)  # 市值>=50亿
    ].copy()

    # 计算相对估值指标
    screened["pe_discount"] = (1 - screened["pe_ttm"] / screened["median_pe"]) * 100
    screened["pb_discount"] = (1 - screened["pb"] / screened["median_pb"]) * 100

    # 计算综合评分（估值折扣越高越好，PE权重40%，PB权重30%，股息率权重30%）
    screened["dividend_yield_ttm"] = screened["dividend_yield_ttm"].fillna(0)
    screened["score"] = (
        screened["pe_discount"] * 0.4
        + screened["pb_discount"] * 0.3
        + screened["dividend_yield_ttm"] * 10 * 0.3  # 股息率转换为百分制
    )

    return screened.sort_values("score", ascending=False)


def main():
    print("=" * 80)
    print("A股低估值股票筛选 - 第一阶段：基础指标筛选")
    print("=" * 80)

    # 检查数据库
    if not DUCKDB_PATH.exists():
        print(f"错误：找不到数据库 {DUCKDB_PATH}")
        sys.exit(1)

    print("\n[1/3] 获取全市场股票基础数据...")
    stock_df = get_stock_universe()
    print(f"      获取到 {len(stock_df)} 只主板非亏损股票")

    print("\n[2/3] 计算行业估值中位数...")
    industry_df = get_industry_medians()
    print(f"      覆盖 {len(industry_df)} 个申万一级行业")
    print("\n各行业估值中位数预览（前10）：")
    print(industry_df.sort_values("median_pe").head(10).to_string(index=False))

    print("\n[3/3] 应用筛选条件...")
    screened = apply_initial_screening(stock_df, industry_df)
    print(f"      通过初步筛选：{len(screened)} 只股票")

    # 显示前20名
    print("\n" + "=" * 80)
    print("低估值候选股票（前20名，按综合评分排序）")
    print("=" * 80)

    display_cols = [
        "symbol",
        "name",
        "industry",
        "price",
        "pe_ttm",
        "median_pe",
        "pb",
        "median_pb",
        "market_cap",
        "pe_discount",
        "pb_discount",
        "score",
    ]

    top20 = screened.head(20)[display_cols].copy()
    top20["price"] = top20["price"].round(2)
    top20["pe_ttm"] = top20["pe_ttm"].round(2)
    top20["median_pe"] = top20["median_pe"].round(2)
    top20["pb"] = top20["pb"].round(2)
    top20["median_pb"] = top20["median_pb"].round(2)
    top20["market_cap"] = top20["market_cap"].round(1)
    top20["pe_discount"] = top20["pe_discount"].round(1)
    top20["pb_discount"] = top20["pb_discount"].round(1)
    top20["score"] = top20["score"].round(2)

    print(top20.to_string(index=False))

    # 保存结果
    output_path = PROJECT_ROOT / "scripts" / "screening_candidates.csv"
    screened.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n[保存] 完整候选列表已保存至: {output_path}")

    # 返回前10名代码供进一步分析
    top10_symbols = screened.head(10)["symbol"].tolist()
    print("\n[输出] Top 10 候选股票代码：")
    print(", ".join(top10_symbols))

    return top10_symbols


if __name__ == "__main__":
    top10 = main()
    # 输出代码到stdout，方便后续脚本使用
    print("\n[SYMBOLS] " + ",".join(top10))
