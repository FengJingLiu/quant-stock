#!/usr/bin/env python3
"""
A股被市场错杀股票筛选器
识别被过度看空但基本面稳健的股票
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# 项目根目录
PROJECT_ROOT = Path("/home/autumn/quant/stock")
DUCKDB_PATH = PROJECT_ROOT / "data" / "duckdb" / "stock.duckdb"

# 筛选参数
MIN_DECLINE = -15  # 近6个月跌幅至少15%
MIN_MARKET_CAP = 50  # 最小市值50亿
MIN_ROE = 5  # 最低ROE 5%
MAX_DEBT_RATIO = 70  # 最高资产负债率70%


def get_price_performance():
    """获取股票近6个月价格表现"""
    os.chdir(PROJECT_ROOT)
    import duckdb

    conn = duckdb.connect(str(DUCKDB_PATH), read_only=True)

    # 获取近6个月价格数据
    query = """
    WITH price_6m AS (
        SELECT 
            symbol,
            close,
            date,
            ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY date DESC) as rn_desc,
            ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY date ASC) as rn_asc
        FROM v_bar_daily_qfq
        WHERE date >= CURRENT_DATE - INTERVAL '6 months'
    ),
    latest_price AS (
        SELECT symbol, close as current_price, date as latest_date
        FROM price_6m WHERE rn_desc = 1
    ),
    price_6m_ago AS (
        SELECT symbol, close as price_6m_ago, date as start_date
        FROM price_6m WHERE rn_asc = 1
    )
    SELECT 
        l.symbol,
        l.current_price,
        p.price_6m_ago,
        l.latest_date,
        p.start_date,
        (l.current_price - p.price_6m_ago) / p.price_6m_ago * 100 as return_6m
    FROM latest_price l
    JOIN price_6m_ago p ON l.symbol = p.symbol
    WHERE p.price_6m_ago > 0
    """

    df = conn.execute(query).fetchdf()
    conn.close()

    return df


def get_fundamental_data():
    """获取基本面数据"""
    os.chdir(PROJECT_ROOT)
    import duckdb

    conn = duckdb.connect(str(DUCKDB_PATH), read_only=True)

    # 获取最新基本面数据
    query = """
    WITH latest_date AS (
        SELECT MAX(date) as max_date FROM v_daily_hfq_w_ind_dim
    )
    SELECT 
        d.symbol,
        s.name,
        s.industry,
        d.close as current_price,
        d.pe_ttm,
        d.pb,
        d.total_mv_10k / 10000 as market_cap,
        d.circ_mv_10k / 10000 as circulating_cap,
        d.turnover_rate,
        d.dividend_yield_ttm
    FROM v_daily_hfq_w_ind_dim d
    LEFT JOIN v_dim_symbol s ON d.symbol = s.symbol
    WHERE d.date = (SELECT max_date FROM latest_date)
      AND s.market_type NOT IN ('科创板', '创业板', '北交所')
      AND d.total_mv_10k >= 500000  -- 市值>=50亿
      AND d.pe_ttm > 0  -- 非亏损
    """

    df = conn.execute(query).fetchdf()
    conn.close()

    return df


def get_industry_comparison():
    """获取行业估值中位数"""
    os.chdir(PROJECT_ROOT)
    import duckdb

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


def get_historical_valuation():
    """获取历史估值数据（近2年）"""
    os.chdir(PROJECT_ROOT)
    import duckdb

    conn = duckdb.connect(str(DUCKDB_PATH), read_only=True)

    query = """
    SELECT 
        symbol,
        median(pe_ttm) as hist_median_pe,
        avg(pe_ttm) as hist_avg_pe,
        stddev(pe_ttm) as hist_std_pe,
        median(pb) as hist_median_pb,
        avg(pb) as hist_avg_pb
    FROM v_daily_hfq_w_ind_dim
    WHERE date >= CURRENT_DATE - INTERVAL '2 years'
      AND pe_ttm > 0 AND pe_ttm < 100
      AND pb > 0 AND pb < 10
    GROUP BY symbol
    """

    df = conn.execute(query).fetchdf()
    conn.close()

    return df


def calculate_sentiment_score(row):
    """计算情绪偏差评分"""
    score = 0

    # 跌幅越大，情绪越差，得分越高（逆向指标）
    return_6m = row.get("return_6m", 0)
    if return_6m <= -30:
        score += 40
    elif return_6m <= -20:
        score += 30
    elif return_6m <= -15:
        score += 20

    # 相对行业跌幅越大，得分越高
    excess_decline = row.get("excess_decline", 0)
    if excess_decline <= -10:
        score += 20
    elif excess_decline <= -5:
        score += 10

    # PE相对历史中位数越低，得分越高
    pe_vs_hist = row.get("pe_vs_hist_median", 0)
    if pe_vs_hist <= -30:
        score += 20
    elif pe_vs_hist <= -20:
        score += 15
    elif pe_vs_hist <= -10:
        score += 10

    # PB相对历史中位数越低，得分越高
    pb_vs_hist = row.get("pb_vs_hist_median", 0)
    if pb_vs_hist <= -30:
        score += 20
    elif pb_vs_hist <= -20:
        score += 10

    return score


def calculate_fundamental_score(row):
    """计算基本面质量评分"""
    score = 0

    # 盈利稳定性
    pe = row.get("pe_ttm", 0)
    if 0 < pe <= 15:
        score += 20
    elif 0 < pe <= 25:
        score += 15
    elif 0 < pe <= 40:
        score += 10

    # 估值合理性
    pb = row.get("pb", 0)
    if 0 < pb <= 1.5:
        score += 20
    elif 0 < pb <= 2.5:
        score += 15
    elif 0 < pb <= 4:
        score += 10

    # 行业地位（市值）
    market_cap = row.get("market_cap", 0)
    if market_cap >= 500:
        score += 15
    elif market_cap >= 200:
        score += 10
    elif market_cap >= 100:
        score += 5

    # 流动性
    turnover = row.get("turnover_rate", 0)
    if turnover >= 1:
        score += 10
    elif turnover >= 0.5:
        score += 5

    return score


def main():
    print("=" * 80)
    print("A股被市场错杀股票筛选器")
    print("=" * 80)
    print(f"筛选条件: 近6个月跌幅≥{abs(MIN_DECLINE)}%, 市值≥{MIN_MARKET_CAP}亿, 非亏损")

    if not DUCKDB_PATH.exists():
        print(f"错误：找不到数据库 {DUCKDB_PATH}")
        sys.exit(1)

    # 获取数据
    print("\n[1/4] 获取近6个月价格表现...")
    price_df = get_price_performance()
    print(f"      获取到 {len(price_df)} 只股票的价格数据")

    print("\n[2/4] 获取基本面数据...")
    fund_df = get_fundamental_data()
    print(f"      获取到 {len(fund_df)} 只股票的基本面数据")

    print("\n[3/4] 获取行业估值对比...")
    industry_df = get_industry_comparison()
    print(f"      覆盖 {len(industry_df)} 个行业")

    print("\n[4/4] 获取历史估值数据...")
    hist_df = get_historical_valuation()
    print(f"      获取到 {len(hist_df)} 只股票的历史估值")

    # 合并数据
    print("\n[分析] 合并数据并计算偏差...")
    merged = price_df.merge(fund_df, on="symbol", how="inner", suffixes=("", "_fund"))
    merged = merged.merge(
        industry_df, on="industry", how="left", suffixes=("", "_industry")
    )
    merged = merged.merge(hist_df, on="symbol", how="left", suffixes=("", "_hist"))

    # 统一current_price列名
    if "current_price_fund" in merged.columns:
        merged["current_price"] = merged["current_price_fund"]

    # 计算相对行业跌幅
    merged["excess_decline"] = merged["return_6m"]  # 简化处理，实际应减去行业指数涨跌幅

    # 计算估值相对历史中位数的偏离
    merged["pe_vs_hist_median"] = (
        (merged["pe_ttm"] - merged["hist_median_pe"]) / merged["hist_median_pe"] * 100
    ).fillna(0)
    merged["pb_vs_hist_median"] = (
        (merged["pb"] - merged["hist_median_pb"]) / merged["hist_median_pb"] * 100
    ).fillna(0)

    # 筛选跌幅较大的股票
    declined = merged[merged["return_6m"] <= MIN_DECLINE].copy()
    print(f"      近6个月跌幅≥{abs(MIN_DECLINE)}%的股票: {len(declined)} 只")

    # 计算评分
    declined["sentiment_score"] = declined.apply(calculate_sentiment_score, axis=1)
    declined["fundamental_score"] = declined.apply(calculate_fundamental_score, axis=1)
    declined["total_score"] = (
        declined["sentiment_score"] + declined["fundamental_score"]
    )

    # 排序：情绪偏差大且基本面好的排前面
    declined = declined.sort_values("total_score", ascending=False)

    # 显示结果
    print("\n" + "=" * 80)
    print("被市场错杀的股票候选（情绪偏差+基本面质量综合排序）")
    print("=" * 80)

    display_cols = [
        "symbol",
        "name",
        "industry",
        "return_6m",
        "current_price",
        "pe_ttm",
        "pb",
        "market_cap",
        "pe_vs_hist_median",
        "sentiment_score",
        "fundamental_score",
        "total_score",
    ]

    top20 = declined.head(20)[display_cols].copy()
    top20["return_6m"] = top20["return_6m"].round(1)
    top20["current_price"] = top20["current_price"].round(2)
    top20["pe_ttm"] = top20["pe_ttm"].round(2)
    top20["pb"] = top20["pb"].round(2)
    top20["market_cap"] = top20["market_cap"].round(1)
    top20["pe_vs_hist_median"] = top20["pe_vs_hist_median"].round(1)

    print(top20.to_string(index=False))

    # 保存结果
    output_path = PROJECT_ROOT / "scripts" / "sentiment_gap_candidates.csv"
    declined.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n[保存] 完整候选列表已保存至: {output_path}")

    # 返回Top 10
    top10 = declined.head(10)
    print("\n" + "=" * 80)
    print("Top 10 被错杀股票详细分析")
    print("=" * 80)

    for rank, (_, row) in enumerate(top10.iterrows(), 1):
        print(f"\n【{rank}】{row['symbol']} {row['name']} ({row['industry']})")
        print(f"  近6个月跌幅: {row['return_6m']:.1f}%")
        print(f"  当前股价: ¥{row['current_price']:.2f}")
        print(
            f"  估值: PE {row['pe_ttm']:.1f} (历史中位数 {row['hist_median_pe']:.1f}, 偏离 {row['pe_vs_hist_median']:.1f}%)"
        )
        print(f"  市值: {row['market_cap']:.1f}亿")
        print(f"  情绪偏差评分: {row['sentiment_score']}/100")
        print(f"  基本面评分: {row['fundamental_score']}/100")
        print(f"  综合评分: {row['total_score']}/200")

    return top10["symbol"].tolist()


if __name__ == "__main__":
    top10_symbols = main()
    print("\n[输出] Top 10 被错杀股票代码：")
    print(", ".join(top10_symbols))
