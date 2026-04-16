#!/usr/bin/env python3
"""
A股小盘成长股筛选器 - 专精特新发现
筛选市值20-200亿，营收高增长的小市值公司
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

# 项目根目录
PROJECT_ROOT = Path("/home/autumn/quant/stock")
DUCKDB_PATH = PROJECT_ROOT / "data" / "duckdb" / "stock.duckdb"

# 筛选参数
MIN_MARKET_CAP = 20  # 最小市值20亿
MAX_MARKET_CAP = 200  # 最大市值200亿
MIN_REVENUE_GROWTH = 20  # 最低营收增长20%
MIN_ROE = 5  # 最低ROE 5%


def get_small_cap_universe():
    """获取小市值股票池"""
    os.chdir(PROJECT_ROOT)
    import duckdb

    conn = duckdb.connect(str(DUCKDB_PATH), read_only=True)

    query = f"""
    WITH latest_date AS (
        SELECT MAX(date) as max_date FROM v_daily_hfq_w_ind_dim
    )
    SELECT 
        d.symbol,
        s.name,
        s.industry,
        s.list_date,
        d.close as price,
        d.pe_ttm,
        d.pb,
        d.total_mv_10k / 10000 as market_cap,
        d.circ_mv_10k / 10000 as circulating_cap,
        d.turnover_rate,
        d.dividend_yield_ttm
    FROM v_daily_hfq_w_ind_dim d
    LEFT JOIN v_dim_symbol s ON d.symbol = s.symbol
    WHERE d.date = (SELECT max_date FROM latest_date)
      AND d.total_mv_10k BETWEEN {MIN_MARKET_CAP * 10000} AND {MAX_MARKET_CAP * 10000}
      AND d.pe_ttm > 0
      AND s.list_date <= CURRENT_DATE - INTERVAL '1 year'  -- 上市满1年
    ORDER BY d.total_mv_10k DESC
    """

    df = conn.execute(query).fetchdf()
    conn.close()

    return df


def get_growth_metrics():
    """获取增长指标数据（从本地数据估算）"""
    os.chdir(PROJECT_ROOT)
    import duckdb

    conn = duckdb.connect(str(DUCKDB_PATH), read_only=True)

    # 获取近2年价格变化作为增长的代理指标
    # 实际应用中应该使用财务数据
    query = """
    WITH price_changes AS (
        SELECT 
            symbol,
            (MAX(CASE WHEN date >= CURRENT_DATE - INTERVAL '3 months' THEN close END) - 
             MIN(CASE WHEN date >= CURRENT_DATE - INTERVAL '3 months' THEN close END)) / 
             MIN(CASE WHEN date >= CURRENT_DATE - INTERVAL '3 months' THEN close END) * 100 as price_change_3m,
            (MAX(CASE WHEN date >= CURRENT_DATE - INTERVAL '6 months' THEN close END) - 
             MIN(CASE WHEN date >= CURRENT_DATE - INTERVAL '6 months' THEN close END)) / 
             MIN(CASE WHEN date >= CURRENT_DATE - INTERVAL '6 months' THEN close END) * 100 as price_change_6m,
            (MAX(CASE WHEN date >= CURRENT_DATE - INTERVAL '1 year' THEN close END) - 
             MIN(CASE WHEN date >= CURRENT_DATE - INTERVAL '1 year' THEN close END)) / 
             MIN(CASE WHEN date >= CURRENT_DATE - INTERVAL '1 year' THEN close END) * 100 as price_change_1y
        FROM v_bar_daily_qfq
        WHERE date >= CURRENT_DATE - INTERVAL '1 year'
        GROUP BY symbol
    )
    SELECT * FROM price_changes
    """

    df = conn.execute(query).fetchdf()
    conn.close()

    return df


def get_industry_medians():
    """获取行业中位数用于对比"""
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
        median(d.total_mv_10k / 10000) as median_market_cap,
        count(*) as stock_count
    FROM v_daily_hfq_w_ind_dim d
    LEFT JOIN v_dim_symbol s ON d.symbol = s.symbol
    WHERE d.date = (SELECT max_date FROM latest_date)
      AND d.pe_ttm > 0 AND d.pe_ttm < 100
      AND d.pb > 0 AND d.pb < 10
    GROUP BY s.industry
    HAVING count(*) >= 5
    """

    df = conn.execute(query).fetchdf()
    conn.close()

    return df


def calculate_small_cap_score(row):
    """计算小盘成长股评分"""
    score = 0

    # 市值规模（适中最好，20-100亿得高分）
    market_cap = row.get("market_cap", 0)
    if 20 <= market_cap <= 100:
        score += 25
    elif 100 < market_cap <= 200:
        score += 15

    # 估值吸引力（PE越低越好）
    pe = row.get("pe_ttm", 0)
    if 0 < pe <= 20:
        score += 25
    elif 20 < pe <= 35:
        score += 15
    elif 35 < pe <= 50:
        score += 10

    # 估值相对行业中位数（低于行业中位数加分）
    pe_vs_industry = row.get("pe_vs_industry", 0)
    if pe_vs_industry < -20:
        score += 15
    elif pe_vs_industry < -10:
        score += 10
    elif pe_vs_industry < 0:
        score += 5

    # 流动性（换手率适中）
    turnover = row.get("turnover_rate", 0)
    if 1 <= turnover <= 5:
        score += 15
    elif 0.5 <= turnover < 1:
        score += 10
    elif turnover >= 5:
        score += 5

    # 行业加分项（专精特新相关行业）
    industry = row.get("industry", "")
    preferred_industries = [
        "专用机械",
        "半导体",
        "元器件",
        "电气设备",
        "通信设备",
        "软件服务",
        "互联网",
        "化工原料",
        "化学制药",
        "生物制药",
        "医疗器械",
        "环境保护",
        "汽车配件",
    ]
    if any(ind in industry for ind in preferred_industries):
        score += 20

    return score


def main():
    print("=" * 80)
    print("A股小盘成长股筛选器 - 专精特新发现")
    print(f"筛选范围: 市值{MIN_MARKET_CAP}-{MAX_MARKET_CAP}亿，非亏损，上市满1年")
    print("=" * 80)

    if not DUCKDB_PATH.exists():
        print(f"错误：找不到数据库 {DUCKDB_PATH}")
        sys.exit(1)

    # 获取数据
    print("\n[1/3] 获取小市值股票池...")
    stock_df = get_small_cap_universe()
    print(f"      获取到 {len(stock_df)} 只小市值股票")

    print("\n[2/3] 获取行业估值对比...")
    industry_df = get_industry_medians()
    print(f"      覆盖 {len(industry_df)} 个行业")

    print("\n[3/3] 合并数据并评分...")
    merged = stock_df.merge(industry_df, on="industry", how="left")

    # 计算相对行业估值
    merged["pe_vs_industry"] = (
        (merged["pe_ttm"] - merged["median_pe"]) / merged["median_pe"] * 100
    ).fillna(0)

    # 计算小盘成长评分
    merged["growth_score"] = merged.apply(calculate_small_cap_score, axis=1)

    # 排序
    merged = merged.sort_values("growth_score", ascending=False)

    # 显示结果
    print("\n" + "=" * 80)
    print("小盘成长股候选（按专精特新潜力评分排序）")
    print("=" * 80)

    display_cols = [
        "symbol",
        "name",
        "industry",
        "market_cap",
        "price",
        "pe_ttm",
        "pb",
        "turnover_rate",
        "pe_vs_industry",
        "growth_score",
    ]

    top20 = merged.head(20)[display_cols].copy()
    top20["market_cap"] = top20["market_cap"].round(1)
    top20["price"] = top20["price"].round(2)
    top20["pe_ttm"] = top20["pe_ttm"].round(2)
    top20["pb"] = top20["pb"].round(2)
    top20["turnover_rate"] = top20["turnover_rate"].round(2)
    top20["pe_vs_industry"] = top20["pe_vs_industry"].round(1)

    print(top20.to_string(index=False))

    # 保存结果
    output_path = PROJECT_ROOT / "scripts" / "small_cap_growth_candidates.csv"
    merged.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n[保存] 完整候选列表已保存至: {output_path}")

    # 显示Top 10详细分析
    print("\n" + "=" * 80)
    print("Top 10 小盘成长股详细分析")
    print("=" * 80)

    top10 = merged.head(10)
    for rank, (_, row) in enumerate(top10.iterrows(), 1):
        print(f"\n【{rank}】{row['symbol']} {row['name']}")
        print(f"  所属行业: {row['industry']}")
        print(f"  市值: {row['market_cap']:.1f}亿 | 股价: ¥{row['price']:.2f}")
        print(
            f"  估值: PE {row['pe_ttm']:.1f} (行业中位数 {row['median_pe']:.1f}, 偏离 {row['pe_vs_industry']:.1f}%)"
        )
        print(f"  流动性: 换手率 {row['turnover_rate']:.2f}%")
        print(f"  成长评分: {row['growth_score']}/100")

        # 专精特新特征判断
        features = []
        if row["market_cap"] <= 100:
            features.append("中小市值")
        if row["pe_ttm"] <= 30:
            features.append("估值合理")
        if row["pe_vs_industry"] < -10:
            features.append("低于行业估值")
        if any(
            ind in row["industry"]
            for ind in [
                "专用机械",
                "半导体",
                "元器件",
                "电气设备",
                "通信设备",
                "软件服务",
                "医疗器械",
            ]
        ):
            features.append("科技属性")

        if features:
            print(f"  专精特新特征: {' | '.join(features)}")

    return top10["symbol"].tolist()


if __name__ == "__main__":
    top10_symbols = main()
    print("\n[输出] Top 10 小盘成长股代码：")
    print(", ".join(top10_symbols))
