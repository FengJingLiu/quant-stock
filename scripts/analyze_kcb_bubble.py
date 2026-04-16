#!/usr/bin/env python3

import duckdb
import pandas as pd


def analyze_kcb_valuation():
    con = duckdb.connect("../data/duckdb/stock.duckdb", read_only=True)

    result = con.execute("SELECT MAX(date) FROM v_indicator_daily").fetchone()
    if result is None or result[0] is None:
        print("无法获取最新日期")
        return None
    latest_date = result[0]
    print(f"数据最新日期: {latest_date}")
    print("=" * 80)

    query_stats = f"""
    SELECT
        COUNT(*) as total_count,
        ROUND(AVG(CASE WHEN pe_ttm > 0 AND pe_ttm < 1000 THEN pe_ttm END), 2) as avg_pe_ttm,
        ROUND(MEDIAN(CASE WHEN pe_ttm > 0 AND pe_ttm < 1000 THEN pe_ttm END), 2) as median_pe_ttm,
        ROUND(AVG(CASE WHEN pb > 0 AND pb < 100 THEN pb END), 2) as avg_pb,
        ROUND(MEDIAN(CASE WHEN pb > 0 AND pb < 100 THEN pb END), 2) as median_pb,
        ROUND(AVG(total_mv_10k / 10000), 2) as avg_mv_yi
    FROM v_dim_symbol d
    JOIN v_indicator_daily i ON d.symbol = i.symbol
    WHERE d.market_type = '科创板'
        AND i.date = '{latest_date}'
        AND d.is_delisted = 0
    """

    print("\n【科创板整体估值统计】")
    df_stats = con.execute(query_stats).fetchdf()
    print(f"公司数量: {df_stats.iloc[0]['total_count']}家")
    print(f"平均PE TTM: {df_stats.iloc[0]['avg_pe_ttm']}倍")
    print(f"PE TTM中位数: {df_stats.iloc[0]['median_pe_ttm']}倍")
    print(f"平均PB: {df_stats.iloc[0]['avg_pb']}倍")
    print(f"PB中位数: {df_stats.iloc[0]['median_pb']}倍")
    print(f"平均市值: {df_stats.iloc[0]['avg_mv_yi']}亿元")

    query_pe = f"""
    SELECT
        d.symbol,
        d.name,
        d.industry,
        i.pe_ttm,
        i.pb,
        i.ps_ttm,
        i.total_mv_10k / 10000 as total_mv_yi,
        b.close
    FROM v_dim_symbol d
    JOIN v_indicator_daily i ON d.symbol = i.symbol
    JOIN v_bar_daily_raw b ON d.symbol = b.symbol AND i.date = b.date
    WHERE d.market_type = '科创板'
        AND i.date = '{latest_date}'
        AND d.is_delisted = 0
        AND i.pe_ttm IS NOT NULL
        AND i.pe_ttm > 0
    ORDER BY i.pe_ttm DESC
    LIMIT 30
    """

    print("\n" + "=" * 80)
    print("【科创板估值泡沫最严重公司 TOP 30（按PE TTM排序）】")
    df_pe = con.execute(query_pe).fetchdf()
    df_pe.index = range(1, len(df_pe) + 1)
    print(df_pe.to_string())

    query_pb = f"""
    SELECT
        d.symbol,
        d.name,
        d.industry,
        i.pe_ttm,
        i.pb,
        i.ps_ttm,
        i.total_mv_10k / 10000 as total_mv_yi,
        b.close
    FROM v_dim_symbol d
    JOIN v_indicator_daily i ON d.symbol = i.symbol
    JOIN v_bar_daily_raw b ON d.symbol = b.symbol AND i.date = b.date
    WHERE d.market_type = '科创板'
        AND i.date = '{latest_date}'
        AND d.is_delisted = 0
        AND i.pb IS NOT NULL
        AND i.pb > 0
    ORDER BY i.pb DESC
    LIMIT 20
    """

    print("\n" + "=" * 80)
    print("【科创板市净率(PB)最高公司 TOP 20】")
    df_pb = con.execute(query_pb).fetchdf()
    df_pb.index = range(1, len(df_pb) + 1)
    print(df_pb.to_string())

    query_industry = f"""
    SELECT
        d.industry,
        COUNT(*) as company_count,
        ROUND(AVG(CASE WHEN pe_ttm > 0 AND pe_ttm < 1000 THEN pe_ttm END), 2) as avg_pe_ttm,
        ROUND(AVG(CASE WHEN pb > 0 AND pb < 100 THEN pb END), 2) as avg_pb,
        ROUND(AVG(total_mv_10k / 10000), 2) as avg_mv_yi
    FROM v_dim_symbol d
    JOIN v_indicator_daily i ON d.symbol = i.symbol
    WHERE d.market_type = '科创板'
        AND i.date = '{latest_date}'
        AND d.is_delisted = 0
    GROUP BY d.industry
    HAVING COUNT(*) >= 5
    ORDER BY avg_pe_ttm DESC NULLS LAST
    LIMIT 15
    """

    print("\n" + "=" * 80)
    print("【科创板细分行业估值（公司数≥5的行业按PE排序）】")
    df_ind = con.execute(query_industry).fetchdf()
    df_ind.index = range(1, len(df_ind) + 1)
    print(df_ind.to_string())

    query_loss = f"""
    SELECT
        d.symbol,
        d.name,
        d.industry,
        i.pe_ttm,
        i.pb,
        i.ps_ttm,
        i.total_mv_10k / 10000 as total_mv_yi
    FROM v_dim_symbol d
    JOIN v_indicator_daily i ON d.symbol = i.symbol
    WHERE d.market_type = '科创板'
        AND i.date = '{latest_date}'
        AND d.is_delisted = 0
        AND (i.pe_ttm IS NULL OR i.pe_ttm < 0)
        AND i.total_mv_10k / 10000 > 50
    ORDER BY i.total_mv_10k DESC
    LIMIT 20
    """

    print("\n" + "=" * 80)
    print("【科创板亏损/PE为负但市值>50亿公司 TOP 20（按市值排序）】")
    df_loss = con.execute(query_loss).fetchdf()
    df_loss.index = range(1, len(df_loss) + 1)
    print(df_loss.to_string())

    con.close()

    return {
        "date": latest_date,
        "top_pe": df_pe,
        "top_pb": df_pb,
        "industry": df_ind,
        "loss_makers": df_loss,
    }


if __name__ == "__main__":
    analyze_kcb_valuation()
