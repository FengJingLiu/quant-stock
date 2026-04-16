import duckdb
import pandas as pd
from datetime import datetime, timedelta

db_path = "/home/autumn/quant/stock/data/duckdb/stock.duckdb"
conn = duckdb.connect(db_path)

result = conn.execute("SELECT MAX(date) FROM v_indicator_daily").fetchone()
latest_date = result[0] if result and result[0] else datetime.now().date()
print(f"数据最新日期: {latest_date}")


high_dividend_query = f"""
SELECT 
    d.symbol,
    s.name,
    s.industry,
    d.close,
    d.dividend_yield_ttm,
    d.pe_ttm,
    d.pb,
    d.total_mv_10k / 10000 as market_cap_bn,
    d.circ_mv_10k / 10000 as float_cap_bn
FROM v_daily_hfq_w_ind_dim d
JOIN v_dim_symbol s ON d.symbol = s.symbol
WHERE d.date = '{latest_date}'
    AND d.dividend_yield_ttm IS NOT NULL
    AND d.dividend_yield_ttm > 4.0
    AND s.is_delisted = FALSE
    AND d.pe_ttm IS NOT NULL
    AND d.pe_ttm > 0
    AND d.total_mv_10k > 0
ORDER BY d.dividend_yield_ttm DESC
LIMIT 100
"""

current_high_div = conn.execute(high_dividend_query).fetchdf()
print(f"\n当前股息率 > 4% 的股票数量: {len(current_high_div)}")
print("\n前20名高股息股票:")
print(current_high_div.head(20).to_string(index=False))

print("\n" + "=" * 80)
print("分红持续性分析（过去3年股息率均值和标准差）")
print("=" * 80)
years_ago = (latest_date - timedelta(days=3 * 365)).strftime("%Y-%m-%d")

consistency_query = f"""
WITH yearly_dividend AS (
    SELECT 
        symbol,
        YEAR(date) as year,
        AVG(dividend_yield_ttm) as avg_dividend_yield,
        MAX(date) as last_date
    FROM v_indicator_daily
    WHERE date >= '{years_ago}'
        AND dividend_yield_ttm IS NOT NULL
        AND dividend_yield_ttm > 0
    GROUP BY symbol, YEAR(date)
),
dividend_stats AS (
    SELECT 
        symbol,
        COUNT(*) as years_with_dividend,
        AVG(avg_dividend_yield) as avg_yield,
        STDDEV(avg_dividend_yield) as std_yield,
        MIN(avg_dividend_yield) as min_yield,
        MAX(avg_dividend_yield) as max_yield
    FROM yearly_dividend
    GROUP BY symbol
    HAVING COUNT(*) >= 2
)
SELECT 
    d.symbol,
    s.name,
    s.industry,
    d.years_with_dividend,
    ROUND(d.avg_yield, 2) as avg_dividend_yield,
    ROUND(d.std_yield, 2) as std_dividend_yield,
    ROUND(d.min_yield, 2) as min_dividend_yield,
    ROUND(d.max_yield, 2) as max_dividend_yield,
    ROUND(d.std_yield / NULLIF(d.avg_yield, 0) * 100, 2) as cv_percent,
    i.pe_ttm,
    i.pb,
    ROUND(i.total_mv_10k / 10000, 2) as market_cap_bn
FROM dividend_stats d
JOIN v_dim_symbol s ON d.symbol = s.symbol
JOIN v_daily_hfq_w_ind_dim i ON d.symbol = i.symbol AND i.date = '{latest_date}'
WHERE d.avg_yield > 3.5
    AND s.is_delisted = FALSE
    AND i.pe_ttm > 0
ORDER BY d.avg_yield DESC
LIMIT 50
"""

dividend_consistency = conn.execute(consistency_query).fetchdf()
print(f"\n过去3年平均股息率 > 3.5% 且数据完整的股票:")
print(dividend_consistency.to_string(index=False))


print("\n" + "=" * 80)
print("综合筛选：高股息 + 分红稳定 + 合理估值")
print("筛选条件：")
print("- 平均股息率 > 4%")
print("- 变异系数 < 30%（分红稳定性）")
print("- PE_TTM < 20（估值合理）")
print("- PB < 3")
print("=" * 80)

quality_query = f"""
WITH yearly_dividend AS (
    SELECT 
        symbol,
        YEAR(date) as year,
        AVG(dividend_yield_ttm) as avg_dividend_yield,
        MAX(date) as last_date
    FROM v_indicator_daily
    WHERE date >= '{years_ago}'
        AND dividend_yield_ttm IS NOT NULL
        AND dividend_yield_ttm > 0
    GROUP BY symbol, YEAR(date)
),
dividend_stats AS (
    SELECT 
        symbol,
        COUNT(*) as years_with_dividend,
        AVG(avg_dividend_yield) as avg_yield,
        STDDEV(avg_dividend_yield) as std_yield,
        MIN(avg_dividend_yield) as min_yield,
        MAX(avg_dividend_yield) as max_yield
    FROM yearly_dividend
    GROUP BY symbol
    HAVING COUNT(*) >= 2
)
SELECT 
    d.symbol,
    s.name,
    s.industry,
    d.years_with_dividend,
    ROUND(d.avg_yield, 2) as avg_dividend_yield,
    ROUND(d.std_yield, 2) as std_dividend_yield,
    ROUND(d.min_yield, 2) as min_dividend_yield,
    ROUND(d.max_yield, 2) as max_dividend_yield,
    ROUND(d.std_yield / NULLIF(d.avg_yield, 0) * 100, 2) as cv_percent,
    i.dividend_yield_ttm as current_yield,
    ROUND(i.pe_ttm, 2) as pe_ttm,
    ROUND(i.pb, 2) as pb,
    ROUND(i.total_mv_10k / 10000, 2) as market_cap_bn
FROM dividend_stats d
JOIN v_dim_symbol s ON d.symbol = s.symbol
JOIN v_daily_hfq_w_ind_dim i ON d.symbol = i.symbol AND i.date = '{latest_date}'
WHERE d.avg_yield > 4.0
    AND (d.std_yield / NULLIF(d.avg_yield, 0)) < 0.30
    AND i.pe_ttm < 20
    AND i.pe_ttm > 0
    AND i.pb < 3
    AND s.is_delisted = FALSE
ORDER BY d.avg_yield DESC
"""

quality_stocks = conn.execute(quality_query).fetchdf()
print(f"\n符合条件的优质高股息股票 ({len(quality_stocks)}只):")
print(quality_stocks.to_string(index=False))


print("\n" + "=" * 80)
print("按行业分布统计")
print("=" * 80)

industry_query = f"""
WITH yearly_dividend AS (
    SELECT 
        symbol,
        YEAR(date) as year,
        AVG(dividend_yield_ttm) as avg_dividend_yield
    FROM v_indicator_daily
    WHERE date >= '{years_ago}'
        AND dividend_yield_ttm IS NOT NULL
        AND dividend_yield_ttm > 0
    GROUP BY symbol, YEAR(date)
),
dividend_stats AS (
    SELECT 
        symbol,
        AVG(avg_dividend_yield) as avg_yield,
        STDDEV(avg_dividend_yield) as std_yield
    FROM yearly_dividend
    GROUP BY symbol
    HAVING COUNT(*) >= 2
)
SELECT 
    s.industry,
    COUNT(*) as stock_count,
    ROUND(AVG(d.avg_yield), 2) as avg_industry_yield,
    ROUND(AVG(i.pe_ttm), 2) as avg_pe,
    ROUND(AVG(i.pb), 2) as avg_pb
FROM dividend_stats d
JOIN v_dim_symbol s ON d.symbol = s.symbol
JOIN v_daily_hfq_w_ind_dim i ON d.symbol = i.symbol AND i.date = '{latest_date}'
WHERE d.avg_yield > 3.5
    AND s.is_delisted = FALSE
    AND i.pe_ttm > 0
GROUP BY s.industry
HAVING COUNT(*) >= 2
ORDER BY avg_industry_yield DESC
LIMIT 20
"""

industry_stats = conn.execute(industry_query).fetchdf()
print("\n高股息行业分布（平均股息率 > 3.5%）:")
print(industry_stats.to_string(index=False))


print("\n" + "=" * 80)
print("银行板块高股息分析")
print("=" * 80)

bank_query = f"""
SELECT 
    d.symbol,
    s.name,
    d.close,
    ROUND(d.dividend_yield_ttm, 2) as current_yield,
    ROUND(d.pe_ttm, 2) as pe_ttm,
    ROUND(d.pb, 2) as pb,
    ROUND(d.total_mv_10k / 10000, 2) as market_cap_bn
FROM v_daily_hfq_w_ind_dim d
JOIN v_dim_symbol s ON d.symbol = s.symbol
WHERE d.date = '{latest_date}'
    AND s.industry LIKE '%银行%'
    AND s.is_delisted = FALSE
    AND d.dividend_yield_ttm IS NOT NULL
ORDER BY d.dividend_yield_ttm DESC
"""

banks = conn.execute(bank_query).fetchdf()
print("\n银行股股息率排名:")
print(banks.to_string(index=False))

conn.close()
output_file = "/home/autumn/quant/stock/scripts/high_dividend_quality_stocks.csv"
quality_stocks.to_csv(output_file, index=False, encoding="utf-8-sig")
print(f"\n\n结果已保存到: {output_file}")

print("\n" + "=" * 80)
print("分析结论")
print("=" * 80)
print(f"""
1. 当前市场共有 {len(current_high_div)} 只股票股息率 > 4%
2. 过去3年平均股息率 > 3.5% 的股票有 {len(dividend_consistency)} 只
3. 综合筛选（高股息+稳定+合理估值）得到 {len(quality_stocks)} 只优质标的

重点关注特征：
- 平均股息率 > 4%
- 变异系数 < 30%（分红波动小，可持续性强）
- PE < 20，PB < 3（估值合理）
- 银行、煤炭、交通运输、公用事业是传统高股息行业
""")
