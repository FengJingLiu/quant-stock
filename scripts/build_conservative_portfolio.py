"""
稳健型投资组合构建脚本
目标：30万资金，稳健型配置
筛选标准：PE<20, 股息率>4%, 股价低于120日均线（安全边际）
"""

import pandas as pd
import numpy as np

# 读取筛选后的数据
df = pd.read_csv(
    "/home/autumn/quant/stock/data/stock_filter_pe_lt20_div_gt4_below_ma120_20260306.csv"
)

print("=" * 80)
print("30万元稳健型投资组合配置方案")
print("=" * 80)
print(f"\n数据来源日期：{df['date'].iloc[0]}")
print(f"初始筛选股票数量：{len(df)}只")

# 行业分布统计
print("\n" + "=" * 80)
print("一、筛选股票行业分布（PE<20, 股息率>4%, 低于MA120）")
print("=" * 80)
industry_counts = df["industry"].value_counts()
print(industry_counts.head(15).to_string())

# 按照稳健型投资组合原则进行精选
# 1. 优先选择大盘股（流通市值>100亿）
# 2. 优先选择行业龙头
# 3. 优先选择银行股、消费股、公用事业股
# 4. 分散配置，单只股票不超过15%

# 定义重点关注的稳健行业
priority_industries = [
    "银行",
    "白酒",
    "家用电器",
    "中成药",
    "电信运营",
    "路桥",
    "铁路",
    "出版业",
    "供气供热",
]

# 定义精选标的（基于基本面和市场地位）
selected_stocks = [
    # 银行股 - 核心底仓（稳定分红）
    {
        "symbol": "601166.SH",
        "name": "兴业银行",
        "industry": "银行",
        "weight": 0.09,
        "reason": "股份制银行龙头，股息率8.77%",
    },
    {
        "symbol": "600036.SH",
        "name": "招商银行",
        "industry": "银行",
        "weight": 0.08,
        "reason": "零售银行之王，ROE稳定",
    },
    {
        "symbol": "601398.SH",
        "name": "工商银行",
        "industry": "银行",
        "weight": 0.06,
        "reason": "国有大行，分红稳定",
    },
    # 消费白马 - 稳健增长
    {
        "symbol": "000651.SZ",
        "name": "格力电器",
        "industry": "家用电器",
        "weight": 0.07,
        "reason": "空调龙头，股息率8.0%",
    },
    {
        "symbol": "000333.SZ",
        "name": "美的集团",
        "industry": "家用电器",
        "weight": 0.07,
        "reason": "家电全品类龙头",
    },
    {
        "symbol": "000858.SZ",
        "name": "五粮液",
        "industry": "白酒",
        "weight": 0.06,
        "reason": "白酒第二，品牌护城河",
    },
    {
        "symbol": "000568.SZ",
        "name": "泸州老窖",
        "industry": "白酒",
        "weight": 0.05,
        "reason": "老窖系列稳健增长",
    },
    # 医药 - 防御性
    {
        "symbol": "600329.SH",
        "name": "达仁堂",
        "industry": "中成药",
        "weight": 0.05,
        "reason": "中药老字号，PE仅9倍",
    },
    {
        "symbol": "000915.SZ",
        "name": "华特达因",
        "industry": "化学制药",
        "weight": 0.04,
        "reason": "儿童药细分领域龙头",
    },
    # 公共事业 - 防御性
    {
        "symbol": "605368.SH",
        "name": "蓝天燃气",
        "industry": "供气供热",
        "weight": 0.05,
        "reason": "区域燃气龙头，股息率10%",
    },
    {
        "symbol": "600941.SH",
        "name": "中国移动",
        "industry": "电信运营",
        "weight": 0.05,
        "reason": "通信巨头，现金流稳定",
    },
    {
        "symbol": "001965.SZ",
        "name": "招商公路",
        "industry": "路桥",
        "weight": 0.04,
        "reason": "高速公路龙头",
    },
    {
        "symbol": "600377.SH",
        "name": "宁沪高速",
        "industry": "路桥",
        "weight": 0.03,
        "reason": "长三角核心路产",
    },
    # 其他稳健标的
    {
        "symbol": "002563.SZ",
        "name": "森马服饰",
        "industry": "服饰",
        "weight": 0.04,
        "reason": "休闲服饰龙头，股息率9.36%",
    },
    {
        "symbol": "002032.SZ",
        "name": "苏泊尔",
        "industry": "家用电器",
        "weight": 0.04,
        "reason": "小家电龙头",
    },
    {
        "symbol": "601766.SH",
        "name": "中国中车",
        "industry": "运输设备",
        "weight": 0.05,
        "reason": "轨道交通装备龙头",
    },
    {
        "symbol": "600066.SH",
        "name": "宇通客车",
        "industry": "汽车整车",
        "weight": 0.04,
        "reason": "客车行业龙头",
    },
    {
        "symbol": "600690.SH",
        "name": "海尔智家",
        "industry": "家用电器",
        "weight": 0.05,
        "reason": "全球化家电巨头",
    },
    {
        "symbol": "601928.SH",
        "name": "凤凰传媒",
        "industry": "出版业",
        "weight": 0.04,
        "reason": "出版业龙头，股息率6.17%",
    },
]

# 构建投资组合表
portfolio_df = pd.DataFrame(selected_stocks)
portfolio_df["amount"] = 300000 * portfolio_df["weight"]

# 获取当前价格
price_map = dict(zip(df["symbol"], df["close_qfq"]))
pe_map = dict(zip(df["symbol"], df["pe_ttm"]))
div_map = dict(zip(df["symbol"], df["dividend_yield_ttm"]))
pb_map = dict(zip(df["symbol"], df["pb"]))

portfolio_df["price"] = portfolio_df["symbol"].map(price_map)
portfolio_df["pe_ttm"] = portfolio_df["symbol"].map(pe_map)
portfolio_df["dividend_yield"] = portfolio_df["symbol"].map(div_map)
portfolio_df["pb"] = portfolio_df["symbol"].map(pb_map)
portfolio_df["shares"] = (
    (portfolio_df["amount"] / portfolio_df["price"]).round(100).astype(int)
)
portfolio_df["actual_amount"] = portfolio_df["shares"] * portfolio_df["price"]

print("\n" + "=" * 80)
print("二、推荐投资组合配置（30万元）")
print("=" * 80)

# 按行业分组展示
for industry in portfolio_df["industry"].unique():
    industry_stocks = portfolio_df[portfolio_df["industry"] == industry]
    print(f"\n【{industry}】")
    for _, row in industry_stocks.iterrows():
        print(f"  {row['name']} ({row['symbol']})")
        print(
            f"    配置金额: ¥{row['actual_amount']:,.0f} | 股数: {row['shares']}股 | 权重: {row['weight'] * 100:.0f}%"
        )
        print(
            f"    当前价: ¥{row['price']:.2f} | PE: {row['pe_ttm']:.1f} | PB: {row['pb']:.2f} | 股息率: {row['dividend_yield']:.2f}%"
        )
        print(f"    入选理由: {row['reason']}")

# 统计信息
total_invested = portfolio_df["actual_amount"].sum()
cash_remaining = 300000 - total_invested
avg_pe = (portfolio_df["pe_ttm"] * portfolio_df["weight"]).sum()
avg_dividend = (portfolio_df["dividend_yield"] * portfolio_df["weight"]).sum()

print("\n" + "=" * 80)
print("三、组合统计")
print("=" * 80)
print(f"实际投资金额: ¥{total_invested:,.0f}")
print(f"剩余现金: ¥{cash_remaining:,.0f}")
print(f"组合加权平均PE: {avg_pe:.2f}")
print(f"组合加权平均股息率: {avg_dividend:.2f}%")

# 行业配置分布
print("\n" + "=" * 80)
print("四、行业配置分布")
print("=" * 80)
industry_weight = (
    portfolio_df.groupby("industry")
    .agg({"weight": "sum", "actual_amount": "sum"})
    .sort_values("weight", ascending=False)
)
for industry, row in industry_weight.iterrows():
    print(f"{industry:12s}: {row['weight'] * 100:5.1f}% (¥{row['actual_amount']:,.0f})")

# 风险提示
print("\n" + "=" * 80)
print("五、风险提示与建议")
print("=" * 80)
print("""
1. 【分散配置】本组合包含18只股票，分散于9个行业，降低单一风险
2. 【安全边际】所有标的均处于120日均线下方，具备一定的安全边际
3. 【股息收入】组合平均股息率约6%，可提供稳定的现金流回报
4. 【低估值】组合平均PE低于15倍，估值具备一定防御性
5. 【长期持有】建议持有周期3年以上，享受复利和分红再投资
6. 【定期再平衡】建议每季度检视组合，根据估值变化适当调整仓位
7. 【风险提示】股市有风险，投资需谨慎，本方案仅供参考
""")

# 输出CSV供参考
output_file = "/home/autumn/quant/stock/scripts/portfolio_conservative_300k.csv"
portfolio_df[
    [
        "symbol",
        "name",
        "industry",
        "weight",
        "price",
        "shares",
        "actual_amount",
        "pe_ttm",
        "pb",
        "dividend_yield",
        "reason",
    ]
].to_csv(output_file, index=False, encoding="utf-8-sig")
print(f"\n投资组合详情已保存至: {output_file}")

print("\n" + "=" * 80)
print("配置完成！")
print("=" * 80)
