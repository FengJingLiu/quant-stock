#!/usr/bin/env python3
"""
A股低估值股票筛选器 - 第二阶段：深度财务指标筛选
获取候选股票的ROE、营收增长、资产负债率、现金流等详细指标
"""

import sys
import os
from pathlib import Path
import pandas as pd
import time

# 添加findata-toolkit-cn路径
sys.path.insert(
    0,
    str(
        Path(__file__).resolve().parent.parent
        / "skills"
        / "findata-toolkit-cn"
        / "scripts"
    ),
)
from common import akshare_patch

# 候选股票代码（从第一阶段输出获取）
CANDIDATES = [
    "603508.SH",
    "601919.SH",
    "601668.SH",
    "600582.SH",
    "601186.SH",
    "000651.SZ",
    "600704.SH",
    "600741.SH",
    "600894.SH",
    "601390.SH",
]

# 筛选标准
MIN_ROE = 8  # 最低ROE 8%
MIN_REVENUE_GROWTH = -5  # 营收增长最低-5%（允许轻微下滑）
MAX_DEBT_RATIO = 70  # 最高资产负债率70%


def normalize_symbol(symbol):
    """标准化代码格式为AKShare格式"""
    sym = symbol.replace(".SH", "").replace(".SZ", "").replace(".BJ", "")
    return sym.zfill(6)


def parse_percent(value):
    """解析百分比字符串为浮点数"""
    if value is None or value == "" or value == "-":
        return None
    if isinstance(value, str):
        value = value.replace("%", "").replace(",", "").strip()
        try:
            return float(value)
        except:
            return None
    return float(value) if value else None


def fetch_financial_abstract(symbol):
    """获取财务摘要数据"""
    ak = akshare_patch.get_akshare()
    sym = normalize_symbol(symbol)

    try:
        # 获取财务摘要
        df = ak.stock_financial_abstract_ths(symbol=sym, indicator="按报告期")
        if df is not None and not df.empty:
            latest = df.iloc[-1]
            return {
                "symbol": symbol,
                "roe": parse_percent(latest.get("净资产收益率")),
                "revenue_growth": parse_percent(latest.get("营业总收入同比增长率")),
                "profit_growth": parse_percent(latest.get("净利润同比增长率")),
                "debt_ratio": parse_percent(latest.get("资产负债率")),
                "gross_margin": parse_percent(latest.get("销售毛利率")),
                "net_margin": parse_percent(latest.get("销售净利率")),
                "current_ratio": parse_percent(latest.get("流动比率")),
                "report_period": str(latest.get("报告期", ""))
                if latest.get("报告期")
                else "",
            }
    except Exception as e:
        print(f"  [警告] 获取 {symbol} 财务摘要失败: {e}")

    return None


def fetch_cashflow_data(symbol):
    """获取现金流数据"""
    ak = akshare_patch.get_akshare()
    sym = normalize_symbol(symbol)

    try:
        # 获取现金流量表
        df = ak.stock_financial_report_sina(stock=sym, symbol="现金流量表")
        if df is not None and not df.empty:
            # 获取最近3年数据
            recent = df.head(3)
            fcf_list = []
            for _, row in recent.iterrows():
                ocf = (
                    float(row.get("经营活动产生的现金流量净额", 0))
                    if row.get("经营活动产生的现金流量净额")
                    else 0
                )
                capex = (
                    float(row.get("购建固定资产、无形资产和其他长期资产支付的现金", 0))
                    if row.get("购建固定资产、无形资产和其他长期资产支付的现金")
                    else 0
                )
                fcf = ocf - capex
                fcf_list.append(fcf)

            return {
                "symbol": symbol,
                "fcf_latest": fcf_list[0] if fcf_list else 0,
                "fcf_avg_3y": sum(fcf_list) / len(fcf_list) if fcf_list else 0,
                "fcf_positive_3y": sum(1 for f in fcf_list if f > 0),
            }
    except Exception as e:
        print(f"  [警告] 获取 {symbol} 现金流失败: {e}")

    return None


def fetch_main_business(symbol):
    """获取主营业务构成"""
    ak = akshare_patch.get_akshare()
    sym = normalize_symbol(symbol)

    try:
        df = ak.stock_zygc_ym(symbol=sym)
        if df is not None and not df.empty:
            latest = df.iloc[-1]
            return {
                "symbol": symbol,
                "main_business": str(latest.get("主营业务", ""))[:100],
                "business_description": str(latest.get("主营构成", ""))[:200],
            }
    except Exception:
        pass

    return None


def apply_advanced_screening(financial_data):
    """应用深度筛选条件"""
    results = []

    for data in financial_data:
        symbol = data.get("symbol")
        fin = data.get("financial", {})
        cf = data.get("cashflow", {})

        # 检查ROE
        roe = fin.get("roe")
        if roe is None or roe < MIN_ROE:
            continue

        # 检查营收增长
        rev_growth = fin.get("revenue_growth")
        if rev_growth is None or rev_growth < MIN_REVENUE_GROWTH:
            continue

        # 检查资产负债率（金融行业除外）
        debt_ratio = fin.get("debt_ratio")
        industry = data.get("industry", "")
        if "金融" not in industry and "银行" not in industry:
            if debt_ratio is None or debt_ratio > MAX_DEBT_RATIO:
                continue

        # 检查自由现金流
        fcf_positive = cf.get("fcf_positive_3y", 0)

        # 计算综合质量评分
        quality_score = 0
        if roe >= 15:
            quality_score += 30
        elif roe >= 10:
            quality_score += 20
        else:
            quality_score += 10

        if rev_growth >= 10:
            quality_score += 20
        elif rev_growth >= 0:
            quality_score += 10

        if debt_ratio and debt_ratio < 50:
            quality_score += 20
        elif debt_ratio and debt_ratio < 70:
            quality_score += 10

        if fcf_positive >= 2:
            quality_score += 30
        elif fcf_positive >= 1:
            quality_score += 15

        data["quality_score"] = quality_score
        results.append(data)

    return sorted(results, key=lambda x: x["quality_score"], reverse=True)


def main():
    print("=" * 80)
    print("A股低估值股票筛选 - 第二阶段：深度财务指标筛选")
    print("=" * 80)

    print(f"\n候选股票: {len(CANDIDATES)} 只")
    print(
        f"筛选标准: ROE >= {MIN_ROE}%, 营收增长 >= {MIN_REVENUE_GROWTH}%, 资产负债率 <= {MAX_DEBT_RATIO}%"
    )

    # 读取第一阶段的基础数据
    phase1_path = Path(__file__).resolve().parent / "screening_candidates.csv"
    if phase1_path.exists():
        phase1_df = pd.read_csv(phase1_path)
        base_data = phase1_df.set_index("symbol").to_dict("index")
    else:
        base_data = {}

    # 获取每只股票的详细财务数据
    financial_data = []

    for i, symbol in enumerate(CANDIDATES, 1):
        print(f"\n[{i}/{len(CANDIDATES)}] 分析 {symbol}...")

        data = {
            "symbol": symbol,
            "name": base_data.get(symbol, {}).get("name", ""),
            "industry": base_data.get(symbol, {}).get("industry", ""),
            "price": base_data.get(symbol, {}).get("price", 0),
            "pe": base_data.get(symbol, {}).get("pe_ttm", 0),
            "pb": base_data.get(symbol, {}).get("pb", 0),
        }

        # 获取财务摘要
        fin = fetch_financial_abstract(symbol)
        if fin:
            data["financial"] = fin
            print(
                f"  ROE: {fin.get('roe', 'N/A')}%, 营收增长: {fin.get('revenue_growth', 'N/A')}%"
            )
            print(
                f"  资产负债率: {fin.get('debt_ratio', 'N/A')}%, 毛利率: {fin.get('gross_margin', 'N/A')}%"
            )
        else:
            print(f"  [跳过] 无法获取财务数据")
            continue

        # 获取现金流数据
        cf = fetch_cashflow_data(symbol)
        if cf:
            data["cashflow"] = cf
            print(f"  自由现金流(3年正数): {cf.get('fcf_positive_3y', 0)} 年")

        financial_data.append(data)
        time.sleep(0.5)  # 避免请求过快

    # 应用深度筛选
    print("\n" + "=" * 80)
    print("应用深度筛选条件...")
    print("=" * 80)

    filtered = apply_advanced_screening(financial_data)
    print(f"\n通过深度筛选: {len(filtered)} 只股票")

    # 显示结果
    if filtered:
        print("\n" + "=" * 80)
        print("最终推荐标的（按质量评分排序）")
        print("=" * 80)

        for rank, stock in enumerate(filtered[:5], 1):
            fin = stock.get("financial", {})
            cf = stock.get("cashflow", {})

            print(
                f"\n【{rank}】{stock['symbol']} {stock['name']} ({stock['industry']})"
            )
            print(
                f"  股价: ¥{stock['price']:.2f} | PE: {stock['pe']:.2f} | PB: {stock['pb']:.2f}"
            )
            print(
                f"  ROE: {fin.get('roe', 'N/A'):.1f}% | 营收增长: {fin.get('revenue_growth', 'N/A'):.1f}%"
            )
            print(
                f"  资产负债率: {fin.get('debt_ratio', 'N/A'):.1f}% | 毛利率: {fin.get('gross_margin', 'N/A'):.1f}%"
            )
            print(f"  自由现金流3年正值: {cf.get('fcf_positive_3y', 0)} 年")
            print(f"  质量评分: {stock['quality_score']}/100")

    # 保存结果
    import json

    output_path = Path(__file__).resolve().parent / "screening_final_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)
    print(f"\n[保存] 完整结果已保存至: {output_path}")

    return filtered


if __name__ == "__main__":
    results = main()
