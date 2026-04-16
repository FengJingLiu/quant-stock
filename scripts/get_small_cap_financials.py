#!/usr/bin/env python3
"""
获取Top小盘成长股的详细财务数据
"""

import sys
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
from common.utils import safe_float

# Top 5小盘成长股
TOP5 = [
    "300193.SZ",  # 佳士科技
    "920523.BJ",  # 德瑞锂电
    "000885.SZ",  # 城发环境
    "600526.SH",  # 菲达环保
    "002616.SZ",  # 长青集团
]


def normalize_symbol(symbol):
    """标准化代码"""
    return symbol.replace(".SH", "").replace(".SZ", "").replace(".BJ", "")


def fetch_financial_metrics(symbol):
    """获取财务指标"""
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
                "gross_margin": parse_percent(latest.get("销售毛利率")),
                "net_margin": parse_percent(latest.get("销售净利率")),
                "debt_ratio": parse_percent(latest.get("资产负债率")),
            }
    except Exception as e:
        print(f"  [警告] 获取 {symbol} 财务摘要失败: {e}")

    return None


def parse_percent(value):
    """解析百分比"""
    if value is None or value == "" or value == "-":
        return None
    if isinstance(value, str):
        value = value.replace("%", "").replace(",", "").strip()
        try:
            return float(value)
        except:
            return None
    return float(value) if value else None


def main():
    print("=" * 80)
    print("获取Top 5小盘成长股详细财务数据")
    print("=" * 80)

    results = []
    for i, symbol in enumerate(TOP5, 1):
        print(f"\n[{i}/5] 分析 {symbol}...")
        metrics = fetch_financial_metrics(symbol)
        if metrics:
            print(f"  ROE: {metrics.get('roe', 'N/A')}%")
            print(f"  营收增长: {metrics.get('revenue_growth', 'N/A')}%")
            print(f"  净利润增长: {metrics.get('profit_growth', 'N/A')}%")
            print(f"  毛利率: {metrics.get('gross_margin', 'N/A')}%")
            print(f"  资产负债率: {metrics.get('debt_ratio', 'N/A')}%")
            results.append(metrics)
        time.sleep(0.5)

    # 保存结果
    df = pd.DataFrame(results)
    output_path = Path(__file__).resolve().parent / "small_cap_financials.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n[保存] 财务数据已保存至: {output_path}")


if __name__ == "__main__":
    main()
