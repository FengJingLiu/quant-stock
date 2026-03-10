#!/usr/bin/env python3
"""
A股董事长/高管增持分析器
扫描全市场董监高增减持数据，识别董事长及核心高管大幅增持的信号
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

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

# 分析参数
DAYS_WINDOW = 90  # 近90天
MIN_AMOUNT = 100  # 最小增持金额100万元
TOP_N = 10  # Top 10


def normalize_symbol(symbol):
    """标准化代码格式"""
    sym = symbol.replace(".SH", "").replace(".SZ", "").replace(".BJ", "")
    return sym.zfill(6)


def get_exchange_prefix(symbol):
    """获取交易所前缀"""
    sym = normalize_symbol(symbol)
    if sym.startswith("6"):
        return f"SH{sym}"
    elif sym.startswith(("0", "3")):
        return f"SZ{sym}"
    elif sym.startswith(("4", "8")):
        return f"BJ{sym}"
    return f"SZ{sym}"


def fetch_all_insider_trades():
    """获取全市场董监高增减持数据"""
    ak = akshare_patch.get_akshare()

    try:
        # 获取最近的高管持股变动数据
        df = ak.stock_inner_trade_xq()
        if df is None or df.empty:
            return pd.DataFrame()

        # 数据清洗和转换
        df["变动日期"] = pd.to_datetime(df["变动日期"], errors="coerce")

        # 只保留近90天的数据
        cutoff_date = datetime.now() - timedelta(days=DAYS_WINDOW)
        df = df[df["变动日期"] >= cutoff_date]

        # 解析股票代码
        df["symbol"] = df["股票代码"].str.replace(r"^(SH|SZ|BJ)", "", regex=True)

        # 计算变动金额
        df["变动金额"] = df["变动股数"] * df["成交均价"]

        # 确定变动类型
        df["变动类型"] = df["变动股数"].apply(
            lambda x: "增持" if x > 0 else "减持" if x < 0 else "未知"
        )

        # 筛选增持
        df_buys = df[df["变动类型"] == "增持"].copy()

        return df_buys
    except Exception as e:
        print(f"[错误] 获取增减持数据失败: {e}")
        import traceback

        traceback.print_exc()
        return pd.DataFrame()


def get_stock_names(symbols):
    """获取股票名称"""
    ak = akshare_patch.get_akshare()
    names = {}

    try:
        df = ak.stock_zh_a_spot_em()
        if df is not None and not df.empty:
            for _, row in df.iterrows():
                code = str(row.get("代码", "")).zfill(6)
                name = str(row.get("名称", ""))
                if code and name:
                    names[code] = name
    except Exception as e:
        print(f"[警告] 获取股票名称失败: {e}")

    return names


def is_key_person(position, name, relationship):
    """判断是否为关键人物（董事长、总经理、实际控制人等）"""
    if not isinstance(position, str):
        position = str(position) if position else ""
    if not isinstance(relationship, str):
        relationship = str(relationship) if relationship else ""

    position = position.lower()
    relationship = relationship.lower()

    key_titles = [
        "董事长",
        "总经理",
        "总裁",
        "实际控制人",
        "控股股东",
        "财务总监",
        "董秘",
    ]
    key_relations = ["本人", "本人持股"]

    for title in key_titles:
        if title in position:
            return True, title

    for rel in key_relations:
        if (
            rel in relationship
            and "配偶" not in relationship
            and "子女" not in relationship
        ):
            return True, "本人持股"

    return False, ""


def analyze_insider_buys(df_buys):
    """分析增持数据，按公司聚合"""
    if df_buys.empty:
        return pd.DataFrame()

    results = []

    for symbol, group in df_buys.groupby("symbol"):
        # 统计该公司增持情况
        total_amount = group["变动金额"].sum()
        total_shares = group["变动股数"].sum()
        num_buyers = group["变动人"].nunique()

        # 识别关键人物增持
        key_person_buys = []
        for _, row in group.iterrows():
            is_key, title = is_key_person(
                row["董监高职务"], row["变动人"], row["与董监高关系"]
            )
            if is_key:
                key_person_buys.append(
                    {
                        "name": row["变动人"],
                        "title": row["董监高职务"],
                        "amount": row["变动金额"],
                        "shares": row["变动股数"],
                        "price": row["成交均价"],
                        "date": row["变动日期"],
                        "is_key": True,
                    }
                )

        key_person_amount = sum(b["amount"] for b in key_person_buys)

        # 只保留增持金额超过阈值的公司
        if total_amount >= MIN_AMOUNT * 10000:  # 转换为元
            results.append(
                {
                    "symbol": symbol,
                    "total_amount": total_amount,
                    "total_shares": total_shares,
                    "num_buyers": num_buyers,
                    "key_person_amount": key_person_amount,
                    "key_person_count": len(key_person_buys),
                    "latest_date": group["变动日期"].max(),
                    "key_person_details": key_person_buys[:3],  # 只保留前3个关键人物
                }
            )

    df_result = pd.DataFrame(results)
    if not df_result.empty:
        # 按关键人物增持金额排序
        df_result = df_result.sort_values("key_person_amount", ascending=False)

    return df_result


def fetch_basic_info(symbols):
    """获取股票基础信息"""
    ak = akshare_patch.get_akshare()
    info = {}

    try:
        df = ak.stock_zh_a_spot_em()
        if df is not None and not df.empty:
            for _, row in df.iterrows():
                code = str(row.get("代码", "")).zfill(6)
                if code in symbols:
                    info[code] = {
                        "name": str(row.get("名称", "")),
                        "price": safe_float(row.get("最新价")),
                        "market_cap": safe_float(row.get("总市值")),
                        "pe": safe_float(row.get("市盈率-动态")),
                        "pb": safe_float(row.get("市净率")),
                    }
    except Exception as e:
        print(f"[警告] 获取股票信息失败: {e}")

    return info


def main():
    print("=" * 80)
    print("A股董事长/高管增持分析器")
    print(f"时间窗口: 近{DAYS_WINDOW}天 | 最小增持金额: {MIN_AMOUNT}万元")
    print("=" * 80)

    print("\n[1/3] 获取全市场董监高增减持数据...")
    df_buys = fetch_all_insider_trades()
    if df_buys.empty:
        print("[错误] 未获取到数据")
        return

    print(f"      获取到 {len(df_buys)} 条增持记录")

    print("\n[2/3] 分析增持数据...")
    df_analysis = analyze_insider_buys(df_buys)
    if df_analysis.empty:
        print("[警告] 未找到符合条件的增持记录")
        return

    print(f"      筛选出 {len(df_analysis)} 家公司增持金额超过{MIN_AMOUNT}万元")

    # 获取股票名称和基本信息
    print("\n[3/3] 获取股票详细信息...")
    top_symbols = df_analysis.head(TOP_N)["symbol"].tolist()
    stock_info = fetch_basic_info(top_symbols)

    # 显示结果
    print("\n" + "=" * 80)
    print(f"董事长/核心高管大幅增持 Top {TOP_N}")
    print("=" * 80)

    for rank, (_, row) in enumerate(df_analysis.head(TOP_N).iterrows(), 1):
        symbol = row["symbol"]
        info = stock_info.get(symbol, {})

        print(f"\n【{rank}】{symbol} {info.get('name', 'N/A')}")
        print(
            f"  股价: ¥{info.get('price', 0):.2f} | 市值: {info.get('market_cap', 0) / 1e8:.1f}亿"
        )
        print(f"  PE: {info.get('pe', 0):.2f} | PB: {info.get('pb', 0):.2f}")
        print(f"  增持总人数: {row['num_buyers']}人")
        print(
            f"  关键人物增持: {row['key_person_count']}人 | 金额: ¥{row['key_person_amount'] / 1e4:.1f}万元"
        )
        print(f"  总增持金额: ¥{row['total_amount'] / 1e4:.1f}万元")
        print(
            f"  最新增持日期: {row['latest_date'].strftime('%Y-%m-%d') if pd.notna(row['latest_date']) else 'N/A'}"
        )

        # 显示关键人物详情
        if row["key_person_details"]:
            print(f"  关键人物增持详情:")
            for person in row["key_person_details"]:
                print(
                    f"    - {person['name']} ({person['title']}): {person['amount'] / 1e4:.1f}万元 @ ¥{person['price']:.2f}"
                )

    # 保存结果
    output_path = Path(__file__).resolve().parent / "insider_buying_analysis.csv"
    df_analysis.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n[保存] 完整分析结果已保存至: {output_path}")

    # 返回Top 5代码
    top5 = df_analysis.head(5)["symbol"].tolist()
    print("\n[输出] Top 5 增持标的代码：")
    print(", ".join(top5))

    return df_analysis


if __name__ == "__main__":
    df = main()
