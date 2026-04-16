#!/usr/bin/env python3
"""
查询沪深300成分股的ESG评分，找出ESG评分最高的股票
"""

import sys

sys.path.insert(0, "skills/findata-toolkit-cn/scripts")

from common import akshare_patch

ak = akshare_patch.get_akshare()

# 更新所有ak.调用为直接使用ak
import pandas as pd

# 1. 获取沪深300成分股
print("正在获取沪深300成分股列表...")
hs300 = ak.index_stock_cons_weight_csindex(symbol="000300")
hs300_symbols = hs300["成分券代码"].tolist()
print(f"沪深300成分股数量: {len(hs300_symbols)}")

# 2. 获取ESG评级数据 (华证ESG)
print("\n正在获取ESG评级数据...")
try:
    esg_df = ak.stock_esg_hz_sina()
    print(f"ESG数据条数: {len(esg_df)}")
    print(f"ESG数据列: {esg_df.columns.tolist()}")
except Exception as e:
    print(f"获取ESG数据失败: {e}")
    esg_df = None

if esg_df is not None and len(esg_df) > 0:
    # 3. 处理股票代码格式
    # ESG数据中的代码格式可能与沪深300不同，需要匹配
    # 假设ESG数据中的代码列是"股票代码"
    code_col = None
    for col in esg_df.columns:
        if "代码" in col:
            code_col = col
            break

    if code_col:
        print(f"\n使用股票代码列: {code_col}")

        # 标准化沪深300代码（去掉后缀）
        hs300_clean = [s.split(".")[0] if "." in s else s for s in hs300_symbols]

        # 筛选出沪深300成分股的ESG数据
        esg_df["clean_code"] = (
            esg_df[code_col].astype(str).str.replace(r"\D", "", regex=True)
        )
        esg_hs300 = esg_df[esg_df["clean_code"].isin(hs300_clean)]

        print(f"\n沪深300成分股的ESG数据条数: {len(esg_hs300)}")

        # 4. 找出ESG评分最高的股票
        # 找到评分列
        score_col = None
        for col in esg_hs300.columns:
            if (
                "评级" in col
                or "评分" in col
                or "score" in col.lower()
                or "rating" in col.lower()
            ):
                score_col = col
                break

        if score_col:
            print(f"\n使用评分列: {score_col}")

            # 显示数据样本
            print(
                f"\n评分列前10行:\n{esg_hs300[[code_col, '股票名称', score_col]].head(10)}"
            )

            # 按评分排序（假设AAA最高，或者数值越大越好）
            # 先尝试数值排序
            try:
                esg_hs300["score_num"] = pd.to_numeric(
                    esg_hs300[score_col], errors="coerce"
                )
                top_esg = esg_hs300.nlargest(20, "score_num")
                print(f"\n{'=' * 60}")
                print("ESG评分最高的沪深300成分股 (Top 20):")
                print(f"{'=' * 60}")
                for idx, row in top_esg.iterrows():
                    print(
                        f"{row['clean_code']:>8} | {row['股票名称']:<10} | ESG评分: {row[score_col]}"
                    )
            except:
                # 文本评级排序
                rating_order = {
                    "AAA": 9,
                    "AA": 8,
                    "A": 7,
                    "BBB": 6,
                    "BB": 5,
                    "B": 4,
                    "CCC": 3,
                    "CC": 2,
                    "C": 1,
                }
                esg_hs300["score_num"] = esg_hs300[score_col].map(rating_order)
                top_esg = esg_hs300.nlargest(20, "score_num")
                print(f"\n{'=' * 60}")
                print("ESG评级最高的沪深300成分股 (Top 20):")
                print(f"{'=' * 60}")
                for idx, row in top_esg.iterrows():
                    print(
                        f"{row['clean_code']:>8} | {row['股票名称']:<10} | ESG评级: {row[score_col]}"
                    )
        else:
            print("未找到评分列，可用列名:", esg_hs300.columns.tolist())
    else:
        print("未找到股票代码列")
else:
    print("无法获取ESG数据")
