"""
验证 ETF 簇共振因子 (ETFCluster) 的输出。

用途: 快速跑一小段数据，检查输出 schema 和分数分布。
"""

from __future__ import annotations

import sys
from datetime import date

import polars as pl

sys.path.insert(0, "/home/autumn/quant/stock")

from src.national_team.ch_client import get_etf_1m, get_index_1m
from src.national_team.etf_cluster import ETFCluster
from src.national_team.nt_buy_prob import NTBuyProb


def main() -> None:
    start = "2024-01-02"
    end = "2024-01-31"

    print("=" * 60)
    print(f"ETF Cluster Resonance — 验证 ({start} ~ {end})")
    print("=" * 60)

    # ── 1. ETFCluster 独立输出 ───────────────────────────────────────────
    cluster = ETFCluster()
    print(f"\nETF pool: {[s.symbol for s in cluster.etf_pool]}")
    print("Loading cluster data ...")
    cluster_df = cluster.compute(start, end)
    print(f"cluster_df shape: {cluster_df.shape}")
    print(f"columns: {cluster_df.columns}")
    print(cluster_df.describe())

    # 非零分数统计
    buy_nz = cluster_df.filter(pl.col("same_index_buy_score") > 0)
    sell_nz = cluster_df.filter(pl.col("same_index_sell_score") > 0)
    cross_nz = cluster_df.filter(pl.col("cross_index_follow") > 0)
    cnt_nz = cluster_df.filter(pl.col("cluster_resonance_count") > 0)
    total = cluster_df.height

    print(f"\n── 非零分数行数 / 总行数 ({total:,}) ──")
    print(f"  same_index_buy_score > 0:  {buy_nz.height:>6,}  ({buy_nz.height/total*100:.2f}%)")
    print(f"  same_index_sell_score > 0: {sell_nz.height:>6,}  ({sell_nz.height/total*100:.2f}%)")
    print(f"  cross_index_follow > 0:    {cross_nz.height:>6,}  ({cross_nz.height/total*100:.2f}%)")
    print(f"  cluster_resonance >= 1:    {cnt_nz.height:>6,}  ({cnt_nz.height/total*100:.2f}%)")

    # Top events
    print("\n── same_index_buy_score Top 10 ──")
    print(
        cluster_df
        .sort("same_index_buy_score", descending=True)
        .head(10)
        .select("datetime", "same_index_buy_score", "cross_index_follow",
                "primary_amt_z", "cluster_resonance_count")
    )

    # ── 2. NTBuyProb v3 (with cluster) ──────────────────────────────────
    print("\n" + "=" * 60)
    print("NTBuyProb v3 (with cluster_data)")
    print("=" * 60)

    etf_1m = get_etf_1m("510300.SH", start, end)
    idx_1m = get_index_1m("000300", start, end)

    # fleet_etf_data for resonance
    fleet = {}
    for sym in ["510050.SH", "510500.SH", "512100.SH"]:
        d = get_etf_1m(sym, start, end)
        if d.height > 0:
            fleet[sym] = d

    prob = NTBuyProb()

    # v2 baseline
    result_v2 = prob.compute(etf_1m, idx_1m, fleet)
    print(f"\nv2 result shape: {result_v2.shape}")
    v2_high = result_v2.filter(pl.col("nt_buy_prob") > 0.5)
    print(f"v2 prob > 0.5: {v2_high.height} rows")
    print(result_v2.select("nt_buy_prob").describe())

    # v3 with cluster
    result_v3 = prob.compute(etf_1m, idx_1m, fleet, cluster_data=cluster_df)
    print(f"\nv3 result shape: {result_v3.shape}")
    v3_high = result_v3.filter(pl.col("nt_buy_prob") > 0.5)
    print(f"v3 prob > 0.5: {v3_high.height} rows")
    print(result_v3.select("nt_buy_prob").describe())

    # 对比
    cmp = result_v2.select(
        "datetime",
        pl.col("nt_buy_prob").alias("prob_v2"),
    ).join(
        result_v3.select(
            "datetime",
            pl.col("nt_buy_prob").alias("prob_v3"),
        ),
        on="datetime",
        how="inner",
    )
    cmp = cmp.with_columns(
        (pl.col("prob_v3") - pl.col("prob_v2")).alias("delta"),
    )
    print("\n── v3 - v2 概率差 ──")
    print(cmp.select("delta").describe())

    # 高信号时刻
    print("\n── v3 prob > 0.5 的时刻 (前 20) ──")
    print(
        result_v3
        .filter(pl.col("nt_buy_prob") > 0.5)
        .sort("nt_buy_prob", descending=True)
        .head(20)
        .select("datetime", "trade_date", "nt_buy_prob",
                "stress_context", "vol_shock", "resonance_count")
    )

    print("\n✅ 验证完成")


if __name__ == "__main__":
    main()
