"""
同指数 ETF 簇共振因子 (ETF_Cluster_Resonance)

核心方案
--------
把"单 ETF 异动"升级为"同指数产品簇共振"。

分层结构:
  A. 主传感器 (1只) — 510300.SH → 事件时间锚点
  B. 同指数确认 (2-3只) — 159919.SZ / 159925.SZ → 降噪 + 共振确认
  C. 跨指数确认 (1-2只) — 510050.SH / 510500.SH → 扩散判断

每只 ETF 标准化特征:
  - amt_z     : robust_zscore(log(amount+1)) 季节性对齐
  - ret_1m    : 分钟收益率
  - lead_gap  : ret_etf - shadow_index_ret (对现货的剪刀差)
  - wick_ratio: (high - max(open,close)) / (high - low) 上影线占比

同指数聚合:
  same_index_buy_score  = Σ w_e * 1[amt_z > q99] * clip(zscore(lead_gap), 0, 5)
  same_index_sell_score = Σ w_e * 1[amt_z > q99 & wick_ratio > 0.6]

跨指数确认:
  cross_index_follow = 跨指数 ETF 在同指数触发后 1-5 分钟内是否也出现放量

使用:
    from src.national_team.etf_cluster import ETFCluster, ETF_POOL_HS300
    cluster = ETFCluster()
    result = cluster.compute(start_date, end_date)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import NamedTuple

import polars as pl

from src.national_team.ch_client import get_etf_1m, get_index_1m


# ── ETF 池定义 ────────────────────────────────────────────────────────────

class ETFSpec(NamedTuple):
    """ETF 配置。"""
    symbol: str
    role: str          # "primary" | "same_index" | "cross_index"
    track_index: str   # 跟踪的指数代码 (000300 / 000016 / 000905)
    weight: float      # 流动性权重 (同角色内归一化)


# 沪深300 为主的默认池
ETF_POOL_HS300: list[ETFSpec] = [
    # A. 主传感器
    ETFSpec("510300.SH", "primary",     "000300", 1.0),   # noqa: E241
    # B. 同指数确认
    ETFSpec("159919.SZ", "same_index",  "000300", 0.6),   # noqa: E241
    ETFSpec("159925.SZ", "same_index",  "000300", 0.4),   # noqa: E241
    # C. 跨指数确认
    ETFSpec("510050.SH", "cross_index", "000016", 0.5),   # noqa: E241
    ETFSpec("510500.SH", "cross_index", "000905", 0.5),   # noqa: E241
]


@dataclass
class ETFCluster:
    """同指数 ETF 簇共振因子。"""

    etf_pool: list[ETFSpec] = None  # type: ignore[assignment]  # set in __post_init__

    vol_lookback: int = 20
    """amt_z 回看天数"""

    min_vol_periods: int = 10
    """同时间切片最少历史天数"""

    amt_z_threshold: float = 2.326
    """触发异常放量的 z-score (≈ 正态 99 分位)"""

    wick_ratio_threshold: float = 0.6
    """卖出信号: 上影线占比阈值"""

    cross_follow_window: int = 5
    """跨指数跟随: 在同指数触发后 N 分钟内算跟随"""

    ch_kwargs: dict | None = None

    def __post_init__(self) -> None:
        if self.etf_pool is None:
            self.etf_pool = list(ETF_POOL_HS300)

    # ── 公共接口 ─────────────────────────────────────────────────────────

    def compute(
        self,
        start_date: str | date,
        end_date: str | date,
    ) -> pl.DataFrame:
        """
        计算 ETF 簇共振因子。

        Returns
        -------
        DataFrame with columns:
          datetime, trade_date,
          same_index_buy_score, same_index_sell_score,
          cross_index_follow, primary_amt_z, primary_lead_gap,
          cluster_resonance_count
        """
        # 1. 获取所有需要的指数数据 (用于计算 lead_gap)
        index_rets = self._load_index_rets(start_date, end_date)

        # 2. 加载每只 ETF 并计算标准化特征
        etf_features: dict[str, pl.DataFrame] = {}
        for spec in self.etf_pool:
            raw = get_etf_1m(spec.symbol, start_date, end_date, self.ch_kwargs)
            if raw.height == 0:
                continue
            raw = raw.sort("datetime").unique(subset=["datetime"], maintain_order=True)
            features = self._compute_etf_features(raw, index_rets.get(spec.track_index))
            etf_features[spec.symbol] = features

        if not etf_features:
            return self._empty_result()

        # 3. 以主传感器时间轴为基准
        primary_specs = [s for s in self.etf_pool if s.role == "primary"]
        if not primary_specs or primary_specs[0].symbol not in etf_features:
            return self._empty_result()

        primary_sym = primary_specs[0].symbol
        base = etf_features[primary_sym].select("datetime", "trade_date").clone()

        # 带上主传感器特征
        p_feats = etf_features[primary_sym].select(
            "datetime",
            pl.col("amt_z").alias("primary_amt_z"),
            pl.col("lead_gap").alias("primary_lead_gap"),
            pl.col("ret_1m").alias("primary_ret"),
            pl.col("wick_ratio").alias("primary_wick"),
        )
        base = base.join(p_feats, on="datetime", how="left")

        # 4. 同指数聚合
        base = self._aggregate_same_index(base, etf_features)

        # 5. 跨指数跟随
        base = self._aggregate_cross_index(base, etf_features)

        # 6. 综合共振计数
        base = self._compute_cluster_count(base, etf_features)

        return base.select(
            "datetime", "trade_date",
            "same_index_buy_score", "same_index_sell_score",
            "cross_index_follow",
            "primary_amt_z", "primary_lead_gap",
            "cluster_resonance_count",
        )

    def compute_with_etf_data(
        self,
        primary_etf_1m: pl.DataFrame,
        index_1m: pl.DataFrame,
        fleet_etf_data: dict[str, pl.DataFrame] | None = None,
    ) -> pl.DataFrame:
        """
        从已有的 DataFrame 计算（不查 ClickHouse），与 NTBuyProb.compute 签名兼容。

        Parameters
        ----------
        primary_etf_1m : 主 ETF 1m K 线
        index_1m : 主指数 1m K 线
        fleet_etf_data : {symbol: 1m_df} 所有 ETF 数据 (含主 ETF)
        """
        if primary_etf_1m.height == 0:
            return self._empty_result()

        # 构造 index_rets
        idx = index_1m.sort("datetime").unique(subset=["datetime"], maintain_order=True)
        idx = idx.with_columns(
            (pl.col("close") / pl.col("close").shift(1) - 1.0).alias("ret_index"),
        )
        primary_index = self.etf_pool[0].track_index if self.etf_pool else "000300"
        index_rets = {primary_index: idx.select("datetime", "ret_index")}

        # 构造 etf_features
        etf_features: dict[str, pl.DataFrame] = {}

        # 主 ETF
        primary_sym = self.etf_pool[0].symbol if self.etf_pool else "510300.SH"
        raw = primary_etf_1m.sort("datetime").unique(subset=["datetime"], maintain_order=True)
        etf_features[primary_sym] = self._compute_etf_features(raw, index_rets.get(primary_index))

        # 其他 ETF
        if fleet_etf_data:
            for spec in self.etf_pool:
                if spec.symbol == primary_sym:
                    continue
                if spec.symbol in fleet_etf_data:
                    edf = fleet_etf_data[spec.symbol]
                    edf = edf.sort("datetime").unique(subset=["datetime"], maintain_order=True)
                    # 跨指数 ETF 用同一个 index_ret 近似 (beta ≈ high corr)
                    etf_features[spec.symbol] = self._compute_etf_features(
                        edf, index_rets.get(spec.track_index, index_rets.get(primary_index)),
                    )

        if not etf_features:
            return self._empty_result()

        # 以主 ETF 为基准
        base = etf_features[primary_sym].select("datetime", "trade_date").clone()
        p_feats = etf_features[primary_sym].select(
            "datetime",
            pl.col("amt_z").alias("primary_amt_z"),
            pl.col("lead_gap").alias("primary_lead_gap"),
            pl.col("ret_1m").alias("primary_ret"),
            pl.col("wick_ratio").alias("primary_wick"),
        )
        base = base.join(p_feats, on="datetime", how="left")
        base = self._aggregate_same_index(base, etf_features)
        base = self._aggregate_cross_index(base, etf_features)
        base = self._compute_cluster_count(base, etf_features)

        return base.select(
            "datetime", "trade_date",
            "same_index_buy_score", "same_index_sell_score",
            "cross_index_follow",
            "primary_amt_z", "primary_lead_gap",
            "cluster_resonance_count",
        )

    # ── 单 ETF 标准化特征 ────────────────────────────────────────────────

    def _compute_etf_features(
        self,
        df: pl.DataFrame,
        index_ret_df: pl.DataFrame | None,
    ) -> pl.DataFrame:
        """
        为单只 ETF 计算 5 个标准化特征:
          amt_z, ret_1m, lead_gap, wick_ratio, is_bullish
        """
        eps = 1e-10

        # amt_z: robust_zscore(log(amount+1)) 季节性对齐
        df = (
            df.with_columns(
                (pl.col("amount") + 1.0).log().alias("_la"),
                pl.col("datetime").dt.time().alias("_tb"),
            )
            .with_columns(
                pl.col("_la").shift(1)
                .rolling_mean(
                    window_size=self.vol_lookback,
                    min_periods=self.min_vol_periods,
                )
                .over("_tb")
                .alias("_vm"),
                pl.col("_la").shift(1)
                .rolling_std(
                    window_size=self.vol_lookback,
                    min_periods=self.min_vol_periods,
                )
                .over("_tb")
                .alias("_vs"),
            )
            .with_columns(
                (
                    (pl.col("_la") - pl.col("_vm"))
                    / pl.when(pl.col("_vs") < eps)
                    .then(None)
                    .otherwise(pl.col("_vs"))
                )
                .fill_nan(0.0)
                .fill_null(0.0)
                .alias("amt_z"),
            )
            .drop("_la", "_tb", "_vm", "_vs")
        )

        # ret_1m: 分钟收益率
        df = df.with_columns(
            (pl.col("close") / pl.col("close").shift(1) - 1.0)
            .fill_null(0.0)
            .alias("ret_1m"),
        )

        # lead_gap: ret_etf - ret_index
        if index_ret_df is not None:
            df = df.join(index_ret_df, on="datetime", how="left")
            df = df.with_columns(
                (pl.col("ret_1m") - pl.col("ret_index").fill_null(0.0))
                .alias("lead_gap"),
            ).drop("ret_index")
        else:
            df = df.with_columns(pl.lit(0.0).alias("lead_gap"))

        # wick_ratio: (high - max(open,close)) / (high - low)
        df = df.with_columns(
            pl.when((pl.col("high") - pl.col("low")).abs() < eps)
            .then(0.0)
            .otherwise(
                (pl.col("high") - pl.max_horizontal("open", "close"))
                / (pl.col("high") - pl.col("low"))
            )
            .clip(0.0, 1.0)
            .alias("wick_ratio"),
        )

        # is_bullish
        df = df.with_columns(
            (pl.col("close") > pl.col("open")).alias("is_bullish"),
        )

        return df

    # ── 同指数聚合 ───────────────────────────────────────────────────────

    def _aggregate_same_index(
        self,
        base: pl.DataFrame,
        etf_features: dict[str, pl.DataFrame],
    ) -> pl.DataFrame:
        """
        同指数买入/卖出聚合分数:
          buy:  Σ w * 1[amt_z > q99] * clip(lead_gap_z, 0, 5)
          sell: Σ w * 1[amt_z > q99 & wick > thresh]
        """
        same_specs = [s for s in self.etf_pool if s.role in ("primary", "same_index")]

        # 归一化同指数权重
        w_total = sum(s.weight for s in same_specs if s.symbol in etf_features)
        if w_total <= 0:
            return base.with_columns(
                pl.lit(0.0).alias("same_index_buy_score"),
                pl.lit(0.0).alias("same_index_sell_score"),
            )

        buy_parts: list[pl.Expr] = []
        sell_parts: list[pl.Expr] = []

        for spec in same_specs:
            if spec.symbol not in etf_features:
                continue
            w = spec.weight / w_total
            safe = spec.symbol.replace(".", "_")

            feats = etf_features[spec.symbol].select(
                "datetime",
                pl.col("amt_z").alias(f"_az_{safe}"),
                pl.col("lead_gap").alias(f"_lg_{safe}"),
                pl.col("wick_ratio").alias(f"_wr_{safe}"),
                pl.col("is_bullish").alias(f"_bl_{safe}"),
            )
            base = base.join(feats, on="datetime", how="left")
            base = base.with_columns(
                pl.col(f"_az_{safe}").fill_null(0.0),
                pl.col(f"_lg_{safe}").fill_null(0.0),
                pl.col(f"_wr_{safe}").fill_null(0.0),
                pl.col(f"_bl_{safe}").fill_null(False),
            )

            # buy: w * 1[amt_z > q99 & bullish] * clip(lead_gap * 100, 0, 5)
            buy_parts.append(
                pl.lit(w)
                * pl.when(
                    (pl.col(f"_az_{safe}") > self.amt_z_threshold)
                    & pl.col(f"_bl_{safe}")
                )
                .then(
                    (pl.col(f"_lg_{safe}") * 100.0).clip(0.0, 5.0)
                )
                .otherwise(0.0)
            )

            # sell: w * 1[amt_z > q99 & wick > thresh & !bullish]
            sell_parts.append(
                pl.lit(w)
                * pl.when(
                    (pl.col(f"_az_{safe}") > self.amt_z_threshold)
                    & (pl.col(f"_wr_{safe}") > self.wick_ratio_threshold)
                    & ~pl.col(f"_bl_{safe}")
                )
                .then(1.0)
                .otherwise(0.0)
            )

        buy_expr = buy_parts[0]
        for p in buy_parts[1:]:
            buy_expr = buy_expr + p

        sell_expr = sell_parts[0]
        for p in sell_parts[1:]:
            sell_expr = sell_expr + p

        base = base.with_columns(
            buy_expr.alias("same_index_buy_score"),
            sell_expr.alias("same_index_sell_score"),
        )

        # 清理临时列
        tmp_cols = [c for c in base.columns if c.startswith(("_az_", "_lg_", "_wr_", "_bl_"))]
        return base.drop(tmp_cols)

    # ── 跨指数跟随 ───────────────────────────────────────────────────────

    def _aggregate_cross_index(
        self,
        base: pl.DataFrame,
        etf_features: dict[str, pl.DataFrame],
    ) -> pl.DataFrame:
        """
        跨指数跟随: 同指数触发后 N 分钟内，跨指数 ETF 是否也放量。

        同指数触发 = same_index_buy_score > 0
        跨指数跟随 = 在触发后 cross_follow_window 分钟内出现 amt_z > threshold
        """
        cross_specs = [s for s in self.etf_pool if s.role == "cross_index"]

        if not cross_specs:
            return base.with_columns(pl.lit(0.0).alias("cross_index_follow"))

        w_total = sum(s.weight for s in cross_specs if s.symbol in etf_features)
        if w_total <= 0:
            return base.with_columns(pl.lit(0.0).alias("cross_index_follow"))

        follow_parts: list[pl.Expr] = []

        for spec in cross_specs:
            if spec.symbol not in etf_features:
                continue
            w = spec.weight / w_total
            safe = spec.symbol.replace(".", "_")

            feats = etf_features[spec.symbol].select(
                "datetime",
                pl.col("amt_z").alias(f"_caz_{safe}"),
                pl.col("is_bullish").alias(f"_cbl_{safe}"),
            )
            base = base.join(feats, on="datetime", how="left")
            base = base.with_columns(
                pl.col(f"_caz_{safe}").fill_null(0.0),
                pl.col(f"_cbl_{safe}").fill_null(False),
            )

            # 跨指数是否也放量 (当前或近 N 分钟内)
            # 简化: 用 rolling_max 检查近 N 分钟的 amt_z
            trigger = (
                pl.col(f"_caz_{safe}")
                .rolling_max(window_size=self.cross_follow_window, min_periods=1)
            )
            follow_parts.append(
                pl.lit(w)
                * pl.when(
                    (trigger > self.amt_z_threshold)
                    & pl.col(f"_cbl_{safe}")
                )
                .then(1.0)
                .otherwise(0.0)
            )

        follow_expr = follow_parts[0]
        for p in follow_parts[1:]:
            follow_expr = follow_expr + p

        base = base.with_columns(follow_expr.alias("cross_index_follow"))

        tmp_cols = [c for c in base.columns if c.startswith(("_caz_", "_cbl_"))]
        return base.drop(tmp_cols)

    # ── 综合共振计数 ─────────────────────────────────────────────────────

    def _compute_cluster_count(
        self,
        base: pl.DataFrame,
        etf_features: dict[str, pl.DataFrame],
    ) -> pl.DataFrame:
        """统计同一分钟有多少只 ETF 满足 amt_z > threshold & bullish。"""
        count_parts: list[pl.Expr] = []

        for spec in self.etf_pool:
            if spec.symbol not in etf_features:
                continue
            safe = spec.symbol.replace(".", "_")
            col_name = f"_cnt_{safe}"

            feats = etf_features[spec.symbol].select(
                "datetime",
                (
                    (pl.col("amt_z") > self.amt_z_threshold)
                    & pl.col("is_bullish")
                ).cast(pl.Int32).alias(col_name),
            )
            base = base.join(feats, on="datetime", how="left")
            base = base.with_columns(pl.col(col_name).fill_null(0))
            count_parts.append(pl.col(col_name))

        if count_parts:
            base = base.with_columns(
                pl.sum_horizontal(*count_parts).cast(pl.Int32).alias("cluster_resonance_count"),
            )
            tmp_cols = [c for c in base.columns if c.startswith("_cnt_")]
            base = base.drop(tmp_cols)
        else:
            base = base.with_columns(pl.lit(0).cast(pl.Int32).alias("cluster_resonance_count"))

        return base

    # ── 数据加载 ─────────────────────────────────────────────────────────

    def _load_index_rets(
        self,
        start_date: str | date,
        end_date: str | date,
    ) -> dict[str, pl.DataFrame]:
        """加载所有需要的指数分钟收益率。"""
        needed_indices = {s.track_index for s in self.etf_pool}
        result: dict[str, pl.DataFrame] = {}

        for idx_sym in needed_indices:
            idx_1m = get_index_1m(idx_sym, start_date, end_date, self.ch_kwargs)
            if idx_1m.height == 0:
                continue
            idx_1m = idx_1m.sort("datetime").unique(subset=["datetime"], maintain_order=True)
            idx_1m = idx_1m.with_columns(
                (pl.col("close") / pl.col("close").shift(1) - 1.0)
                .fill_null(0.0)
                .alias("ret_index"),
            )
            result[idx_sym] = idx_1m.select("datetime", "ret_index")

        return result

    @staticmethod
    def _empty_result() -> pl.DataFrame:
        return pl.DataFrame(schema={
            "datetime": pl.Datetime("us"),
            "trade_date": pl.Date,
            "same_index_buy_score": pl.Float64,
            "same_index_sell_score": pl.Float64,
            "cross_index_follow": pl.Float64,
            "primary_amt_z": pl.Float64,
            "primary_lead_gap": pl.Float64,
            "cluster_resonance_count": pl.Int32,
        })
