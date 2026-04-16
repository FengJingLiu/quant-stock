"""
因子二：国家队压盘出货因子 v2 (NT_Sell_Prob)

v2 改进 (基于方法论优化)
------------------------
1. distribution 替代 vol_price_divergence: vol_shock × max(-alpha, 0) 高位放量低效
2. wick_ratio 替代 ceiling_hit: 连续上影线占比 (非布尔计数)
3. vwap_fail (新增): close < 日内累计 VWAP → 跌回事件均价
4. lead_reversal (新增): lead_gap 符号翻转 (正→负) → ETF 领先失效
5. propagation_fail (新增): 近期大放量但价格未跟进 → 传导断裂

信号合成
--------
线性组合 + sigmoid (v1 先线性，后续可接 isotonic 校准):
  logit = w1*distribution + w2*wick + w3*vwap_fail + w4*lead_rev + w5*prop_fail + bias
  prob = sigmoid(logit)

使用
----
>>> from src.national_team.nt_sell_prob import NTSellProb
>>> factor = NTSellProb()
>>> result = factor.compute(etf_1m, index_1m)
>>> result.filter(pl.col("nt_sell_prob") > 0.5)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl


@dataclass
class NTSellProb:
    """国家队压盘出货因子 v2。"""

    # ── 数据参数 ─────────────────────────────────────────────────────────
    vol_lookback: int = 20
    """vol_shock 回看天数"""

    min_vol_periods: int = 10
    """同时间切片最少历史天数"""

    # ── propagation_fail 参数 ────────────────────────────────────────────
    prop_fail_window: int = 5
    """传导断裂检测回看 bar 数"""

    prop_fail_vol_q: float = 2.326
    """vol_shock 阈值 (≈正态 99 分位, 与 buy 侧共振阈值一致)"""

    weak_feat_vol_gate: float = 1.0
    """vwap_fail / lead_reversal 的 vol_shock 门槛 (>1σ 才激活)"""

    # ── sigmoid 组合权重 ─────────────────────────────────────────────────
    w_distribution: float = field(default=2.0, repr=False)
    """distribution 权重"""
    w_wick: float = field(default=1.5, repr=False)
    """wick_ratio 权重"""
    w_vwap_fail: float = field(default=1.5, repr=False)
    """vwap_fail 权重"""
    w_lead_rev: float = field(default=2.0, repr=False)
    """lead_reversal 权重"""
    w_prop_fail: float = field(default=1.5, repr=False)
    """propagation_fail 权重"""
    bias: float = field(default=-5.0, repr=False)
    """sigmoid 偏置项 (sigmoid(-5) ≈ 0.7%)"""

    ch_kwargs: dict | None = None

    # ── 公共接口 ─────────────────────────────────────────────────────────

    def compute(
        self,
        etf_1m: pl.DataFrame,
        index_1m: pl.DataFrame,
    ) -> pl.DataFrame:
        """
        计算 NT_Sell_Prob v2。

        Parameters
        ----------
        etf_1m : 目标 ETF 1 分钟 K 线 (Polars DataFrame)
        index_1m : 参考指数 1 分钟 K 线

        Returns
        -------
        Polars DataFrame with columns:
          datetime, trade_date, symbol,
          distribution, wick_ratio, vwap_fail,
          lead_reversal, propagation_fail, nt_sell_prob
        """
        df = etf_1m.clone()
        if df.height == 0:
            return self._empty_result()

        # 0. 基础准备: 排序去重 + 分钟收益率
        df = df.sort("datetime").unique(subset=["datetime"], maintain_order=True)
        df = df.with_columns(
            (pl.col("close") / pl.col("close").shift(1) - 1.0).alias("ret_etf"),
        )

        # 对齐指数分钟收益率
        idx = index_1m.sort("datetime").unique(subset=["datetime"], maintain_order=True)
        idx = idx.with_columns(
            (pl.col("close") / pl.col("close").shift(1) - 1.0).alias("ret_index"),
        )
        df = df.join(idx.select("datetime", "ret_index"), on="datetime", how="left")
        df = df.with_columns(
            pl.col("ret_index").fill_null(0.0).fill_nan(0.0),
        )

        # alpha = ret_etf - ret_index
        df = df.with_columns(
            (pl.col("ret_etf") - pl.col("ret_index")).alias("alpha"),
        )

        # lead_gap (用于 lead_reversal)
        df = df.with_columns(
            (pl.col("ret_etf") - pl.col("ret_index")).alias("lead_gap"),
        )

        # vol_shock (内部用，不输出)
        df = self._add_vol_shock(df)

        # 1. distribution = vol_shock × max(-alpha, 0)
        df = df.with_columns(
            (pl.col("vol_shock") * (-pl.col("alpha")).clip(lower_bound=0.0))
            .fill_null(0.0)
            .alias("distribution"),
        )

        # 2. wick_ratio = (high - max(open, close)) / (high - low + eps)
        df = self._add_wick_ratio(df)

        # 3. vwap_fail: close < 日内累计 VWAP AND vol_shock > 1.0
        df = self._add_vwap_fail(df)
        df = df.with_columns(
            (pl.col("vwap_fail") * (pl.col("vol_shock") > self.weak_feat_vol_gate).cast(pl.Float64))
            .alias("vwap_fail"),
        )

        # 4. lead_reversal: lead_gap 从正翻负 AND vol_shock > 1.0
        df = df.with_columns(
            (
                (pl.col("lead_gap").shift(1) > 0)
                & (pl.col("lead_gap") < 0)
                & (pl.col("vol_shock") > self.weak_feat_vol_gate)
            )
            .cast(pl.Float64)
            .fill_null(0.0)
            .alias("lead_reversal"),
        )

        # 5. propagation_fail: 近期高 vol_shock 但价格未跟涨
        df = self._add_propagation_fail(df)

        # 6. sigmoid 线性组合
        df = self._sigmoid_combine(df)

        return df.select(
            "datetime", "trade_date", "symbol",
            "distribution", "wick_ratio", "vwap_fail",
            "lead_reversal", "propagation_fail", "nt_sell_prob",
        )

    @staticmethod
    def rolling_daily_prob(
        result: pl.DataFrame,
        window: int = 3,
    ) -> pl.DataFrame:
        """
        对 compute() 输出做多日滚动累积。

        每日取 nt_sell_prob 最大值，然后做 *window* 日滚动求和。
        """
        daily = (
            result
            .group_by("trade_date")
            .agg(pl.col("nt_sell_prob").max().alias("daily_max_prob"))
            .sort("trade_date")
        )
        return daily.with_columns(
            pl.col("daily_max_prob")
            .rolling_sum(window_size=window, min_periods=1)
            .alias("roll_sum_prob"),
        )

    # ── 内部计算方法 ──────────────────────────────────────────────────────

    def _add_vol_shock(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        vol_shock: robust_zscore(log(amount+1)) 季节性标准化。
        与 buy 侧相同的计算逻辑，distribution 复用此值。
        """
        eps = 1e-10
        return (
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
                .alias("vol_shock"),
            )
            .drop("_la", "_tb", "_vm", "_vs")
        )

    @staticmethod
    def _add_wick_ratio(df: pl.DataFrame) -> pl.DataFrame:
        """
        上影线占比: (high - max(open, close)) / (high - low + eps)。
        连续值 0~1，比布尔 ceiling_hit 更细腻。
        """
        eps = 1e-10
        body_top = pl.max_horizontal("close", "open")
        bar_range = pl.col("high") - pl.col("low") + eps
        return df.with_columns(
            ((pl.col("high") - body_top) / bar_range)
            .clip(0.0, 1.0)
            .fill_null(0.0)
            .alias("wick_ratio"),
        )

    @staticmethod
    def _add_vwap_fail(df: pl.DataFrame) -> pl.DataFrame:
        """
        VWAP 失败: close < 日内累计 VWAP → 价格跌回均价下方。

        日内累计 VWAP = cumsum(amount) / cumsum(volume × 100)。
        volume 单位是手，amount 单位是元。
        """
        return (
            df.with_columns(
                pl.col("amount").cum_sum().over("trade_date").alias("_cum_amt"),
                (pl.col("volume") * 100)
                .cum_sum()
                .over("trade_date")
                .alias("_cum_vol"),
            )
            .with_columns(
                (
                    pl.col("_cum_amt")
                    / pl.when(pl.col("_cum_vol") > 0)
                    .then(pl.col("_cum_vol"))
                    .otherwise(None)
                )
                .fill_null(pl.col("close"))
                .alias("_session_vwap"),
            )
            .with_columns(
                (pl.col("close") < pl.col("_session_vwap"))
                .cast(pl.Float64)
                .alias("vwap_fail"),
            )
            .drop("_cum_amt", "_cum_vol", "_session_vwap")
        )

    def _add_propagation_fail(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        传导断裂: 近 N bar 有大放量 (vol_shock > q95) 但累积收益为负。

        即：国家队放了量，但没有把价格推上去 → 可能是在出货。
        """
        w = self.prop_fail_window
        return (
            df.with_columns(
                pl.col("vol_shock")
                .rolling_max(window_size=w, min_periods=1)
                .alias("_vs_max"),
                pl.col("ret_etf")
                .rolling_sum(window_size=w, min_periods=1)
                .alias("_ret_sum"),
            )
            .with_columns(
                (
                    (pl.col("_vs_max") > self.prop_fail_vol_q)
                    & (pl.col("_ret_sum") < 0)
                )
                .cast(pl.Float64)
                .fill_null(0.0)
                .alias("propagation_fail"),
            )
            .drop("_vs_max", "_ret_sum")
        )

    def _sigmoid_combine(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        线性组合 + sigmoid 输出。

        特征归一化:
          distribution: clip(0, 0.1) / 0.1  (分钟级 vol_shock×alpha 量级 ~0.001-0.05)
          wick_ratio: 已经 0~1
          vwap_fail: 0 或 1
          lead_reversal: 0 或 1
          propagation_fail: 0 或 1
        """
        logit = (
            self.w_distribution
            * (pl.col("distribution").clip(0.0, 0.1) / 0.1)
            + self.w_wick * pl.col("wick_ratio")
            + self.w_vwap_fail * pl.col("vwap_fail")
            + self.w_lead_rev * pl.col("lead_reversal")
            + self.w_prop_fail * pl.col("propagation_fail")
            + self.bias
        )
        return df.with_columns(
            (1.0 / (1.0 + (-logit).exp())).clip(0.0, 1.0).alias("nt_sell_prob"),
        )

    @staticmethod
    def _empty_result() -> pl.DataFrame:
        return pl.DataFrame(schema={
            "datetime": pl.Datetime("us"),
            "trade_date": pl.Date,
            "symbol": pl.Utf8,
            "distribution": pl.Float64,
            "wick_ratio": pl.Float64,
            "vwap_fail": pl.Float64,
            "lead_reversal": pl.Float64,
            "propagation_fail": pl.Float64,
            "nt_sell_prob": pl.Float64,
        })
