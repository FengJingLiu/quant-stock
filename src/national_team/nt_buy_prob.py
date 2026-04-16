"""
因子一：国家队护盘买入因子 v2 (NT_Buy_Prob)

v2 改进 (基于方法论优化)
------------------------
1. stress_context 替代 macro_bear: 多日回撤 + 均线偏离，而非单日日内跌幅
2. vol_shock 替代 vol_zscore: robust_zscore(log(amount+1)) 季节性标准化
3. absorption 替代 price_resilience: vol_shock × max(alpha, 0) 量化被托住的程度
4. resonance 升级: vol_shock > q99 AND alpha > 0 (非仅成交量放大+收阳)
5. lead_gap 替代 spread_zscore: ETF 收益率 - 指数收益率 (原始剪刀差)

信号合成
--------
线性组合 + sigmoid 替代贝叶斯似然比（v1 先线性，后续可接 isotonic 校准）:
  logit = w1*stress + w2*vol_shock + w3*absorption + w4*resonance + w5*lead_gap + bias
  prob = sigmoid(logit)

使用
----
>>> from src.national_team.nt_buy_prob import NTBuyProb
>>> factor = NTBuyProb()
>>> result = factor.compute(etf_1m, index_1m, fleet_etf_data)
>>> result.filter(pl.col("nt_buy_prob") > 0.5)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl


# 核心宽基 ETF → 对应现货指数
DEFAULT_ETF_INDEX_PAIRS = [
    ("510300.SH", "000300"),  # 沪深300
    ("510050.SH", "000016"),  # 上证50
    ("510500.SH", "000905"),  # 中证500
    ("512100.SH", "000852"),  # 中证1000
]


@dataclass
class NTBuyProb:
    """国家队护盘买入因子 v2。"""

    # ── 数据参数 ─────────────────────────────────────────────────────────
    etf_index_pairs: list[tuple[str, str]] | None = None
    """ETF→指数配对，用于共振计算"""

    vol_lookback: int = 20
    """vol_shock 回看天数"""

    min_vol_periods: int = 10
    """同时间切片最少历史天数"""

    # ── stress_context 参数 ──────────────────────────────────────────────
    stress_ma_window: int = 20
    """指数均线偏离计算窗口"""

    stress_drawdown_days: int = 5
    """回撤观察天数"""

    stress_ma_dev_thresh: float = -0.03
    """均线偏离阈值 (-3%)，低于此视为压力"""

    stress_drawdown_thresh: float = -0.05
    """N日回撤阈值 (-5%)，低于此视为压力"""

    # ── vol_shock / resonance 参数 ───────────────────────────────────────
    vol_shock_q99: float = 2.326
    """触发共振的 z-score 阈值 (≈正态分布 99 分位)"""

    # ── sigmoid 组合权重 ─────────────────────────────────────────────────
    w_stress: float = field(default=2.0, repr=False)
    """stress_context 权重"""
    w_vol_shock: float = field(default=1.5, repr=False)
    """vol_shock 权重"""
    w_absorption: float = field(default=2.0, repr=False)
    """absorption 权重"""
    w_resonance: float = field(default=1.5, repr=False)
    """resonance_count 权重"""
    w_lead_gap: float = field(default=1.0, repr=False)
    """lead_gap 权重"""
    bias: float = field(default=-5.0, repr=False)
    """sigmoid 偏置项 (sigmoid(-5) ≈ 0.7%)"""

    ch_kwargs: dict | None = None

    def __post_init__(self) -> None:
        if self.etf_index_pairs is None:
            self.etf_index_pairs = list(DEFAULT_ETF_INDEX_PAIRS)

    # ── 公共接口 ─────────────────────────────────────────────────────────

    def compute(
        self,
        etf_1m: pl.DataFrame,
        index_1m: pl.DataFrame,
        fleet_etf_data: dict[str, pl.DataFrame] | None = None,
    ) -> pl.DataFrame:
        """
        计算 NT_Buy_Prob v2。

        Parameters
        ----------
        etf_1m : 目标 ETF 1 分钟 K 线 (Polars DataFrame)
        index_1m : 参考指数 1 分钟 K 线
        fleet_etf_data : {etf_symbol: 1m_df} 舰队 ETF 数据，用于共振计算

        Returns
        -------
        Polars DataFrame with columns:
          datetime, trade_date, symbol,
          stress_context, vol_shock, absorption,
          resonance_count, lead_gap, nt_buy_prob
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

        # alpha = ret_etf - ret_index (beta≈1 for ETF tracking index)
        df = df.with_columns(
            (pl.col("ret_etf") - pl.col("ret_index")).alias("alpha"),
        )

        # 1. stress_context: 多日均线偏离 + 回撤
        df = self._add_stress_context(df, index_1m)

        # 2. vol_shock: robust_zscore(log(amount+1)) 季节性标准化
        df = self._add_vol_shock(df)

        # 3. absorption = vol_shock × max(alpha, 0)
        df = df.with_columns(
            (pl.col("vol_shock") * pl.col("alpha").clip(lower_bound=0.0))
            .fill_null(0.0)
            .alias("absorption"),
        )

        # 4. resonance: 跨 ETF vol_shock > q99 AND alpha > 0 计数
        df = self._add_resonance(df, fleet_etf_data, idx)

        # 5. lead_gap = ret_etf - ret_index (原始剪刀差)
        df = df.with_columns(
            (pl.col("ret_etf") - pl.col("ret_index")).alias("lead_gap"),
        )

        # 6. sigmoid 线性组合
        df = self._sigmoid_combine(df)

        return df.select(
            "datetime", "trade_date", "symbol",
            "stress_context", "vol_shock", "absorption",
            "resonance_count", "lead_gap", "nt_buy_prob",
        )

    # ── 内部计算方法 ──────────────────────────────────────────────────────

    def _add_stress_context(
        self, df: pl.DataFrame, index_1m: pl.DataFrame,
    ) -> pl.DataFrame:
        """
        市场压力背景 (0~1):
        - 指数收盘价 / 20日均线 偏离度
        - 近 N 日累计回撤
        两项归一化后取均值。
        """
        idx = index_1m.sort("datetime").unique(subset=["datetime"], maintain_order=True)
        daily = (
            idx.group_by("trade_date")
            .agg(pl.col("close").last().alias("daily_close"))
            .sort("trade_date")
        )

        # 20d SMA 偏离
        daily = daily.with_columns(
            pl.col("daily_close")
            .rolling_mean(window_size=self.stress_ma_window, min_periods=5)
            .alias("_sma"),
        )
        daily = daily.with_columns(
            (pl.col("daily_close") / pl.col("_sma") - 1.0).alias("_ma_dev"),
        )

        # N 日累计回撤
        daily = daily.with_columns(
            (pl.col("daily_close")
             / pl.col("daily_close").shift(self.stress_drawdown_days) - 1.0)
            .alias("_dd_ret"),
        )

        # 综合压力分 (0~1): 负偏离/回撤越大 → 压力越大
        ma_score = (
            -pl.col("_ma_dev") / (-self.stress_ma_dev_thresh)
        ).clip(0.0, 1.0)
        dd_score = (
            -pl.col("_dd_ret") / (-self.stress_drawdown_thresh)
        ).clip(0.0, 1.0)

        daily = daily.with_columns(
            ((ma_score + dd_score) / 2.0).fill_null(0.0).alias("stress_context"),
        )
        stress_daily = daily.select("trade_date", "stress_context")

        df = df.join(stress_daily, on="trade_date", how="left")
        return df.with_columns(
            pl.col("stress_context").forward_fill().fill_null(0.0),
        )

    def _add_vol_shock(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        vol_shock: 时间切片对齐的 robust_zscore(log(amount+1))。

        log 变换使分布更对称；时间切片对齐消除日内模式(开盘/收盘放量)。
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

    def _add_resonance(
        self,
        df: pl.DataFrame,
        fleet_etf_data: dict[str, pl.DataFrame] | None,
        idx_with_ret: pl.DataFrame,
    ) -> pl.DataFrame:
        """
        跨 ETF 共振: 在同一分钟，有多少只 ETF 满足 vol_shock > q99 AND alpha > 0。
        """
        if not fleet_etf_data:
            return df.with_columns(pl.lit(0).cast(pl.Int32).alias("resonance_count"))

        idx_sub = idx_with_ret.select(
            "datetime",
            pl.col("ret_index").alias("_fleet_idx_ret"),
        )
        eps = 1e-10
        trigger_cols: list[str] = []

        for sym, edf in fleet_etf_data.items():
            safe = sym.replace(".", "_")
            tr_name = f"_tr_{safe}"

            tmp = edf.sort("datetime").unique(subset=["datetime"], maintain_order=True)

            # log(amount+1)
            tmp = tmp.with_columns(
                (pl.col("amount") + 1.0).log().alias("_la"),
                pl.col("datetime").dt.time().alias("_tb"),
            )
            # 季节性 z-score
            tmp = tmp.with_columns(
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
            tmp = tmp.with_columns(
                (
                    (pl.col("_la") - pl.col("_vm"))
                    / pl.when(pl.col("_vs") < eps)
                    .then(None)
                    .otherwise(pl.col("_vs"))
                )
                .fill_nan(0.0)
                .fill_null(0.0)
                .alias("_vs_z"),
            )

            # 分钟收益率 + alpha
            tmp = tmp.with_columns(
                (pl.col("close") / pl.col("close").shift(1) - 1.0).alias("_ret_e"),
            )
            tmp = tmp.join(idx_sub, on="datetime", how="left")
            tmp = tmp.with_columns(
                pl.col("_fleet_idx_ret").fill_null(0.0),
                (pl.col("_ret_e") - pl.col("_fleet_idx_ret").fill_null(0.0))
                .alias("_alpha_e"),
            )

            # 触发条件: vol_shock > q99 AND alpha > 0
            tmp = tmp.with_columns(
                (
                    (pl.col("_vs_z") > self.vol_shock_q99)
                    & (pl.col("_alpha_e") > 0)
                )
                .cast(pl.Int32)
                .alias(tr_name),
            )

            df = df.join(tmp.select("datetime", tr_name), on="datetime", how="left")
            df = df.with_columns(pl.col(tr_name).fill_null(0))
            trigger_cols.append(tr_name)

        df = df.with_columns(
            pl.sum_horizontal(*[pl.col(c) for c in trigger_cols])
            .cast(pl.Int32)
            .alias("resonance_count"),
        )
        return df.drop(trigger_cols)

    def _sigmoid_combine(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        线性组合 + sigmoid 输出。

        特征归一化到 [0, 1] 后加权求和:
          stress_context: 已经 0~1
          vol_shock: clip(0, 10) / 10
          absorption: clip(0, 0.1) / 0.1  (分钟级 vol_shock×alpha 量级 ~0.001-0.05)
          resonance: / 4
          lead_gap: clip(-0.01, 0.01) * 100 → -1 ~ 1
        """
        logit = (
            self.w_stress * pl.col("stress_context")
            + self.w_vol_shock * (pl.col("vol_shock").clip(0.0, 10.0) / 10.0)
            + self.w_absorption * (pl.col("absorption").clip(0.0, 0.1) / 0.1)
            + self.w_resonance
            * (pl.col("resonance_count").cast(pl.Float64) / 4.0)
            + self.w_lead_gap * (pl.col("lead_gap").clip(-0.01, 0.01) * 100.0)
            + self.bias
        )
        return df.with_columns(
            (1.0 / (1.0 + (-logit).exp())).clip(0.0, 1.0).alias("nt_buy_prob"),
        )

    @staticmethod
    def _empty_result() -> pl.DataFrame:
        return pl.DataFrame(schema={
            "datetime": pl.Datetime("us"),
            "trade_date": pl.Date,
            "symbol": pl.Utf8,
            "stress_context": pl.Float64,
            "vol_shock": pl.Float64,
            "absorption": pl.Float64,
            "resonance_count": pl.Int32,
            "lead_gap": pl.Float64,
            "nt_buy_prob": pl.Float64,
        })
