"""
ETF 出货共振因子 (ETF_Sell_Resonance)

核心逻辑
--------
国家队出货同样是"舰队式作业"，多只宽基 ETF 同时出现：
  - 成交量放大（相对同时间切片历史均值），但
  - 价格滞涨（收阴线 / 留长上影线 / 振幅极小）
即"天量滞涨"信号。当多只 ETF 在同一分钟并发此信号，强烈暗示有大资金在高位出货。

算法
----
1. 监控核心宽基 ETF 池（510300/510500/512100/510050）
2. 对每只 ETF：
   a. 计算时间切片对齐的分钟成交量 Z-Score（同 buy-side）
   b. 判断"滞涨"：收阴线(close < open) 或 上影线占比 > 阈值
   c. Vol_ZScore > 3 且 滞涨 → 该 ETF 触发出货信号
3. 统计同一分钟有几只 ETF 同时触发
4. ≥3 只同时触发 → 强出货共振

数据源: ClickHouse astock.klines_1m_etf

使用
----
>>> from src.national_team.etf_sell_resonance import ETFSellResonance
>>> factor = ETFSellResonance()
>>> result = factor.compute("2024-10-01", "2024-10-15")
>>> result.filter(pl.col("sell_resonance_count") >= 3)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import polars as pl

from src.national_team.ch_client import get_etf_1m


DEFAULT_ETF_POOL = [
    "510300.SH",
    "510500.SH",
    "512100.SH",
    "510050.SH",
]

_EMPTY_SCHEMA = {
    "datetime": pl.Datetime("us"),
    "trade_date": pl.Date,
    "sell_resonance_count": pl.Int32,
    "sell_triggered_etfs": pl.Utf8,
    "sell_avg_vol_zscore": pl.Float64,
}


@dataclass
class ETFSellResonance:
    """ETF 出货共振因子。"""

    etf_pool: list[str] | None = None

    vol_lookback: int = 20
    """成交量 z-score 回看天数"""

    vol_zscore_thresh: float = 3.0
    """触发"天量"的 z-score 阈值"""

    min_vol_periods: int = 10
    """同时间切片最少历史天数"""

    shadow_ratio_thresh: float = 0.4
    """上影线占比阈值，超过视为"被压"（滞涨的另一种形态）"""

    ch_kwargs: dict | None = None

    def __post_init__(self) -> None:
        if self.etf_pool is None:
            self.etf_pool = list(DEFAULT_ETF_POOL)

    def compute(
        self,
        start_date: str | date,
        end_date: str | date,
    ) -> pl.DataFrame:
        """
        计算 ETF 出货共振因子。

        Returns
        -------
        Polars DataFrame with columns:
          datetime, trade_date, sell_resonance_count,
          sell_triggered_etfs, sell_avg_vol_zscore
        """
        etf_frames: dict[str, pl.DataFrame] = {}
        for sym in self.etf_pool:
            raw = get_etf_1m(sym, start_date, end_date, self.ch_kwargs)
            if raw.height == 0:
                continue
            raw = raw.sort("datetime").unique(subset=["datetime"], maintain_order=True)
            etf_frames[sym] = self._add_vol_zscore(raw)

        if not etf_frames:
            return pl.DataFrame(schema=_EMPTY_SCHEMA)

        first_sym = next(iter(etf_frames))
        base = etf_frames[first_sym].select("datetime", "trade_date")

        zs_cols: list[str] = []
        trig_cols: list[str] = []
        sym_order: list[str] = []

        for sym, edf in etf_frames.items():
            safe = sym.replace(".", "_")
            sym_order.append(sym)
            zs_name = f"_zs_{safe}"
            bear_name = f"_bear_{safe}"
            shadow_name = f"_shd_{safe}"
            tr_name = f"_tr_{safe}"

            # 计算上影线占比
            body_top = pl.max_horizontal("close", "open")
            bar_range = pl.col("high") - pl.col("low")
            shadow_expr = (
                (pl.col("high") - body_top)
                / pl.when(bar_range == 0).then(None).otherwise(bar_range)
            ).fill_null(0.0).fill_nan(0.0)

            sub = edf.select(
                "datetime",
                pl.col("vol_zscore").alias(zs_name),
                (pl.col("close") < pl.col("open")).alias(bear_name),
                shadow_expr.alias(shadow_name),
            )
            base = base.join(sub, on="datetime", how="left")
            base = base.with_columns(
                pl.col(zs_name).fill_null(0.0),
                pl.col(bear_name).fill_null(False),
                pl.col(shadow_name).fill_null(0.0),
            )

            # 触发条件：天量 + (收阴线 或 长上影线)
            stagnant_expr = pl.col(bear_name) | (pl.col(shadow_name) > self.shadow_ratio_thresh)
            trig_expr = (pl.col(zs_name) > self.vol_zscore_thresh) & stagnant_expr

            base = base.with_columns(trig_expr.alias(tr_name))
            zs_cols.append(zs_name)
            trig_cols.append(tr_name)

        # 聚合
        base = base.with_columns(
            pl.sum_horizontal(*[pl.col(c).cast(pl.Int32) for c in trig_cols])
              .alias("sell_resonance_count"),
            pl.mean_horizontal(*[pl.col(c) for c in zs_cols])
              .alias("sell_avg_vol_zscore"),
        )

        triggered_list_expr = pl.concat_list([
            pl.when(pl.col(tc)).then(pl.lit(sym)).otherwise(pl.lit(None).cast(pl.Utf8))
            for tc, sym in zip(trig_cols, sym_order)
        ]).list.drop_nulls().list.join(",")

        base = base.with_columns(triggered_list_expr.alias("sell_triggered_etfs"))

        return base.select(
            "datetime", "trade_date",
            "sell_resonance_count", "sell_triggered_etfs", "sell_avg_vol_zscore",
        )

    def _add_vol_zscore(self, df: pl.DataFrame) -> pl.DataFrame:
        """时间切片对齐的成交量 z-score。"""
        return (
            df.with_columns(pl.col("datetime").dt.time().alias("_tb"))
            .with_columns(
                pl.col("volume").shift(1)
                  .rolling_mean(window_size=self.vol_lookback,
                                min_periods=self.min_vol_periods)
                  .over("_tb").alias("_vm"),
                pl.col("volume").shift(1)
                  .rolling_std(window_size=self.vol_lookback,
                               min_periods=self.min_vol_periods)
                  .over("_tb").alias("_vs"),
            )
            .with_columns(
                ((pl.col("volume") - pl.col("_vm"))
                 / pl.when(pl.col("_vs") == 0).then(None).otherwise(pl.col("_vs")))
                .fill_nan(0.0).fill_null(0.0)
                .alias("vol_zscore")
            )
            .drop("_tb", "_vm", "_vs")
        )
