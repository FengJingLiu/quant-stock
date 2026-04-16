"""
ETF 舰队共振因子 (Resonance_Count)

核心逻辑
--------
国家队救市是"舰队式作业"，极少只买单一 ETF，往往是沪深300、上证50、中证500 一起托。
放弃对单只 ETF 溢价的死磕，转而在横截面上寻找异常的"并发性"。

算法
----
1. 监控 ETF 核心池（510300 华泰300、510500 南方500、512100 南方1000、510050 华夏50）
2. 对每只 ETF 计算时间切片对齐的分钟成交量 Z-Score
3. 在同一分钟内，统计有几只 ETF 同时触发"天量买盘"（Z-Score > 3 且收阳线）
4. ≥3 只同时触发 → 100% 国家队级别信号

数据源: ClickHouse astock.klines_1m_etf

使用
----
>>> from src.national_team.etf_resonance import ETFResonance
>>> factor = ETFResonance()
>>> result = factor.compute("2026-03-15", "2026-04-14")
>>> result.filter(pl.col("resonance_count") >= 3)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import polars as pl

from src.national_team.ch_client import get_etf_1m


# 核心宽基 ETF 池
DEFAULT_ETF_POOL = [
    "510300.SH",  # 华泰柏瑞沪深300ETF
    "510500.SH",  # 南方中证500ETF
    "512100.SH",  # 南方中证1000ETF
    "510050.SH",  # 华夏上证50ETF
]

_EMPTY_SCHEMA = {
    "datetime": pl.Datetime("us"),
    "trade_date": pl.Date,
    "resonance_count": pl.Int32,
    "triggered_etfs": pl.Utf8,
    "avg_vol_zscore": pl.Float64,
}


@dataclass
class ETFResonance:
    """ETF 舰队共振因子。"""

    etf_pool: list[str] | None = None
    """ETF 核心池，默认沪深300/中证500/中证1000/上证50"""

    vol_lookback: int = 20
    """成交量 z-score 回看天数"""

    vol_zscore_thresh: float = 3.0
    """触发"天量"的 z-score 阈值"""

    min_vol_periods: int = 10
    """同时间切片最少历史天数"""

    require_bullish: bool = True
    """是否要求分钟 K 线收阳线 (close > open) 才计为有效"""

    ch_kwargs: dict | None = None
    """覆盖 ClickHouse 连接参数"""

    def __post_init__(self) -> None:
        if self.etf_pool is None:
            self.etf_pool = list(DEFAULT_ETF_POOL)

    def compute(
        self,
        start_date: str | date,
        end_date: str | date,
    ) -> pl.DataFrame:
        """
        计算 ETF 舰队共振因子。

        Returns
        -------
        Polars DataFrame with columns:
          datetime, trade_date, resonance_count, triggered_etfs, avg_vol_zscore
        """
        # 1. 加载并计算每只 ETF 的 vol_zscore（仅查一次）
        etf_frames: dict[str, pl.DataFrame] = {}
        for sym in self.etf_pool:
            raw = get_etf_1m(sym, start_date, end_date, self.ch_kwargs)
            if raw.height == 0:
                continue
            raw = raw.sort("datetime").unique(subset=["datetime"], maintain_order=True)
            etf_frames[sym] = self._add_vol_zscore(raw)

        if not etf_frames:
            return pl.DataFrame(schema=_EMPTY_SCHEMA)

        # 2. 以第一只 ETF 的时间轴为基准
        first_sym = next(iter(etf_frames))
        base = etf_frames[first_sym].select("datetime", "trade_date")

        zs_cols: list[str] = []
        trig_cols: list[str] = []
        sym_order: list[str] = []

        for sym, edf in etf_frames.items():
            safe = sym.replace(".", "_")
            sym_order.append(sym)
            zs_name = f"_zs_{safe}"
            bl_name = f"_bl_{safe}"
            tr_name = f"_tr_{safe}"

            sub = edf.select(
                "datetime",
                pl.col("vol_zscore").alias(zs_name),
                (pl.col("close") > pl.col("open")).alias(bl_name),
            )
            base = base.join(sub, on="datetime", how="left")
            base = base.with_columns(
                pl.col(zs_name).fill_null(0.0),
                pl.col(bl_name).fill_null(False),
            )

            if self.require_bullish:
                trig_expr = (pl.col(zs_name) > self.vol_zscore_thresh) & pl.col(bl_name)
            else:
                trig_expr = pl.col(zs_name) > self.vol_zscore_thresh

            base = base.with_columns(trig_expr.alias(tr_name))
            zs_cols.append(zs_name)
            trig_cols.append(tr_name)

        # 3. 聚合
        base = base.with_columns(
            pl.sum_horizontal(*[pl.col(c).cast(pl.Int32) for c in trig_cols])
              .alias("resonance_count"),
            pl.mean_horizontal(*[pl.col(c) for c in zs_cols])
              .alias("avg_vol_zscore"),
        )

        # 4. 构建触发的 ETF 列表字符串
        triggered_list_expr = pl.concat_list([
            pl.when(pl.col(tc)).then(pl.lit(sym)).otherwise(pl.lit(None).cast(pl.Utf8))
            for tc, sym in zip(trig_cols, sym_order)
        ]).list.drop_nulls().list.join(",")

        base = base.with_columns(triggered_list_expr.alias("triggered_etfs"))

        return base.select(
            "datetime", "trade_date",
            "resonance_count", "triggered_etfs", "avg_vol_zscore",
        )

    # ── 内部 ──────────────────────────────────────────────────────────────

    def _add_vol_zscore(self, df: pl.DataFrame) -> pl.DataFrame:
        """添加 vol_zscore 列：时间切片对齐的成交量 z-score。"""
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
