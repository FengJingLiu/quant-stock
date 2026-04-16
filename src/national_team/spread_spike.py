"""
ETF-指数剪刀差因子 (Spread_Spike)

核心逻辑
--------
IOPV 的本质是 ETF 底层成分股的实时公允价值。用现货指数（如 000300）作为锚点。
当国家队暴力买入 510300 时，ETF 市价会瞬间拉升，但指数由 300 只股票成交价加权而成，
反应有几秒到十几秒滞后。此时 ETF 涨幅 > 指数涨幅，形成"剪刀差"，等效于 IOPV 正溢价。

算法
----
1. 提取 ETF（如 510300.SH）和对应现货指数（000300）的分钟线
2. 计算分钟收益率: Ret_ETF = close / prev_close - 1
3. Spread = Ret_ETF - Ret_Index
4. 对 Spread 做时间切片 z-score（20日同时刻均值/标准差）
5. Spread_Zscore > 3 + ETF 同时放量 → 高度等效 IOPV 正溢价

数据源: ClickHouse astock.klines_1m_etf + astock.klines_1m_index

使用
----
>>> from src.national_team.spread_spike import SpreadSpike
>>> factor = SpreadSpike()
>>> result = factor.compute("2026-03-15", "2026-04-14")
>>> result.filter(pl.col("spread_zscore") > 3)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import polars as pl

from src.national_team.ch_client import get_etf_1m, get_index_1m


# ETF → 对应现货指数
DEFAULT_ETF_INDEX_PAIRS = [
    ("510300.SH", "000300"),  # 沪深300
    ("510050.SH", "000016"),  # 上证50
    ("510500.SH", "000905"),  # 中证500
    ("512100.SH", "000852"),  # 中证1000
]

_EMPTY_SCHEMA = {
    "datetime": pl.Datetime("us"),
    "trade_date": pl.Date,
    "etf_symbol": pl.Utf8,
    "index_symbol": pl.Utf8,
    "ret_etf": pl.Float64,
    "ret_index": pl.Float64,
    "spread": pl.Float64,
    "spread_zscore": pl.Float64,
}


@dataclass
class SpreadSpike:
    """ETF-指数剪刀差因子。"""

    etf_index_pairs: list[tuple[str, str]] | None = None
    """(ETF symbol, Index symbol) 配对列表"""

    spread_lookback: int = 20
    """Spread z-score 回看天数"""

    spread_zscore_thresh: float = 3.0
    """触发阈值"""

    min_periods: int = 10
    """同时间切片最少历史天数"""

    ch_kwargs: dict | None = None
    """覆盖 ClickHouse 连接参数"""

    def __post_init__(self) -> None:
        if self.etf_index_pairs is None:
            self.etf_index_pairs = list(DEFAULT_ETF_INDEX_PAIRS)

    def compute(
        self,
        start_date: str | date,
        end_date: str | date,
    ) -> pl.DataFrame:
        """
        计算所有 ETF-指数对的 Spread_Spike。

        Returns
        -------
        Polars DataFrame with columns:
          datetime, trade_date, etf_symbol, index_symbol,
          ret_etf, ret_index, spread, spread_zscore
        纵向堆叠所有配对。
        """
        frames: list[pl.DataFrame] = []
        for etf_sym, idx_sym in self.etf_index_pairs:
            result = self._compute_pair(etf_sym, idx_sym, start_date, end_date)
            if result.height > 0:
                frames.append(result)

        if not frames:
            return pl.DataFrame(schema=_EMPTY_SCHEMA)

        return pl.concat(frames)

    def compute_single(
        self,
        etf_symbol: str,
        index_symbol: str,
        start_date: str | date,
        end_date: str | date,
    ) -> pl.DataFrame:
        """计算单个 ETF-指数对的 Spread_Spike。"""
        return self._compute_pair(etf_symbol, index_symbol, start_date, end_date)

    # ── 内部 ──────────────────────────────────────────────────────────────

    def _compute_pair(
        self,
        etf_sym: str,
        idx_sym: str,
        start_date: str | date,
        end_date: str | date,
    ) -> pl.DataFrame:
        """计算单个配对的剪刀差因子。"""
        etf_df = get_etf_1m(etf_sym, start_date, end_date, self.ch_kwargs)
        idx_df = get_index_1m(idx_sym, start_date, end_date, self.ch_kwargs)

        if etf_df.height == 0 or idx_df.height == 0:
            return pl.DataFrame(schema=_EMPTY_SCHEMA)

        etf_df = etf_df.sort("datetime").unique(subset=["datetime"], maintain_order=True)
        idx_df = idx_df.sort("datetime").unique(subset=["datetime"], maintain_order=True)

        # 分钟收益率
        etf_df = etf_df.with_columns(
            (pl.col("close") / pl.col("close").shift(1) - 1.0).alias("ret_etf")
        )
        idx_df = idx_df.with_columns(
            (pl.col("close") / pl.col("close").shift(1) - 1.0).alias("ret_index")
        )

        # 以 ETF datetime 为基准对齐指数收益率
        idx_sub = idx_df.select("datetime", "ret_index")
        merged = etf_df.join(idx_sub, on="datetime", how="left")
        merged = merged.with_columns(
            pl.col("ret_index").fill_null(0.0).fill_nan(0.0)
        )

        # Spread = Ret_ETF - Ret_Index
        merged = merged.with_columns(
            (pl.col("ret_etf") - pl.col("ret_index")).alias("spread")
        )

        # 时间切片对齐的 z-score
        merged = (
            merged.with_columns(pl.col("datetime").dt.time().alias("_tb"))
            .with_columns(
                pl.col("spread").shift(1)
                  .rolling_mean(window_size=self.spread_lookback,
                                min_periods=self.min_periods)
                  .over("_tb").alias("_sm"),
                pl.col("spread").shift(1)
                  .rolling_std(window_size=self.spread_lookback,
                               min_periods=self.min_periods)
                  .over("_tb").alias("_ss"),
            )
            .with_columns(
                ((pl.col("spread") - pl.col("_sm"))
                 / pl.when(pl.col("_ss") == 0).then(None).otherwise(pl.col("_ss")))
                .fill_nan(0.0).fill_null(0.0)
                .alias("spread_zscore")
            )
            .drop("_tb", "_sm", "_ss")
        )

        return merged.select(
            "datetime", "trade_date",
            pl.lit(etf_sym).alias("etf_symbol"),
            pl.lit(idx_sym).alias("index_symbol"),
            "ret_etf", "ret_index", "spread", "spread_zscore",
        )
