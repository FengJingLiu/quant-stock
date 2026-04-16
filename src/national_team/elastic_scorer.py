"""
事件后横截面选股器 — 国家队买入事件触发后，对候选池个股做弹性打分。

打分维度:
  1. impact_absorption — 冲击承接度: 指数权重 / ADV20 流动性比率
  2. pressure_vacuum  — 抛压真空度: 近5~20日下跌放量是否衰减
  3. event_response   — 事件响应强度: 事件日午后相对ETF/指数的超额收益
  4. tradability       — 可交易惩罚项: 涨停/ST/停牌/异常换手

输出: elastic_score (越高越好), 以及各子分数

使用:
    from src.national_team.elastic_scorer import ElasticScorer
    scorer = ElasticScorer()
    scored = scorer.score(ch, signal_date, candidate_symbols)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import clickhouse_connect
import polars as pl


@dataclass
class ElasticScorer:
    """事件后横截面弹性打分器。"""

    # ── 参数 ─────────────────────────────────────────────────────────────
    adv_window: int = 20
    """ADV (日均成交额) 回看天数"""

    pressure_short_window: int = 5
    """抛压真空度: 短周期"""

    pressure_long_window: int = 20
    """抛压真空度: 长周期"""

    afternoon_start: str = "13:00:00"
    """事件响应: 午后起始时间"""

    turnover_cap: float = 15.0
    """可交易惩罚: 换手率上限 (%)"""

    turnover_floor: float = 0.3
    """可交易惩罚: 换手率下限 (%)"""

    # ── 子分数权重 ────────────────────────────────────────────────────────
    w_absorption: float = 1.0
    w_vacuum: float = 1.0
    w_response: float = 2.0
    w_tradability: float = 1.0

    def score(
        self,
        ch: clickhouse_connect.driver.Client,
        signal_date: date,
        candidate_symbols: list[str],
        index_symbol: str = "000300",
    ) -> pl.DataFrame:
        """
        对候选池个股在事件日做弹性打分。

        Parameters
        ----------
        ch : ClickHouse 客户端
        signal_date : 国家队买入事件日
        candidate_symbols : 候选股票列表 (klines_1m_stock 格式, 如 'sh600519')
        index_symbol : 参考指数

        Returns
        -------
        DataFrame[symbol, impact_absorption, pressure_vacuum, event_response,
                  tradability, elastic_score]
        按 elastic_score 降序排列。
        """
        if not candidate_symbols:
            return self._empty()

        lookback_sd = signal_date - timedelta(days=self.adv_window * 2 + 10)

        # ── 获取个股日线聚合 ─────────────────────────────────────────────
        daily = self._query_daily_agg(ch, candidate_symbols, lookback_sd, signal_date)
        if daily.height == 0:
            return self._empty()

        # ── 获取指数日线 (用于事件响应) ──────────────────────────────────
        idx_daily_ret = self._query_index_daily_ret(ch, index_symbol, signal_date)

        # ── 获取事件日午后个股分钟数据 (事件响应) ────────────────────────
        afternoon_alpha = self._query_afternoon_alpha(
            ch, candidate_symbols, signal_date, index_symbol,
        )

        # ── 获取指数权重 ─────────────────────────────────────────────────
        weights = self._query_index_weights(ch, candidate_symbols, signal_date)

        # ── 1. 冲击承接度 ────────────────────────────────────────────────
        # ADV20 = 近20日日均成交额
        adv = (
            daily.filter(pl.col("trade_date") < signal_date)
            .group_by("symbol")
            .agg(
                pl.col("daily_amount").tail(self.adv_window).mean().alias("adv20"),
            )
        )
        # 合并权重 → impact_absorption = weight / (adv20 / 1e8)
        # 权重越高、流动性越充裕 → 更容易被国家队买到 → 弹性更好
        absorption = adv.join(weights, on="symbol", how="left").with_columns(
            pl.col("weight").fill_null(0.0),
            pl.col("adv20").fill_null(1.0),
        ).with_columns(
            (pl.col("weight") * 100 / (pl.col("adv20") / 1e8).clip(lower_bound=0.01))
            .alias("impact_absorption_raw"),
        )

        # ── 2. 抛压真空度 ────────────────────────────────────────────────
        # 计算近5日和近20日下跌日的平均成交额比值 (短期下跌放量是否衰减)
        pre_daily = daily.filter(pl.col("trade_date") < signal_date)
        vacuum = (
            pre_daily.with_columns(
                (pl.col("daily_close") < pl.col("daily_open")).alias("is_down"),
            )
            .group_by("symbol")
            .agg(
                # 短期下跌日均量
                pl.when(pl.col("is_down"))
                .then(pl.col("daily_amount"))
                .otherwise(None)
                .tail(self.pressure_short_window)
                .mean()
                .alias("down_vol_short"),
                # 长期下跌日均量
                pl.when(pl.col("is_down"))
                .then(pl.col("daily_amount"))
                .otherwise(None)
                .tail(self.pressure_long_window)
                .mean()
                .alias("down_vol_long"),
            )
            .with_columns(
                # 短期下跌放量衰减 → vacuum 越高越好 (抛压减弱)
                (1.0 - pl.col("down_vol_short").fill_null(0.0)
                 / pl.col("down_vol_long").fill_null(1.0).clip(lower_bound=1.0))
                .clip(0.0, 1.0)
                .alias("pressure_vacuum_raw"),
            )
        )

        # ── 3. 事件响应强度 ──────────────────────────────────────────────
        # 事件日午后个股收益 - 指数收益 = 超额
        response = afternoon_alpha.with_columns(
            (pl.col("stock_pm_ret") - idx_daily_ret).alias("event_response_raw"),
        )

        # ── 4. 可交易惩罚 ────────────────────────────────────────────────
        # 获取事件日数据
        event_day = daily.filter(pl.col("trade_date") == signal_date)
        tradability = event_day.select(
            "symbol",
            # 涨停检测: close == high 且涨幅 > 9%
            (
                (pl.col("daily_close") == pl.col("daily_high"))
                & ((pl.col("daily_close") / pl.col("daily_open") - 1.0) > 0.09)
            ).alias("is_limit_up"),
            pl.col("daily_amount").alias("event_amount"),
        )

        # 换手率近似: event_amount / adv20
        tradability = tradability.join(adv.select("symbol", "adv20"), on="symbol", how="left")
        tradability = tradability.with_columns(
            (pl.col("event_amount") / pl.col("adv20").clip(lower_bound=1.0))
            .alias("turnover_ratio"),
        ).with_columns(
            # 涨停不可买 → 0分; 换手异常 → 惩罚
            pl.when(pl.col("is_limit_up"))
            .then(0.0)
            .otherwise(
                1.0 - (
                    pl.when(pl.col("turnover_ratio") > self.turnover_cap / 100.0)
                    .then(0.5)
                    .when(pl.col("turnover_ratio") < self.turnover_floor / 100.0)
                    .then(0.3)
                    .otherwise(0.0)
                )
            )
            .alias("tradability_raw"),
        )

        # ── 合并 & 综合打分 ──────────────────────────────────────────────
        scored = (
            absorption.select("symbol", "impact_absorption_raw")
            .join(vacuum.select("symbol", "pressure_vacuum_raw"), on="symbol", how="outer_coalesce")
            .join(response.select("symbol", "event_response_raw"), on="symbol", how="outer_coalesce")
            .join(tradability.select("symbol", "tradability_raw"), on="symbol", how="outer_coalesce")
        )

        # 归一化各子分数到 [0, 1] (rank percentile)
        for col in ["impact_absorption_raw", "pressure_vacuum_raw", "event_response_raw"]:
            out_col = col.replace("_raw", "")
            scored = scored.with_columns(
                pl.col(col).fill_null(0.0).rank("ordinal").cast(pl.Float64)
                .truediv(pl.col(col).count().cast(pl.Float64))
                .alias(out_col),
            )
        scored = scored.with_columns(
            pl.col("tradability_raw").fill_null(0.5).alias("tradability"),
        )

        # 加权合成
        w_total = self.w_absorption + self.w_vacuum + self.w_response + self.w_tradability
        scored = scored.with_columns(
            (
                self.w_absorption * pl.col("impact_absorption")
                + self.w_vacuum * pl.col("pressure_vacuum")
                + self.w_response * pl.col("event_response")
                + self.w_tradability * pl.col("tradability")
            ).truediv(w_total).alias("elastic_score"),
        )

        return (
            scored.select(
                "symbol",
                "impact_absorption", "pressure_vacuum",
                "event_response", "tradability",
                "elastic_score",
            )
            .sort("elastic_score", descending=True)
        )

    # ── 数据查询 (内部) ──────────────────────────────────────────────────

    def _query_daily_agg(
        self,
        ch: clickhouse_connect.driver.Client,
        symbols: list[str],
        start_date: date,
        end_date: date,
    ) -> pl.DataFrame:
        """从 klines_1m_stock 聚合日线。"""
        r = ch.query_arrow(
            """
            SELECT symbol, trade_date,
                   argMin(open, datetime) as daily_open,
                   max(high) as daily_high,
                   min(low) as daily_low,
                   argMax(close, datetime) as daily_close,
                   sum(amount) as daily_amount
            FROM klines_1m_stock
            WHERE trade_date BETWEEN %(sd)s AND %(ed)s
              AND symbol IN %(syms)s
            GROUP BY symbol, trade_date
            ORDER BY symbol, trade_date
            """,
            parameters={"sd": start_date, "ed": end_date, "syms": symbols},
        )
        if r.num_rows == 0:
            return pl.DataFrame()
        return pl.from_arrow(r).with_columns(
            pl.col("trade_date").cast(pl.Date),
            pl.col("daily_open").cast(pl.Float64),
            pl.col("daily_high").cast(pl.Float64),
            pl.col("daily_low").cast(pl.Float64),
            pl.col("daily_close").cast(pl.Float64),
            pl.col("daily_amount").cast(pl.Float64),
        )

    def _query_index_daily_ret(
        self,
        ch: clickhouse_connect.driver.Client,
        index_symbol: str,
        signal_date: date,
    ) -> float:
        """获取事件日指数日收益率。"""
        r = ch.query_arrow(
            """
            SELECT trade_date,
                   argMin(open, datetime) as d_open,
                   argMax(close, datetime) as d_close
            FROM klines_1m_index
            WHERE symbol = %(sym)s
              AND trade_date BETWEEN %(sd)s AND %(ed)s
            GROUP BY trade_date
            ORDER BY trade_date
            """,
            parameters={
                "sym": index_symbol,
                "sd": signal_date - timedelta(days=5),
                "ed": signal_date,
            },
        )
        if r.num_rows == 0:
            return 0.0
        df = pl.from_arrow(r).with_columns(
            pl.col("trade_date").cast(pl.Date),
        ).sort("trade_date")
        if df.height < 2:
            # 只有事件日本身 — 用日内收益
            row = df.tail(1).row(0, named=True)
            o, c = float(row["d_open"]), float(row["d_close"])
            return (c / o - 1.0) if o > 0 else 0.0
        last = df.tail(2)
        prev_close = float(last.row(0, named=True)["d_close"])
        today_close = float(last.row(1, named=True)["d_close"])
        return (today_close / prev_close - 1.0) if prev_close > 0 else 0.0

    def _query_afternoon_alpha(
        self,
        ch: clickhouse_connect.driver.Client,
        symbols: list[str],
        signal_date: date,
        index_symbol: str,
    ) -> pl.DataFrame:
        """事件日午后个股收益。"""
        r = ch.query_arrow(
            """
            SELECT symbol,
                   argMin(close, datetime) as pm_open,
                   argMax(close, datetime) as pm_close
            FROM klines_1m_stock
            WHERE trade_date = %(d)s
              AND symbol IN %(syms)s
              AND toHour(datetime) >= 13
            GROUP BY symbol
            """,
            parameters={"d": signal_date, "syms": symbols},
        )
        if r.num_rows == 0:
            return pl.DataFrame({"symbol": [], "stock_pm_ret": []},
                                schema={"symbol": pl.Utf8, "stock_pm_ret": pl.Float64})
        df = pl.from_arrow(r)
        return df.with_columns(
            (pl.col("pm_close").cast(pl.Float64) / pl.col("pm_open").cast(pl.Float64) - 1.0)
            .fill_nan(0.0).fill_null(0.0).alias("stock_pm_ret"),
        ).select("symbol", "stock_pm_ret")

    def _query_index_weights(
        self,
        ch: clickhouse_connect.driver.Client,
        symbols: list[str],
        signal_date: date,
    ) -> pl.DataFrame:
        """获取最近一期指数权重，映射到 klines_1m_stock 格式。"""
        r = ch.query_arrow(
            """
            SELECT con_code, weight FROM dim_index_weights
            WHERE index_code = '399300.SZ'
              AND trade_date = (
                  SELECT max(trade_date) FROM dim_index_weights
                  WHERE index_code = '399300.SZ' AND trade_date <= %(d)s
              )
            """,
            parameters={"d": signal_date},
        )
        if r.num_rows == 0:
            return pl.DataFrame({"symbol": [], "weight": []},
                                schema={"symbol": pl.Utf8, "weight": pl.Float64})
        df = pl.from_arrow(r)
        # con_code (600519.SH) → klines symbol (sh600519)
        return df.with_columns(
            pl.col("con_code").map_elements(
                lambda s: f"{s.split('.')[1].lower()}{s.split('.')[0]}",
                return_dtype=pl.Utf8,
            ).alias("symbol"),
            pl.col("weight").cast(pl.Float64),
        ).select("symbol", "weight")

    def _empty(self) -> pl.DataFrame:
        return pl.DataFrame(
            schema={
                "symbol": pl.Utf8,
                "impact_absorption": pl.Float64,
                "pressure_vacuum": pl.Float64,
                "event_response": pl.Float64,
                "tradability": pl.Float64,
                "elastic_score": pl.Float64,
            }
        )
