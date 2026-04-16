"""
持有期与退出状态机 — 事件组合的持仓管理和退出规则。

三套并行退出规则:
  1. FixedHolding   — 固定持有: T+3 / T+5 / T+10 到期全平
  2. SellSignalExit — 事件衰减退出: nt_sell_prob 触发减仓或清仓
  3. RelativeExit   — 个股相对强弱退出: 跌破事件VWAP / 相对ETF明显转弱

状态机:
  PENDING → ENTERED (T+1 买入) → HOLDING → EXITED (触发退出)

使用:
    from src.national_team.exit_engine import ExitEngine, ExitRule
    engine = ExitEngine(rules=[ExitRule.FIXED_5])
    trades = engine.simulate(ch, portfolio, signal_date)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum
from typing import NamedTuple

import clickhouse_connect
import polars as pl


class ExitRule(Enum):
    FIXED_3 = "fixed_3"
    FIXED_5 = "fixed_5"
    FIXED_10 = "fixed_10"
    SELL_SIGNAL = "sell_signal"
    RELATIVE_WEAKNESS = "relative_weakness"


class Trade(NamedTuple):
    """单笔交易记录。"""
    symbol: str
    weight: float
    entry_date: date
    entry_price: float
    exit_date: date
    exit_price: float
    exit_rule: str
    holding_days: int
    ret: float           # 个股收益
    contribution: float  # weight × ret


@dataclass
class ExitEngine:
    """事件组合退出引擎。"""

    rules: list[ExitRule] = field(default_factory=lambda: [ExitRule.FIXED_5])
    """激活的退出规则 (任一触发即退出)"""

    # 相对强弱退出参数
    relative_weakness_threshold: float = -0.03
    """个股日收益 - ETF日收益 < 此值，连续N天触发退出"""

    relative_weakness_days: int = 2
    """连续弱势天数"""

    # 卖出信号退出参数
    sell_prob_threshold: float = 0.5
    """nt_sell_prob > 此值触发退出"""

    def simulate(
        self,
        ch: clickhouse_connect.driver.Client,
        portfolio: pl.DataFrame,
        signal_date: date,
        sell_probs: pl.DataFrame | None = None,
        max_hold_days: int = 20,
    ) -> list[Trade]:
        """
        模拟事件组合的持有与退出。

        Parameters
        ----------
        ch : ClickHouse 客户端
        portfolio : PortfolioBuilder.build() 输出 [symbol, elastic_score, weight]
        signal_date : 买入事件日 (T+0)
        sell_probs : 可选，卖出因子数据 [trade_date, nt_sell_prob]
        max_hold_days : 最大持有交易日

        Returns
        -------
        每只股票对应一条 Trade 记录
        """
        if portfolio.height == 0:
            return []

        symbols = portfolio["symbol"].to_list()
        weights = dict(zip(
            portfolio["symbol"].to_list(),
            portfolio["weight"].to_list(),
        ))

        # 获取交易日历
        data_end = signal_date + timedelta(days=max_hold_days * 2)
        trading_dates = self._get_trading_dates(ch, signal_date, data_end)

        if len(trading_dates) < 2:
            return []

        # T+0 是信号日, T+1 是买入日
        t0_idx = None
        for i, d in enumerate(trading_dates):
            if d >= signal_date:
                t0_idx = i
                break
        if t0_idx is None or t0_idx + 1 >= len(trading_dates):
            return []

        entry_date = trading_dates[t0_idx + 1]  # T+1 买入

        # 获取个股日线 + 前复权因子
        daily = self._get_stock_daily(ch, symbols, signal_date - timedelta(days=5), data_end)
        if daily.height == 0:
            return []

        # 获取 ETF 日线 (用于相对强弱)
        etf_daily = self._get_etf_daily(ch, signal_date - timedelta(days=5), data_end)

        # 固定持有天数映射
        fixed_days = {
            ExitRule.FIXED_3: 3,
            ExitRule.FIXED_5: 5,
            ExitRule.FIXED_10: 10,
        }

        trades: list[Trade] = []

        for sym in symbols:
            sym_daily = daily.filter(pl.col("symbol") == sym).sort("trade_date")
            sym_dates = sym_daily["trade_date"].to_list()
            sym_close = sym_daily["adj_close"].to_list()

            if entry_date not in sym_dates:
                continue

            entry_idx = sym_dates.index(entry_date)
            entry_price = sym_close[entry_idx]
            if entry_price is None or entry_price <= 0:
                continue

            exit_date = None
            exit_price = None
            exit_rule = "max_hold"

            # 逐日检查退出条件 (从T+2开始，因为T+1买当天不能卖)
            weakness_streak = 0
            for day_offset in range(1, max_hold_days + 1):
                check_idx = entry_idx + day_offset
                if check_idx >= len(sym_dates):
                    break

                cur_date = sym_dates[check_idx]
                cur_price = sym_close[check_idx]
                if cur_price is None:
                    continue

                # 检查固定持有退出
                for rule in self.rules:
                    if rule in fixed_days and day_offset >= fixed_days[rule]:
                        exit_date = cur_date
                        exit_price = cur_price
                        exit_rule = rule.value
                        break

                if exit_date:
                    break

                # 检查卖出信号退出
                if ExitRule.SELL_SIGNAL in self.rules and sell_probs is not None:
                    day_sell = sell_probs.filter(pl.col("trade_date") == cur_date)
                    if day_sell.height > 0:
                        sp = day_sell["nt_sell_prob"].max()
                        if sp is not None and sp > self.sell_prob_threshold:
                            exit_date = cur_date
                            exit_price = cur_price
                            exit_rule = ExitRule.SELL_SIGNAL.value
                            break

                # 检查相对强弱退出
                if ExitRule.RELATIVE_WEAKNESS in self.rules:
                    prev_price = sym_close[check_idx - 1] if check_idx > 0 else cur_price
                    stock_ret = (cur_price / prev_price - 1.0) if prev_price > 0 else 0.0

                    etf_ret = self._get_etf_ret(etf_daily, cur_date)
                    relative = stock_ret - etf_ret

                    if relative < self.relative_weakness_threshold:
                        weakness_streak += 1
                    else:
                        weakness_streak = 0

                    if weakness_streak >= self.relative_weakness_days:
                        exit_date = cur_date
                        exit_price = cur_price
                        exit_rule = ExitRule.RELATIVE_WEAKNESS.value
                        break

            # 如果没触发退出，用最后可用日
            if exit_date is None:
                last_idx = min(entry_idx + max_hold_days, len(sym_dates) - 1)
                exit_date = sym_dates[last_idx]
                exit_price = sym_close[last_idx]

            if exit_price is None or exit_price <= 0:
                continue

            ret = exit_price / entry_price - 1.0
            holding = sum(1 for d in sym_dates if entry_date < d <= exit_date)

            trades.append(Trade(
                symbol=sym,
                weight=weights.get(sym, 0.0),
                entry_date=entry_date,
                entry_price=entry_price,
                exit_date=exit_date,
                exit_price=exit_price,
                exit_rule=exit_rule,
                holding_days=holding,
                ret=ret,
                contribution=weights.get(sym, 0.0) * ret,
            ))

        return trades

    # ── 数据查询 ─────────────────────────────────────────────────────────

    def _get_trading_dates(
        self, ch: clickhouse_connect.driver.Client, start: date, end: date,
    ) -> list[date]:
        r = ch.query_arrow(
            """
            SELECT DISTINCT trade_date FROM klines_1m_index
            WHERE symbol = '000300'
              AND trade_date BETWEEN %(sd)s AND %(ed)s
            ORDER BY trade_date
            """,
            parameters={"sd": start, "ed": end},
        )
        if r.num_rows == 0:
            return []
        return pl.from_arrow(r).with_columns(
            pl.col("trade_date").cast(pl.Date),
        )["trade_date"].to_list()

    def _get_stock_daily(
        self,
        ch: clickhouse_connect.driver.Client,
        symbols: list[str],
        start: date,
        end: date,
    ) -> pl.DataFrame:
        """获取个股日线 + 前复权收盘价。"""
        r = ch.query_arrow(
            """
            SELECT symbol, trade_date,
                   argMax(close, datetime) as daily_close
            FROM klines_1m_stock
            WHERE trade_date BETWEEN %(sd)s AND %(ed)s
              AND symbol IN %(syms)s
            GROUP BY symbol, trade_date
            ORDER BY symbol, trade_date
            """,
            parameters={"sd": start, "ed": end, "syms": symbols},
        )
        if r.num_rows == 0:
            return pl.DataFrame()

        daily = pl.from_arrow(r).with_columns(
            pl.col("trade_date").cast(pl.Date),
            pl.col("daily_close").cast(pl.Float64),
        )

        # 获取前复权因子
        ts_symbols = [f"{s[2:]}.{s[:2].upper()}" for s in symbols]
        adj_r = ch.query_arrow(
            """
            SELECT symbol as ts_sym, trade_date, factor
            FROM adj_factor
            WHERE adj_type = 'qfq' AND fund_type = 'stock'
              AND trade_date BETWEEN %(sd)s AND %(ed)s
              AND symbol IN %(syms)s
            """,
            parameters={"sd": start, "ed": end, "syms": ts_symbols},
        )

        if adj_r.num_rows > 0:
            adj_df = pl.from_arrow(adj_r).with_columns(
                pl.col("trade_date").cast(pl.Date),
                pl.col("factor").cast(pl.Float64),
                # ts_sym -> local symbol
                pl.col("ts_sym").map_elements(
                    lambda s: f"{s.split('.')[1].lower()}{s.split('.')[0]}",
                    return_dtype=pl.Utf8,
                ).alias("symbol"),
            ).select("symbol", "trade_date", "factor")

            daily = daily.join(adj_df, on=["symbol", "trade_date"], how="left")
            daily = daily.with_columns(
                (pl.col("daily_close") * pl.col("factor").fill_null(1.0))
                .alias("adj_close"),
            )
        else:
            daily = daily.with_columns(
                pl.col("daily_close").alias("adj_close"),
            )

        return daily

    def _get_etf_daily(
        self, ch: clickhouse_connect.driver.Client, start: date, end: date,
    ) -> pl.DataFrame:
        """获取 510300 ETF 日线。"""
        r = ch.query_arrow(
            """
            SELECT trade_date,
                   argMax(close, datetime) as etf_close
            FROM klines_1m_etf
            WHERE symbol = '510300.SH'
              AND trade_date BETWEEN %(sd)s AND %(ed)s
            GROUP BY trade_date
            ORDER BY trade_date
            """,
            parameters={"sd": start, "ed": end},
        )
        if r.num_rows == 0:
            return pl.DataFrame()
        return pl.from_arrow(r).with_columns(
            pl.col("trade_date").cast(pl.Date),
            pl.col("etf_close").cast(pl.Float64),
        )

    def _get_etf_ret(self, etf_daily: pl.DataFrame, cur_date: date) -> float:
        """获取 ETF 当日收益率。"""
        if etf_daily.height == 0:
            return 0.0
        df = etf_daily.sort("trade_date")
        dates = df["trade_date"].to_list()
        closes = df["etf_close"].to_list()
        if cur_date not in dates:
            return 0.0
        idx = dates.index(cur_date)
        if idx == 0:
            return 0.0
        prev = closes[idx - 1]
        cur = closes[idx]
        return (cur / prev - 1.0) if prev and prev > 0 else 0.0
