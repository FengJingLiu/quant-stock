#!/usr/bin/env python3
"""Backtest highest-dividend bank stock (full position) vs CSI300.

Methodology
- Universe: A-share stocks with industry='银行' from local DuckDB view `v_daily_hfq_w_ind_dim`
- Rebalance: first trading day of each year from 2015 onwards
- Signal date: latest trading day before rebalance date (avoid look-ahead)
- Selection: pick one bank with max `dividend_yield_ttm` at signal date
- Positioning: full position in selected stock, hold until next annual rebalance
- Benchmark: CSI300 (沪深300), prefer local DuckDB; fallback to AkShare with proxy patch
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import akshare as ak
import akshare_proxy_patch
import duckdb
import numpy as np
import pandas as pd
from akquant import Strategy, run_backtest


@dataclass(frozen=True)
class YearPick:
    year: int
    rebalance_date: pd.Timestamp
    signal_date: pd.Timestamp
    symbol: str
    name: str
    dividend_yield_ttm: float


class YearlySingleBankStrategy(Strategy):
    warmup_period = 1

    def __init__(self, daily_plan: dict[pd.Timestamp, str]):
        self.warmup_period = 1
        self.daily_plan = {as_normalized_ts(k): str(v) for k, v in daily_plan.items()}
        self.current_date: pd.Timestamp | None = None
        self.current_target: str | None = None

    def _switch_day(self, bar_date: pd.Timestamp) -> None:
        if self.current_date == bar_date:
            return
        self.current_date = bar_date
        self.current_target = self.daily_plan.get(bar_date)

    def on_bar(self, bar):
        bar_date = as_normalized_ts(getattr(bar, "timestamp"))
        self._switch_day(bar_date)

        symbol = str(bar.symbol)
        pos = self.get_position(symbol)

        if self.current_target is None:
            if pos > 0:
                self.order_target_percent(0.0, symbol)
            return

        if symbol == self.current_target:
            self.order_target_percent(1.0, symbol)
        elif pos > 0:
            self.order_target_percent(0.0, symbol)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Backtest highest-dividend bank stock vs CSI300"
    )
    p.add_argument("--db", type=Path, default=Path("data/duckdb/stock.duckdb"))
    p.add_argument("--start-date", type=str, default="2015-01-01")
    p.add_argument(
        "--end-date", type=str, default=None, help="Default: max(date) in bank data"
    )
    p.add_argument("--initial-cash", type=float, default=1_000_000.0)
    p.add_argument("--commission-rate", type=float, default=0.0003)
    p.add_argument(
        "--out-picks",
        type=Path,
        default=Path("data/backtest_bank_high_dividend_yearly_picks.csv"),
    )
    p.add_argument(
        "--out-summary",
        type=Path,
        default=Path("data/backtest_bank_high_dividend_vs_hs300_summary.csv"),
    )
    return p.parse_args()


def as_normalized_ts(value: Any) -> pd.Timestamp:
    ts = cast(pd.Timestamp, pd.Timestamp(value))
    if pd.isna(ts):
        raise ValueError(f"Invalid timestamp: {value}")
    if ts.tzinfo is not None:
        ts = cast(pd.Timestamp, ts.tz_localize(None))
    return cast(pd.Timestamp, ts.normalize())


def resolve_end_date(
    con: duckdb.DuckDBPyConnection, end_date: str | None
) -> pd.Timestamp:
    if end_date:
        return as_normalized_ts(end_date)

    row = con.execute(
        """
        SELECT MAX(date)
        FROM v_daily_hfq_w_ind_dim
        WHERE industry='银行'
        """
    ).fetchone()
    if not row or row[0] is None:
        raise RuntimeError("No bank data found in v_daily_hfq_w_ind_dim")
    return as_normalized_ts(row[0])


def load_trading_dates(
    con: duckdb.DuckDBPyConnection,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> list[pd.Timestamp]:
    df = con.execute(
        """
        SELECT DISTINCT date
        FROM v_daily_hfq_w_ind_dim
        WHERE industry='银行' AND date BETWEEN ? AND ?
        ORDER BY date
        """,
        [start_date.date(), end_date.date()],
    ).df()
    if df.empty:
        return []
    return [as_normalized_ts(d) for d in df["date"].tolist()]


def load_trading_dates_until(
    con: duckdb.DuckDBPyConnection,
    end_date: pd.Timestamp,
) -> list[pd.Timestamp]:
    df = con.execute(
        """
        SELECT DISTINCT date
        FROM v_daily_hfq_w_ind_dim
        WHERE industry='银行' AND date <= ?
        ORDER BY date
        """,
        [end_date.date()],
    ).df()
    if df.empty:
        return []
    return [as_normalized_ts(d) for d in df["date"].tolist()]


def select_top_dividend_bank(
    con: duckdb.DuckDBPyConnection,
    signal_date: pd.Timestamp,
) -> tuple[str, str, float] | None:
    df = con.execute(
        """
        SELECT
          symbol,
          name,
          dividend_yield_ttm
        FROM v_daily_hfq_w_ind_dim
        WHERE industry='银行'
          AND date = ?
          AND dividend_yield_ttm IS NOT NULL
          AND close > 0
          AND COALESCE(is_delisted, FALSE) = FALSE
        ORDER BY dividend_yield_ttm DESC, symbol ASC
        LIMIT 1
        """,
        [signal_date.date()],
    ).df()
    if df.empty:
        return None
    row = df.iloc[0]
    return (str(row["symbol"]), str(row["name"]), float(row["dividend_yield_ttm"]))


def build_yearly_picks(
    con: duckdb.DuckDBPyConnection,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> list[YearPick]:
    all_dates = load_trading_dates_until(con, end_date)
    if not all_dates:
        return []

    trading_dates = [d for d in all_dates if start_date <= d <= end_date]
    if not trading_dates:
        return []

    years = sorted({d.year for d in trading_dates if d >= start_date})
    picks: list[YearPick] = []

    for year in years:
        first_trade = next((d for d in trading_dates if d.year == year), None)
        if first_trade is None:
            continue

        signal_candidates = [d for d in all_dates if d < first_trade]
        if not signal_candidates:
            continue
        signal_date = signal_candidates[-1]

        top = select_top_dividend_bank(con, signal_date)
        if top is None:
            continue

        symbol, name, div = top
        picks.append(
            YearPick(
                year=year,
                rebalance_date=first_trade,
                signal_date=signal_date,
                symbol=symbol,
                name=name,
                dividend_yield_ttm=div,
            )
        )

    return picks


def build_daily_plan(
    trading_dates: list[pd.Timestamp], picks: list[YearPick]
) -> dict[pd.Timestamp, str]:
    if not trading_dates or not picks:
        return {}

    picks_sorted = sorted(picks, key=lambda x: x.rebalance_date)
    plan: dict[pd.Timestamp, str] = {}

    pick_idx = 0
    for d in trading_dates:
        while (
            pick_idx + 1 < len(picks_sorted)
            and d >= picks_sorted[pick_idx + 1].rebalance_date
        ):
            pick_idx += 1
        if d >= picks_sorted[pick_idx].rebalance_date:
            plan[d] = picks_sorted[pick_idx].symbol

    return plan


def load_price_data(
    con: duckdb.DuckDBPyConnection,
    symbols: list[str],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> dict[str, pd.DataFrame]:
    if not symbols:
        return {}

    sym_df = pd.DataFrame({"symbol": sorted(set(symbols))})
    con.register("tmp_symbols", sym_df)
    try:
        df = con.execute(
            """
            SELECT
              date,
              symbol,
              open,
              high,
              low,
              close,
              volume
            FROM v_bar_daily_hfq
            INNER JOIN tmp_symbols USING (symbol)
            WHERE date BETWEEN ? AND ?
            ORDER BY symbol, date
            """,
            [start_date.date(), end_date.date()],
        ).df()
    finally:
        con.unregister("tmp_symbols")

    if df.empty:
        return {}

    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = cast(pd.Series, pd.to_numeric(df[c], errors="coerce"))

    out: dict[str, pd.DataFrame] = {}
    for sym, g in df.groupby("symbol", sort=True):
        work = g.copy()
        work["date"] = cast(pd.Series, pd.to_datetime(work["date"], errors="coerce"))
        work = work.dropna(subset=["date", "open", "high", "low", "close", "volume"])
        work = work.sort_values(by="date").drop_duplicates(subset=["date"], keep="last")
        if not work.empty:
            out[str(sym)] = cast(
                pd.DataFrame,
                work[
                    ["date", "open", "high", "low", "close", "volume", "symbol"]
                ].reset_index(drop=True),
            )

    return out


def fetch_hs300_from_local(
    con: duckdb.DuckDBPyConnection,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.Series | None:
    candidates = ["000300.SH", "sh000300", "SH000300"]
    for sym in candidates:
        df = con.execute(
            """
            SELECT date, close
            FROM v_bar_daily_hfq
            WHERE symbol = ? AND date BETWEEN ? AND ?
            ORDER BY date
            """,
            [sym, start_date.date(), end_date.date()],
        ).df()
        if df.empty:
            continue
        df["date"] = cast(pd.Series, pd.to_datetime(df["date"], errors="coerce"))
        df["close"] = cast(pd.Series, pd.to_numeric(df["close"], errors="coerce"))
        df = df.dropna(subset=["date", "close"]).drop_duplicates(
            subset=["date"], keep="last"
        )
        if not df.empty:
            s = cast(pd.Series, df.set_index("date")["close"].sort_index())
            s.name = "hs300"
            return s
    return None


def fetch_hs300_from_akshare(
    start_date: pd.Timestamp, end_date: pd.Timestamp
) -> pd.Series:
    akshare_proxy_patch.install_patch(os.environ["AKSHARE_PROXY_HOST"], os.environ["AKSHARE_PROXY_TOKEN"], retry=30)
    df = cast(pd.DataFrame, ak.stock_zh_index_daily_em(symbol="sh000300"))
    if df is None or df.empty:
        raise RuntimeError("AkShare returned empty data for sh000300")

    df["date"] = cast(pd.Series, pd.to_datetime(df["date"], errors="coerce"))
    df["close"] = cast(pd.Series, pd.to_numeric(df["close"], errors="coerce"))
    df = cast(pd.DataFrame, df.dropna(subset=["date", "close"]))
    df = cast(pd.DataFrame, df[(df["date"] >= start_date) & (df["date"] <= end_date)])
    df = cast(
        pd.DataFrame,
        df.sort_values(by=["date"]).drop_duplicates(subset=["date"], keep="last"),
    )
    if df.empty:
        raise RuntimeError("AkShare hs300 data is empty in selected date range")

    s = cast(pd.Series, df.set_index("date")["close"])
    s.name = "hs300"
    return s


def calc_perf_from_nav(nav: pd.Series) -> tuple[float, float, float]:
    nav = cast(pd.Series, nav.dropna())
    if nav.empty:
        return (np.nan, np.nan, np.nan)

    total_return = float(nav.iloc[-1] / nav.iloc[0] - 1.0)
    start_ts = as_normalized_ts(nav.index[0])
    end_ts = as_normalized_ts(nav.index[-1])
    days = max(1, int((end_ts - start_ts).days))
    years = days / 365.25
    cagr = float((1.0 + total_return) ** (1.0 / years) - 1.0) if years > 0 else np.nan

    rolling_peak = cast(pd.Series, nav.cummax())
    drawdown = cast(pd.Series, nav / rolling_peak - 1.0)
    max_dd = float(drawdown.min()) if not drawdown.empty else np.nan
    return (total_return, cagr, max_dd)


def main() -> None:
    args = parse_args()

    if not args.db.exists():
        raise FileNotFoundError(f"DuckDB not found: {args.db}")

    con = duckdb.connect(args.db.as_posix(), read_only=True)
    try:
        start_date = as_normalized_ts(args.start_date)
        end_date = resolve_end_date(con, args.end_date)
        if end_date < start_date:
            raise ValueError(
                f"end_date {end_date.date()} < start_date {start_date.date()}"
            )

        trading_dates = load_trading_dates(con, start_date, end_date)
        yearly_picks = build_yearly_picks(con, start_date, end_date)

        if not trading_dates:
            raise RuntimeError("No bank trading dates found in selected window")
        if not yearly_picks:
            raise RuntimeError(
                "No yearly picks found; check dividend_yield_ttm coverage"
            )

        daily_plan = build_daily_plan(trading_dates, yearly_picks)
        if not daily_plan:
            raise RuntimeError("Daily plan is empty")

        strategy_symbols = sorted({x.symbol for x in yearly_picks})
        market_data = load_price_data(con, strategy_symbols, start_date, end_date)
        if not market_data:
            raise RuntimeError("No market data loaded for selected bank symbols")

        hs300 = fetch_hs300_from_local(con, start_date, end_date)
    finally:
        con.close()

    if hs300 is None:
        hs300 = fetch_hs300_from_akshare(start_date, end_date)

    result = run_backtest(
        data=market_data,
        strategy=YearlySingleBankStrategy(daily_plan=daily_plan),
        symbol=list(market_data.keys()),
        initial_cash=float(args.initial_cash),
        commission_rate=float(args.commission_rate),
    )

    strategy_total_return = float(result.metrics.total_return_pct) / 100.0
    days = max(1, int((end_date - start_date).days))
    years = days / 365.25
    strategy_cagr = (
        float((1.0 + strategy_total_return) ** (1.0 / years) - 1.0)
        if years > 0
        else np.nan
    )
    strategy_max_dd = -abs(float(result.metrics.max_drawdown_pct) / 100.0)

    hs300_nav = cast(pd.Series, hs300 / hs300.iloc[0])
    bench_total_return, bench_cagr, bench_max_dd = calc_perf_from_nav(hs300_nav)

    picks_df = pd.DataFrame(
        [
            {
                "year": p.year,
                "signal_date": p.signal_date.date().isoformat(),
                "rebalance_date": p.rebalance_date.date().isoformat(),
                "symbol": p.symbol,
                "name": p.name,
                "dividend_yield_ttm": p.dividend_yield_ttm,
            }
            for p in yearly_picks
        ]
    )

    summary_df = pd.DataFrame(
        [
            {
                "strategy": "最高股息银行股(年初满仓切换)",
                "start_date": start_date.date().isoformat(),
                "end_date": end_date.date().isoformat(),
                "total_return_pct": strategy_total_return * 100.0,
                "cagr_pct": strategy_cagr * 100.0,
                "max_drawdown_pct": strategy_max_dd * 100.0,
            },
            {
                "strategy": "沪深300",
                "start_date": as_normalized_ts(hs300.index[0]).date().isoformat(),
                "end_date": as_normalized_ts(hs300.index[-1]).date().isoformat(),
                "total_return_pct": bench_total_return * 100.0,
                "cagr_pct": bench_cagr * 100.0,
                "max_drawdown_pct": bench_max_dd * 100.0,
            },
        ]
    )

    args.out_picks.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    picks_df.to_csv(args.out_picks, index=False, encoding="utf-8-sig")
    summary_df.to_csv(args.out_summary, index=False, encoding="utf-8-sig")

    print(
        f"[INFO] window: {start_date.date()} -> {end_date.date()} | years={len(picks_df)} | "
        f"symbols={sorted(set(picks_df['symbol'].tolist()))}"
    )
    print("\n[Yearly Picks]")
    print(picks_df.to_string(index=False))

    print("\n[Performance Summary]")
    print(summary_df.to_string(index=False))

    print(f"\n[DONE] yearly picks -> {args.out_picks}")
    print(f"[DONE] summary      -> {args.out_summary}")


if __name__ == "__main__":
    main()
