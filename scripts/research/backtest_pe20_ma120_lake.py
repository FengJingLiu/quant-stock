#!/usr/bin/env python3
"""Backtest: PE<20 and close below MA120 using local Lake (DuckDB views).

Selection rules (daily)
- pe_ttm > 0 and pe_ttm < 20
- close < MA120, where MA120 is computed from v_bar_daily_hfq.close
- exclude ST/delisting-risk names: upper(name) not like '%ST%' and name not like '%退%'

Execution
- pick top N per day (default 1), sorted by pe_ttm asc, then symbol asc
- rebalance daily: full position / equal weight among daily targets

Outputs
- data/backtest_pe20_ma120_lake_metrics.csv
- data/backtest_pe20_ma120_lake_trades.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import pandas as pd
from akquant import Strategy, run_backtest


class DailyFullPositionFactorStrategy(Strategy):
    """Daily target rebalancing strategy (full position / equal weight)."""

    warmup_period = 1

    def __init__(self, daily_plan: dict[pd.Timestamp, list[str]]):
        self.warmup_period = 1
        self.daily_plan = {pd.Timestamp(k).normalize(): list(v) for k, v in daily_plan.items()}
        self.current_date: pd.Timestamp | None = None
        self.current_targets: set[str] = set()

    def _switch_day(self, bar_date: pd.Timestamp) -> None:
        if self.current_date == bar_date:
            return
        self.current_date = bar_date
        self.current_targets = set(self.daily_plan.get(bar_date, []))

    def on_bar(self, bar):
        bar_date = pd.Timestamp(bar.timestamp).tz_localize(None).normalize()
        self._switch_day(bar_date)

        symbol = bar.symbol
        pos = self.get_position(symbol)

        if not self.current_targets:
            if pos > 0:
                self.order_target_percent(0.0, symbol)
            return

        if symbol in self.current_targets:
            target_weight = 1.0 / len(self.current_targets)
            self.order_target_percent(target_weight, symbol)
        elif pos > 0:
            self.order_target_percent(0.0, symbol)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backtest PE<20 & close<MA120 on Lake")
    p.add_argument("--db", type=Path, default=Path("data/duckdb/stock.duckdb"))
    p.add_argument("--start-date", type=str, default="2021-03-01")
    p.add_argument("--end-date", type=str, default=None, help="Default: max(date) from v_bar_daily_hfq")
    p.add_argument("--top-per-day", type=int, default=1)
    p.add_argument("--initial-cash", type=float, default=200_000.0)
    p.add_argument("--commission-rate", type=float, default=0.0003)
    p.add_argument(
        "--out-metrics",
        type=Path,
        default=Path("data/backtest_pe20_ma120_lake_metrics.csv"),
    )
    p.add_argument(
        "--out-trades",
        type=Path,
        default=Path("data/backtest_pe20_ma120_lake_trades.csv"),
    )
    return p.parse_args()


def resolve_end_date(con: duckdb.DuckDBPyConnection, end_date: str | None) -> pd.Timestamp:
    if end_date:
        return pd.Timestamp(end_date).normalize()

    row = con.execute("SELECT MAX(date) FROM v_bar_daily_hfq").fetchone()
    if not row or row[0] is None:
        raise RuntimeError("v_bar_daily_hfq has no data")
    return pd.Timestamp(row[0]).normalize()


def build_daily_selection(
    con: duckdb.DuckDBPyConnection,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    top_per_day: int,
) -> pd.DataFrame:
    sql = """
    WITH base AS (
      SELECT
        date,
        symbol,
        close,
        pe_ttm,
        name
      FROM v_daily_hfq_w_ind_dim
      WHERE date BETWEEN ? AND ?
    ),
    ma AS (
      SELECT
        date,
        symbol,
        close,
        pe_ttm,
        name,
        AVG(close) OVER (
          PARTITION BY symbol
          ORDER BY date
          ROWS BETWEEN 119 PRECEDING AND CURRENT ROW
        ) AS ma120
      FROM base
    ),
    candidates AS (
      SELECT
        date,
        symbol,
        pe_ttm,
        close,
        ma120,
        name
      FROM ma
      WHERE
        ma120 IS NOT NULL
        AND pe_ttm > 0
        AND pe_ttm < 20
        AND close < ma120
        AND UPPER(COALESCE(name, '')) NOT LIKE '%ST%'
        AND COALESCE(name, '') NOT LIKE '%退%'
    ),
    ranked AS (
      SELECT
        date,
        symbol,
        pe_ttm,
        close,
        ma120,
        name,
        ROW_NUMBER() OVER (
          PARTITION BY date
          ORDER BY pe_ttm ASC, symbol ASC
        ) AS rn
      FROM candidates
    )
    SELECT date, symbol, pe_ttm, close, ma120, name
    FROM ranked
    WHERE rn <= ?
    ORDER BY date, pe_ttm, symbol
    """

    return con.execute(sql, [start_date.date(), end_date.date(), int(top_per_day)]).df()


def build_daily_plan(selection_df: pd.DataFrame) -> dict[pd.Timestamp, list[str]]:
    if selection_df.empty:
        return {}

    plan: dict[pd.Timestamp, list[str]] = {}
    for d, g in selection_df.groupby("date", sort=True):
        dt = pd.Timestamp(d).normalize()
        syms = [str(x) for x in g["symbol"].tolist() if pd.notna(x)]
        plan[dt] = syms
    return plan


def load_price_data_for_symbols(
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
              b.date,
              b.symbol,
              b.open,
              b.high,
              b.low,
              b.close,
              b.volume
            FROM v_bar_daily_hfq b
            INNER JOIN tmp_symbols s USING (symbol)
            WHERE b.date BETWEEN ? AND ?
            ORDER BY b.symbol, b.date
            """,
            [start_date.date(), end_date.date()],
        ).df()
    finally:
        con.unregister("tmp_symbols")

    if df.empty:
        return {}

    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    out: dict[str, pd.DataFrame] = {}
    for sym, g in df.groupby("symbol", sort=True):
        work = g.copy()
        work["date"] = pd.to_datetime(work["date"], errors="coerce")
        work = work.dropna(subset=["date", "open", "high", "low", "close", "volume"])
        work = work.sort_values("date").drop_duplicates(subset=["date"], keep="last")
        if not work.empty:
            out[str(sym)] = work[["date", "open", "high", "low", "close", "volume", "symbol"]].reset_index(
                drop=True
            )

    return out


def summarize_plan(plan: dict[pd.Timestamp, list[str]]) -> tuple[int, int, float, int, int, int]:
    if not plan:
        return (0, 0, 0.0, 0, 0)

    counts = [len(v) for v in plan.values()]
    non_empty_days = sum(1 for c in counts if c > 0)
    avg_n = float(sum(counts) / len(counts)) if counts else 0.0
    union_symbols = len({s for syms in plan.values() for s in syms})
    return (len(plan), non_empty_days, avg_n, min(counts), max(counts), union_symbols)


def main() -> None:
    args = parse_args()

    if not args.db.exists():
        raise FileNotFoundError(f"DuckDB not found: {args.db}")

    con = duckdb.connect(args.db.as_posix())
    try:
        start_date = pd.Timestamp(args.start_date).normalize()
        end_date = resolve_end_date(con, args.end_date)
        if end_date < start_date:
            raise ValueError(f"end_date {end_date.date()} < start_date {start_date.date()}")

        selection_df = build_daily_selection(
            con,
            start_date=start_date,
            end_date=end_date,
            top_per_day=max(1, int(args.top_per_day)),
        )
        daily_plan = build_daily_plan(selection_df)

        if not daily_plan:
            raise RuntimeError("No daily candidates found under current filters")

        symbols = sorted({s for syms in daily_plan.values() for s in syms})
        market_data = load_price_data_for_symbols(
            con,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
        )
    finally:
        con.close()

    if not market_data:
        raise RuntimeError("No market data loaded for selected symbols")

    # keep only symbols that actually have price data
    valid_symbols = set(market_data.keys())
    daily_plan = {d: [s for s in syms if s in valid_symbols] for d, syms in daily_plan.items()}

    plan_days, non_empty_days, avg_n, min_n, max_n, union_n = summarize_plan(daily_plan)

    print(
        f"[INFO] backtest window: {start_date.date()} -> {end_date.date()} | "
        f"top_per_day={max(1, int(args.top_per_day))}"
    )
    print(
        f"[INFO] daily plan: days={plan_days}, non_empty_days={non_empty_days}, "
        f"avg/min/max per day={avg_n:.2f}/{min_n}/{max_n}, symbol_coverage={union_n}"
    )
    print(f"[INFO] loading symbols: {len(valid_symbols)}")

    result = run_backtest(
        data=market_data,
        strategy=DailyFullPositionFactorStrategy(daily_plan=daily_plan),
        symbol=list(valid_symbols),
        initial_cash=float(args.initial_cash),
        commission_rate=float(args.commission_rate),
    )

    print("\n[RESULT]")
    print(f"total_return_pct={result.metrics.total_return_pct:.2f}")
    print(f"sharpe_ratio={result.metrics.sharpe_ratio:.2f}")
    print(f"max_drawdown_pct={result.metrics.max_drawdown_pct:.2f}")

    metrics_df = result.metrics_df.copy()
    if metrics_df.shape[1] == 1:
        value_col = metrics_df.columns[0]
        metrics_out = pd.DataFrame(
            {
                "metric": metrics_df.index.astype(str),
                "value": metrics_df[value_col].values,
            }
        )
    else:
        metrics_out = metrics_df.reset_index(drop=False)

    args.out_metrics.parent.mkdir(parents=True, exist_ok=True)
    args.out_trades.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.to_csv(args.out_metrics, index=False, encoding="utf-8-sig")
    result.trades_df.to_csv(args.out_trades, index=False, encoding="utf-8-sig")

    print(f"[DONE] metrics -> {args.out_metrics}")
    print(f"[DONE] trades  -> {args.out_trades}")


if __name__ == "__main__":
    main()
