#!/usr/bin/env python3
"""Backtest: local-data-priority small-cap strategy using DuckDB Lake + AKQuant.

Rules
- weekly rebalance
- rank PE/PB within each industry and keep the cheapest bucket
- then choose the smallest market caps
- time the market with CSI 1000 MA20

Local-first policy
- stock cross-section and bars come from local DuckDB views
- only the CSI 1000 index series is fetched online and cached locally
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import akshare as ak
import akshare_proxy_patch
import duckdb
import pandas as pd
from akquant import ExecutionMode, Strategy, run_backtest


PROXY_HOST = "***AKSHARE_HOST***"
PROXY_TOKEN = "***AKSHARE_TOKEN***"
INDEX_CODE = "000852"


class WeeklyPlanStrategy(Strategy):
    """Apply a precomputed daily plan with equal-weight rebalancing."""

    warmup_period = 1

    def __init__(self, daily_plan: dict[pd.Timestamp, list[str]]):
        self.warmup_period = 1
        self.daily_plan = {
            pd.Timestamp(k).normalize(): list(v) for k, v in daily_plan.items()
        }
        self.current_date: pd.Timestamp | None = None
        self.current_targets: list[str] = []

    def _switch_day(self, bar_date: pd.Timestamp) -> None:
        if self.current_date == bar_date:
            return
        self.current_date = bar_date
        self.current_targets = list(self.daily_plan.get(bar_date, []))

    def on_bar(self, bar) -> None:
        bar_date = pd.Timestamp(bar.timestamp).tz_localize(None).normalize()
        self._switch_day(bar_date)

        symbol = str(bar.symbol)
        pos = self.get_position(symbol)

        if not self.current_targets:
            if pos > 0:
                self.order_target_percent(0.0, symbol)
            return

        if symbol in self.current_targets:
            self.order_target_percent(1.0 / len(self.current_targets), symbol)
        elif pos > 0:
            self.order_target_percent(0.0, symbol)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backtest local small-cap strategy on Lake")
    p.add_argument("--db", type=Path, default=Path("data/duckdb/stock.duckdb"))
    p.add_argument("--start-date", type=str, default="2010-01-01")
    p.add_argument("--end-date", type=str, default=None)
    p.add_argument("--stock-num", type=int, default=10)
    p.add_argument("--value-quantile", type=float, default=0.4)
    p.add_argument("--min-list-days", type=int, default=250)
    p.add_argument("--timing-ma", type=int, default=20)
    p.add_argument("--initial-cash", type=float, default=200_000.0)
    p.add_argument("--commission-rate", type=float, default=0.0003)
    p.add_argument("--stamp-tax-rate", type=float, default=0.0005)
    p.add_argument("--min-commission", type=float, default=5.0)
    p.add_argument("--slippage", type=float, default=0.002)
    p.add_argument(
        "--index-cache",
        type=Path,
        default=Path("data/cache/index_000852.csv"),
    )
    p.add_argument(
        "--out-metrics",
        type=Path,
        default=Path("data/backtest_small_cap_lake_metrics.csv"),
    )
    p.add_argument(
        "--out-trades",
        type=Path,
        default=Path("data/backtest_small_cap_lake_trades.csv"),
    )
    p.add_argument(
        "--out-weekly-picks",
        type=Path,
        default=Path("data/backtest_small_cap_lake_weekly_picks.csv"),
    )
    p.add_argument(
        "--out-equity",
        type=Path,
        default=Path("data/backtest_small_cap_lake_equity.csv"),
    )
    return p.parse_args()


def to_ts(value: object) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if pd.isna(ts):
        raise ValueError(f"invalid timestamp: {value}")
    if ts.tzinfo is not None:
        ts = ts.tz_convert("Asia/Shanghai").tz_localize(None)
    return pd.Timestamp(ts).normalize()


def first_existing(columns: Iterable[object], candidates: list[str]) -> str | None:
    cols = [str(c) for c in columns]
    for name in candidates:
        if name in cols:
            return name
    return None


def normalize_date_close(df: pd.DataFrame) -> pd.Series:
    date_col = first_existing(
        list(df.columns), ["date", "Date", "日期", "时间", "月份", "month"]
    )
    close_col = first_existing(list(df.columns), ["close", "Close", "收盘", "收盘价"])

    if date_col is None or close_col is None:
        lower_map = {str(c).lower(): str(c) for c in df.columns}
        for key, value in lower_map.items():
            if date_col is None and (
                "date" in key or "日期" in key or "时间" in key or "月" in key
            ):
                date_col = value
            if close_col is None and ("close" in key or "收盘" in key):
                close_col = value

    if date_col is None or close_col is None:
        raise RuntimeError(f"date/close columns not found: {df.columns.tolist()}")

    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df[date_col], errors="coerce"),
            "close": pd.to_numeric(df[close_col], errors="coerce"),
        }
    )
    out = out.dropna(subset=["date", "close"]).sort_values("date")
    out = out[~out["date"].duplicated(keep="last")]
    if out.empty:
        raise RuntimeError("normalized date-close is empty")

    index = pd.DatetimeIndex(out["date"]).tz_localize(None)
    return pd.Series(out["close"].to_numpy(dtype=float), index=index, name="close")


def load_cached_series(path: Path) -> pd.Series | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    try:
        return normalize_date_close(df)
    except Exception:
        return None


def save_cached_series(path: Path, series: pd.Series) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"date": pd.DatetimeIndex(series.index), "close": series.values}).to_csv(
        path, index=False, encoding="utf-8-sig"
    )


def ensure_proxy() -> None:
    akshare_proxy_patch.install_patch(PROXY_HOST, PROXY_TOKEN, retry=30)


def fetch_index_price(code: str, cache_path: Path) -> pd.Series:
    cached = load_cached_series(cache_path)
    if cached is not None:
        return cached

    ensure_proxy()
    df = ak.index_zh_a_hist(
        symbol=code,
        period="daily",
        start_date="19900101",
        end_date="21000101",
    )
    if df is None or df.empty:
        raise RuntimeError(f"index_zh_a_hist empty for {code}")

    series = normalize_date_close(df)
    save_cached_series(cache_path, series)
    return series


def resolve_end_date(
    con: duckdb.DuckDBPyConnection, end_date: str | None
) -> pd.Timestamp:
    if end_date:
        return to_ts(end_date)
    row = con.execute("SELECT MAX(date) FROM v_bar_daily_hfq").fetchone()
    if not row or row[0] is None:
        raise RuntimeError("v_bar_daily_hfq has no data")
    return to_ts(row[0])


def get_trading_dates(
    con: duckdb.DuckDBPyConnection, start_date: pd.Timestamp, end_date: pd.Timestamp
) -> pd.DatetimeIndex:
    df = con.execute(
        """
        SELECT DISTINCT date
        FROM v_bar_daily_hfq
        WHERE date BETWEEN ? AND ?
        ORDER BY date
        """,
        [start_date.date(), end_date.date()],
    ).df()
    if df.empty:
        raise RuntimeError("No trading dates found")
    dates = pd.to_datetime(df["date"], errors="coerce").dropna()
    return pd.DatetimeIndex(dates).tz_localize(None).normalize()


def build_weekly_schedule(trading_dates: Iterable[pd.Timestamp]) -> pd.DataFrame:
    dates = pd.DatetimeIndex(pd.to_datetime(list(trading_dates), errors="coerce"))
    dates = dates.dropna().tz_localize(None).normalize().sort_values().unique()
    if len(dates) < 2:
        return pd.DataFrame(columns=["signal_date", "rebalance_date"])

    df = pd.DataFrame({"date": dates})
    df["week"] = df["date"].dt.to_period("W-FRI")
    last_in_week = df.groupby("week", as_index=False).tail(1).copy()
    next_dates = pd.Series(df["date"].to_list(), index=df["date"]).shift(-1)
    last_in_week["rebalance_date"] = last_in_week["date"].map(next_dates)
    last_in_week = last_in_week.dropna(subset=["rebalance_date"])
    last_in_week = last_in_week.rename(columns={"date": "signal_date"})
    return last_in_week[["signal_date", "rebalance_date"]].reset_index(drop=True)


def select_small_cap_candidates(
    snapshot_df: pd.DataFrame,
    signal_date: pd.Timestamp,
    stock_num: int = 10,
    value_quantile: float = 0.4,
    min_list_days: int = 250,
) -> pd.DataFrame:
    if snapshot_df.empty:
        return snapshot_df.iloc[0:0].copy()

    work = snapshot_df.copy()
    work["signal_date"] = to_ts(signal_date)
    work["list_date"] = pd.to_datetime(work["list_date"], errors="coerce")
    work["pe_ttm"] = pd.to_numeric(work["pe_ttm"], errors="coerce")
    work["pb"] = pd.to_numeric(work["pb"], errors="coerce")
    work["total_mv_10k"] = pd.to_numeric(work["total_mv_10k"], errors="coerce")
    work = work.dropna(
        subset=["symbol", "industry", "list_date", "pe_ttm", "pb", "total_mv_10k"]
    )
    work = work[
        (work["pe_ttm"] > 0)
        & (work["pb"] > 0)
        & (work["industry"].astype(str).str.len() > 0)
    ]
    if work.empty:
        return work

    cutoff_date = to_ts(signal_date) - pd.Timedelta(days=max(0, int(min_list_days)))
    work = work[work["list_date"] <= cutoff_date]
    if work.empty:
        return work

    work["pe_rank"] = work.groupby("industry")["pe_ttm"].rank(
        pct=True, ascending=True, method="average"
    )
    work["pb_rank"] = work.groupby("industry")["pb"].rank(
        pct=True, ascending=True, method="average"
    )
    work = work[
        (work["pe_rank"] <= float(value_quantile))
        & (work["pb_rank"] <= float(value_quantile))
    ]
    if work.empty:
        return work

    work = work.sort_values(["total_mv_10k", "symbol"], ascending=[True, True])
    return work.head(max(1, int(stock_num))).reset_index(drop=True)


def load_weekly_selection_frames(
    con: duckdb.DuckDBPyConnection,
    signal_dates: list[pd.Timestamp],
) -> pd.DataFrame:
    if not signal_dates:
        return pd.DataFrame()

    date_df = pd.DataFrame({"signal_date": pd.to_datetime(signal_dates)})
    con.register("tmp_signal_dates", date_df)
    try:
        df = con.execute(
            """
            SELECT
              s.date AS signal_date,
              s.symbol,
              s.name,
              s.industry,
              s.pe_ttm,
              s.pb,
              s.total_mv_10k,
              s.list_date
            FROM v_daily_hfq_w_ind_dim s
            INNER JOIN tmp_signal_dates d
              ON s.date = d.signal_date
            WHERE s.pe_ttm IS NOT NULL
              AND s.pb IS NOT NULL
              AND s.total_mv_10k IS NOT NULL
              AND s.list_date IS NOT NULL
              AND s.industry IS NOT NULL
            ORDER BY s.date, s.symbol
            """
        ).df()
    finally:
        con.unregister("tmp_signal_dates")

    if df.empty:
        return df

    df["signal_date"] = pd.to_datetime(df["signal_date"], errors="coerce")
    df["list_date"] = pd.to_datetime(df["list_date"], errors="coerce")
    return df.dropna(subset=["signal_date", "list_date"]).reset_index(drop=True)


def build_weekly_picks(
    snapshot_df: pd.DataFrame,
    stock_num: int,
    value_quantile: float,
    min_list_days: int,
) -> pd.DataFrame:
    if snapshot_df.empty:
        return pd.DataFrame(
            columns=[
                "signal_date",
                "symbol",
                "name",
                "industry",
                "pe_ttm",
                "pb",
                "total_mv_10k",
            ]
        )

    picked: list[pd.DataFrame] = []
    for signal_date, group in snapshot_df.groupby("signal_date", sort=True):
        sel = select_small_cap_candidates(
            group,
            signal_date=to_ts(signal_date),
            stock_num=stock_num,
            value_quantile=value_quantile,
            min_list_days=min_list_days,
        )
        if not sel.empty:
            picked.append(sel)

    if not picked:
        return pd.DataFrame(
            columns=[
                "signal_date",
                "symbol",
                "name",
                "industry",
                "pe_ttm",
                "pb",
                "total_mv_10k",
            ]
        )
    return pd.concat(picked, ignore_index=True)


def build_timing_flags(
    index_series: pd.Series,
    signal_dates: Iterable[pd.Timestamp],
    ma_window: int = 20,
) -> pd.DataFrame:
    series = index_series.copy().sort_index()
    series.index = pd.DatetimeIndex(series.index).tz_localize(None).normalize()
    frame = pd.DataFrame({"date": series.index, "close": series.values})
    frame["ma"] = frame["close"].rolling(max(1, int(ma_window))).mean()

    signals = pd.DataFrame(
        {"date": pd.DatetimeIndex(pd.to_datetime(list(signal_dates))).normalize()}
    ).sort_values("date")
    merged = pd.merge_asof(
        signals,
        frame.sort_values("date"),
        on="date",
        direction="backward",
    )
    merged["safe_market"] = (
        merged["close"].notna() & merged["ma"].notna() & (merged["close"] >= merged["ma"])
    )
    return merged[["date", "close", "ma", "safe_market"]]


def build_rebalance_plan(
    schedule_df: pd.DataFrame,
    weekly_picks_df: pd.DataFrame,
    timing_df: pd.DataFrame,
) -> dict[pd.Timestamp, list[str]]:
    picks_map: dict[pd.Timestamp, list[str]] = {}
    if not weekly_picks_df.empty:
        work = weekly_picks_df.copy()
        work["signal_date"] = pd.to_datetime(work["signal_date"], errors="coerce")
        work = work.dropna(subset=["signal_date", "symbol"])
        for signal_date, group in work.groupby("signal_date", sort=True):
            picks_map[to_ts(signal_date)] = [str(x) for x in group["symbol"].tolist()]

    timing_map: dict[pd.Timestamp, bool] = {}
    if not timing_df.empty:
        work = timing_df.copy()
        work["date"] = pd.to_datetime(work["date"], errors="coerce")
        work = work.dropna(subset=["date"])
        for _, row in work.iterrows():
            timing_map[to_ts(row["date"])] = bool(row["safe_market"])

    plan: dict[pd.Timestamp, list[str]] = {}
    for _, row in schedule_df.iterrows():
        signal_date = to_ts(row["signal_date"])
        rebalance_date = to_ts(row["rebalance_date"])
        if timing_map.get(signal_date, False):
            plan[rebalance_date] = list(picks_map.get(signal_date, []))
        else:
            plan[rebalance_date] = []
    return plan


def build_daily_plan(
    trading_dates: Iterable[pd.Timestamp],
    rebalance_plan: dict[pd.Timestamp, list[str]],
) -> dict[pd.Timestamp, list[str]]:
    plan: dict[pd.Timestamp, list[str]] = {}
    current_targets: list[str] = []
    for raw_date in pd.DatetimeIndex(pd.to_datetime(list(trading_dates))).sort_values():
        date = to_ts(raw_date)
        if date in rebalance_plan:
            current_targets = list(rebalance_plan[date])
        plan[date] = list(current_targets)
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

    for column in ["open", "high", "low", "close", "volume"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    out: dict[str, pd.DataFrame] = {}
    for symbol, group in df.groupby("symbol", sort=True):
        work = group.copy()
        work["date"] = pd.to_datetime(work["date"], errors="coerce")
        work = work.dropna(subset=["date", "open", "high", "low", "close", "volume"])
        work = work.sort_values("date").drop_duplicates(subset=["date"], keep="last")
        if not work.empty:
            out[str(symbol)] = work[
                ["date", "open", "high", "low", "close", "volume", "symbol"]
            ].reset_index(drop=True)
    return out


def summarize_plan(plan: dict[pd.Timestamp, list[str]]) -> tuple[int, int, float, int, int, int]:
    if not plan:
        return (0, 0, 0.0, 0, 0, 0)

    counts = [len(v) for v in plan.values()]
    non_empty_days = sum(1 for c in counts if c > 0)
    avg_n = float(sum(counts) / len(counts)) if counts else 0.0
    union_symbols = len({s for syms in plan.values() for s in syms})
    return (len(plan), non_empty_days, avg_n, min(counts), max(counts), union_symbols)


def metrics_to_frame(metrics_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_df.shape[1] == 1:
        value_col = metrics_df.columns[0]
        return pd.DataFrame(
            {
                "metric": metrics_df.index.astype(str),
                "value": metrics_df[value_col].values,
            }
        )
    return metrics_df.reset_index(drop=False)


def extract_equity_frame(result: object) -> pd.DataFrame:
    equity_curve = getattr(result, "equity_curve", None)
    cash_curve = getattr(result, "cash_curve", None)
    if isinstance(equity_curve, pd.Series) and not equity_curve.empty:
        equity = equity_curve.copy()
        equity.index = pd.DatetimeIndex(equity.index).tz_localize(None)
        out = pd.DataFrame({"date": equity.index, "equity": equity.values})
        if isinstance(cash_curve, pd.Series) and not cash_curve.empty:
            cash = cash_curve.copy()
            cash.index = pd.DatetimeIndex(cash.index).tz_localize(None)
            cash = cash.reindex(equity.index)
            out["cash"] = cash.values
        return out

    for attr in [
        "equity_df",
        "equity_curve_df",
        "portfolio_df",
        "daily_returns_df",
    ]:
        value = getattr(result, attr, None)
        if isinstance(value, pd.DataFrame) and not value.empty:
            return value.copy()

    for attr in ["analyzer_outputs"]:
        value = getattr(result, attr, None)
        if isinstance(value, dict):
            for item in value.values():
                if isinstance(item, pd.DataFrame) and not item.empty:
                    lowered = {str(c).lower() for c in item.columns}
                    if "equity" in lowered or "date" in lowered:
                        return item.copy()

    metrics = getattr(result, "metrics_df", None)
    if isinstance(metrics, pd.DataFrame):
        return pd.DataFrame()
    return pd.DataFrame()


def main() -> None:
    args = parse_args()

    if not args.db.exists():
        raise FileNotFoundError(f"DuckDB not found: {args.db}")

    con = duckdb.connect(args.db.as_posix())
    try:
        start_date = to_ts(args.start_date)
        end_date = resolve_end_date(con, args.end_date)
        if end_date < start_date:
            raise ValueError(f"end_date {end_date.date()} < start_date {start_date.date()}")

        trading_dates = get_trading_dates(con, start_date, end_date)
        schedule_df = build_weekly_schedule(trading_dates)
        if schedule_df.empty:
            raise RuntimeError("Weekly schedule is empty")

        signal_dates = [to_ts(x) for x in schedule_df["signal_date"].tolist()]
        snapshot_df = load_weekly_selection_frames(con, signal_dates)
        weekly_picks_df = build_weekly_picks(
            snapshot_df,
            stock_num=max(1, int(args.stock_num)),
            value_quantile=float(args.value_quantile),
            min_list_days=max(0, int(args.min_list_days)),
        )

        index_series = fetch_index_price(INDEX_CODE, args.index_cache)
        timing_df = build_timing_flags(
            index_series,
            signal_dates=signal_dates,
            ma_window=max(1, int(args.timing_ma)),
        )
        rebalance_plan = build_rebalance_plan(schedule_df, weekly_picks_df, timing_df)
        daily_plan = build_daily_plan(trading_dates, rebalance_plan)

        symbols = sorted({s for syms in daily_plan.values() for s in syms})
        if not symbols:
            raise RuntimeError("No symbols selected under current filters")

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

    valid_symbols = set(market_data.keys())
    weekly_picks_df = weekly_picks_df[weekly_picks_df["symbol"].isin(valid_symbols)].copy()
    rebalance_plan = {
        date: [s for s in symbols_for_day if s in valid_symbols]
        for date, symbols_for_day in rebalance_plan.items()
    }
    daily_plan = build_daily_plan(trading_dates, rebalance_plan)

    schedule_with_timing = schedule_df.merge(
        timing_df.rename(columns={"date": "signal_date"}),
        on="signal_date",
        how="left",
    )
    weekly_out = schedule_with_timing.merge(
        weekly_picks_df,
        on="signal_date",
        how="left",
    )

    plan_days, non_empty_days, avg_n, min_n, max_n, union_n = summarize_plan(daily_plan)
    print(
        f"[INFO] backtest window: {start_date.date()} -> {end_date.date()} | "
        f"stock_num={max(1, int(args.stock_num))} value_quantile={float(args.value_quantile):.2f}"
    )
    print(
        f"[INFO] daily plan: days={plan_days}, non_empty_days={non_empty_days}, "
        f"avg/min/max per day={avg_n:.2f}/{min_n}/{max_n}, symbol_coverage={union_n}"
    )
    print(f"[INFO] loading symbols: {len(valid_symbols)}")

    result = run_backtest(
        data=market_data,
        strategy=WeeklyPlanStrategy(daily_plan=daily_plan),
        symbol=list(valid_symbols),
        initial_cash=float(args.initial_cash),
        commission_rate=float(args.commission_rate),
        stamp_tax_rate=float(args.stamp_tax_rate),
        min_commission=float(args.min_commission),
        slippage=float(args.slippage),
        execution_mode=ExecutionMode.NextOpen,
        timezone="Asia/Shanghai",
        t_plus_one=True,
    )

    print("\n[RESULT]")
    print(f"total_return_pct={result.metrics.total_return_pct:.2f}")
    print(f"sharpe_ratio={result.metrics.sharpe_ratio:.2f}")
    print(f"max_drawdown_pct={result.metrics.max_drawdown_pct:.2f}")

    args.out_metrics.parent.mkdir(parents=True, exist_ok=True)
    args.out_trades.parent.mkdir(parents=True, exist_ok=True)
    args.out_weekly_picks.parent.mkdir(parents=True, exist_ok=True)
    args.out_equity.parent.mkdir(parents=True, exist_ok=True)

    metrics_to_frame(result.metrics_df.copy()).to_csv(
        args.out_metrics, index=False, encoding="utf-8-sig"
    )
    result.trades_df.to_csv(args.out_trades, index=False, encoding="utf-8-sig")
    weekly_out.to_csv(args.out_weekly_picks, index=False, encoding="utf-8-sig")

    equity_df = extract_equity_frame(result)
    if equity_df.empty:
        pd.DataFrame(
            {
                "message": [
                    "AKQuant result did not expose an equity dataframe in a known attribute"
                ]
            }
        ).to_csv(args.out_equity, index=False, encoding="utf-8-sig")
    else:
        equity_df.to_csv(args.out_equity, index=False, encoding="utf-8-sig")

    print(f"[DONE] metrics     -> {args.out_metrics}")
    print(f"[DONE] trades      -> {args.out_trades}")
    print(f"[DONE] weekly picks -> {args.out_weekly_picks}")
    print(f"[DONE] equity      -> {args.out_equity}")


if __name__ == "__main__":
    main()
