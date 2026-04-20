#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, cast

import akshare as ak
import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data_clients import ensure_akshare_proxy_patch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Buy-and-hold CMB (招商银行) vs HS300 from 2015"
    )
    parser.add_argument("--db", type=Path, default=Path("data/duckdb/stock.duckdb"))
    parser.add_argument("--symbol", type=str, default="600036.SH")
    parser.add_argument("--start-date", type=str, default="2015-01-01")
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument(
        "--out-summary",
        type=Path,
        default=Path("data/backtest_cmb_hold_vs_hs300_summary.csv"),
    )
    return parser.parse_args()


def as_normalized_ts(value: Any) -> pd.Timestamp:
    ts = cast(pd.Timestamp, pd.Timestamp(value))
    if pd.isna(ts):
        raise ValueError(f"Invalid timestamp: {value}")
    if ts.tzinfo is not None:
        ts = cast(pd.Timestamp, ts.tz_localize(None))
    return cast(pd.Timestamp, ts.normalize())


def resolve_end_date(
    con: duckdb.DuckDBPyConnection, symbol: str, end_date: str | None
) -> pd.Timestamp:
    if end_date:
        return as_normalized_ts(end_date)

    row = con.execute(
        """
        SELECT MAX(date)
        FROM v_bar_daily_hfq
        WHERE symbol = ?
        """,
        [symbol],
    ).fetchone()
    if not row or row[0] is None:
        raise RuntimeError(f"No hfq data found for symbol={symbol}")
    return as_normalized_ts(row[0])


def load_hfq_close_series(
    con: duckdb.DuckDBPyConnection,
    symbol: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.Series:
    df = con.execute(
        """
        SELECT date, close
        FROM v_bar_daily_hfq
        WHERE symbol = ? AND date BETWEEN ? AND ?
        ORDER BY date
        """,
        [symbol, start_date.date(), end_date.date()],
    ).df()

    if df.empty:
        raise RuntimeError(f"No hfq close data in window for symbol={symbol}")

    df["date"] = cast(pd.Series, pd.to_datetime(df["date"], errors="coerce"))
    df["close"] = cast(pd.Series, pd.to_numeric(df["close"], errors="coerce"))
    df = cast(pd.DataFrame, df.dropna(subset=["date", "close"]))
    df = cast(
        pd.DataFrame,
        df.sort_values(by=["date"]).drop_duplicates(subset=["date"], keep="last"),
    )
    if df.empty:
        raise RuntimeError(f"No valid close rows after cleaning for symbol={symbol}")

    series = cast(pd.Series, df.set_index("date")["close"])
    series.name = symbol
    return series


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
        df = cast(pd.DataFrame, df.dropna(subset=["date", "close"]))
        df = cast(
            pd.DataFrame,
            df.sort_values(by=["date"]).drop_duplicates(subset=["date"], keep="last"),
        )
        if not df.empty:
            series = cast(pd.Series, df.set_index("date")["close"])
            series.name = "hs300"
            return series
    return None


def fetch_hs300_from_akshare(
    start_date: pd.Timestamp, end_date: pd.Timestamp
) -> pd.Series:
    ensure_akshare_proxy_patch(
        proxy_host=os.environ["AKSHARE_PROXY_HOST"],
        token=os.environ["AKSHARE_PROXY_TOKEN"],
        retry=30,
    )
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

    series = cast(pd.Series, df.set_index("date")["close"])
    series.name = "hs300"
    return series


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
    max_drawdown = float(drawdown.min()) if not drawdown.empty else np.nan
    return (total_return, cagr, max_drawdown)


def main() -> None:
    args = parse_args()

    if not args.db.exists():
        raise FileNotFoundError(f"DuckDB not found: {args.db}")

    con = duckdb.connect(args.db.as_posix(), read_only=True)
    try:
        start_date = as_normalized_ts(args.start_date)
        end_date = resolve_end_date(con, args.symbol, args.end_date)
        if end_date < start_date:
            raise ValueError(
                f"end_date {end_date.date()} < start_date {start_date.date()}"
            )

        cmb_close = load_hfq_close_series(con, args.symbol, start_date, end_date)
        hs300_close = fetch_hs300_from_local(con, start_date, end_date)
    finally:
        con.close()

    if hs300_close is None:
        hs300_close = fetch_hs300_from_akshare(start_date, end_date)

    common_dates = cast(
        pd.DatetimeIndex, cmb_close.index.intersection(hs300_close.index)
    )
    if common_dates.empty:
        raise RuntimeError("No overlapping dates between CMB and HS300")

    cmb_common = cast(pd.Series, cmb_close.loc[common_dates])
    hs300_common = cast(pd.Series, hs300_close.loc[common_dates])

    cmb_nav = cast(pd.Series, cmb_common / cmb_common.iloc[0])
    hs300_nav = cast(pd.Series, hs300_common / hs300_common.iloc[0])

    cmb_ret, cmb_cagr, cmb_mdd = calc_perf_from_nav(cmb_nav)
    hs300_ret, hs300_cagr, hs300_mdd = calc_perf_from_nav(hs300_nav)

    summary_df = pd.DataFrame(
        [
            {
                "strategy": f"{args.symbol} 满仓持有(后复权)",
                "start_date": as_normalized_ts(common_dates.min()).date().isoformat(),
                "end_date": as_normalized_ts(common_dates.max()).date().isoformat(),
                "total_return_pct": cmb_ret * 100.0,
                "cagr_pct": cmb_cagr * 100.0,
                "max_drawdown_pct": cmb_mdd * 100.0,
            },
            {
                "strategy": "沪深300(价格指数)",
                "start_date": as_normalized_ts(common_dates.min()).date().isoformat(),
                "end_date": as_normalized_ts(common_dates.max()).date().isoformat(),
                "total_return_pct": hs300_ret * 100.0,
                "cagr_pct": hs300_cagr * 100.0,
                "max_drawdown_pct": hs300_mdd * 100.0,
            },
        ]
    )

    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(args.out_summary, index=False, encoding="utf-8-sig")

    print(
        f"[INFO] window(common): {as_normalized_ts(common_dates.min()).date()} -> "
        f"{as_normalized_ts(common_dates.max()).date()} | symbol={args.symbol}"
    )
    print("\n[Performance Summary]")
    print(summary_df.to_string(index=False))
    print(f"\n[DONE] summary -> {args.out_summary}")


if __name__ == "__main__":
    main()
