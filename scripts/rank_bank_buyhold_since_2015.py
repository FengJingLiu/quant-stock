#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast

import duckdb
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank buy-and-hold returns of all bank stocks since 2015"
    )
    parser.add_argument("--db", type=Path, default=Path("data/duckdb/stock.duckdb"))
    parser.add_argument("--start-date", type=str, default="2015-01-01")
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/bank_buyhold_since_2015_ranking.csv"),
    )
    parser.add_argument("--top", type=int, default=10)
    return parser.parse_args()


def as_normalized_ts(value: Any) -> pd.Timestamp:
    ts = cast(pd.Timestamp, pd.Timestamp(value))
    if pd.isna(ts):
        raise ValueError(f"Invalid timestamp: {value}")
    if ts.tzinfo is not None:
        ts = cast(pd.Timestamp, ts.tz_localize(None))
    return cast(pd.Timestamp, ts.normalize())


def resolve_window(
    con: duckdb.DuckDBPyConnection, start_date: str, end_date: str | None
) -> tuple[pd.Timestamp, pd.Timestamp]:
    start_ts = as_normalized_ts(start_date)
    row = con.execute(
        """
        SELECT MIN(date)
        FROM v_daily_hfq_w_ind_dim
        WHERE industry='银行' AND date >= ?
        """,
        [start_ts.date()],
    ).fetchone()
    if not row or row[0] is None:
        raise RuntimeError("No bank trading date found after start-date")
    start_trade = as_normalized_ts(row[0])

    if end_date is not None:
        end_ts = as_normalized_ts(end_date)
    else:
        row2 = con.execute(
            """
            SELECT MAX(date)
            FROM v_daily_hfq_w_ind_dim
            WHERE industry='银行'
            """
        ).fetchone()
        if not row2 or row2[0] is None:
            raise RuntimeError("No bank data found for end date")
        end_ts = as_normalized_ts(row2[0])

    if end_ts < start_trade:
        raise ValueError(f"end_date {end_ts.date()} < start_trade {start_trade.date()}")
    return start_trade, end_ts


def rank_buyhold_returns(
    con: duckdb.DuckDBPyConnection,
    start_trade: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> pd.DataFrame:
    df = con.execute(
        """
        WITH bank_universe AS (
          SELECT DISTINCT symbol, name
          FROM v_dim_symbol
          WHERE industry = '银行'
            AND COALESCE(is_delisted, FALSE) = FALSE
        ),
        bank_prices AS (
          SELECT b.symbol, b.date, b.close
          FROM v_bar_daily_hfq b
          INNER JOIN bank_universe u USING (symbol)
          WHERE b.date BETWEEN ? AND ?
        ),
        start_px AS (
          SELECT symbol, close AS start_close
          FROM bank_prices
          WHERE date = ?
        ),
        end_px AS (
          SELECT
            symbol,
            MAX(date) AS end_obs_date,
            MAX_BY(close, date) AS end_close
          FROM bank_prices
          WHERE date <= ?
          GROUP BY symbol
        )
        SELECT
          u.symbol,
          u.name,
          ?::DATE AS start_date,
          e.end_obs_date AS end_date,
          s.start_close,
          e.end_close,
          (e.end_close / s.start_close - 1.0) * 100.0 AS total_return_pct
        FROM bank_universe u
        INNER JOIN start_px s USING (symbol)
        INNER JOIN end_px e USING (symbol)
        WHERE s.start_close > 0 AND e.end_close > 0
        ORDER BY total_return_pct DESC, u.symbol ASC
        """,
        [
            start_trade.date(),
            end_ts.date(),
            start_trade.date(),
            end_ts.date(),
            start_trade.date(),
        ],
    ).df()

    if df.empty:
        raise RuntimeError("Ranking result is empty")

    df["start_date"] = cast(
        pd.Series, pd.to_datetime(df["start_date"], errors="coerce")
    )
    df["end_date"] = cast(pd.Series, pd.to_datetime(df["end_date"], errors="coerce"))
    df["start_close"] = cast(
        pd.Series, pd.to_numeric(df["start_close"], errors="coerce")
    )
    df["end_close"] = cast(pd.Series, pd.to_numeric(df["end_close"], errors="coerce"))
    df["total_return_pct"] = cast(
        pd.Series, pd.to_numeric(df["total_return_pct"], errors="coerce")
    )
    df = cast(
        pd.DataFrame,
        df.dropna(
            subset=[
                "start_date",
                "end_date",
                "start_close",
                "end_close",
                "total_return_pct",
            ]
        ),
    )

    days = (df["end_date"] - df["start_date"]).dt.days.clip(lower=1)
    years = days / 365.25
    df["cagr_pct"] = (
        (1.0 + df["total_return_pct"] / 100.0) ** (1.0 / years) - 1.0
    ) * 100.0

    return cast(pd.DataFrame, df.reset_index(drop=True))


def main() -> None:
    args = parse_args()
    if not args.db.exists():
        raise FileNotFoundError(f"DuckDB not found: {args.db}")

    con = duckdb.connect(args.db.as_posix(), read_only=True)
    try:
        start_trade, end_ts = resolve_window(con, args.start_date, args.end_date)
        ranking = rank_buyhold_returns(con, start_trade, end_ts)
    finally:
        con.close()

    ranking["rank"] = np.arange(1, len(ranking) + 1)
    ranking = ranking[
        [
            "rank",
            "symbol",
            "name",
            "start_date",
            "end_date",
            "start_close",
            "end_close",
            "total_return_pct",
            "cagr_pct",
        ]
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(args.out, index=False, encoding="utf-8-sig")

    top_n = max(1, int(args.top))
    top_df = ranking.head(top_n)

    best = ranking.iloc[0]
    print(
        f"[INFO] window: {as_normalized_ts(best['start_date']).date()} -> "
        f"{as_normalized_ts(best['end_date']).date()} | bank_count={len(ranking)}"
    )
    print(
        f"[BEST] {best['symbol']} {best['name']} | "
        f"total_return={float(best['total_return_pct']):.6f}% | "
        f"cagr={float(best['cagr_pct']):.6f}%"
    )
    print("\n[Top Ranking]")
    print(top_df.to_string(index=False))
    print(f"\n[DONE] ranking -> {args.out}")


if __name__ == "__main__":
    main()
