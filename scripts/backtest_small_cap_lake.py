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
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def parse_percent(value: object) -> float | None:
    if value is None or value == "" or value == "-" or value is False:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        text = value.replace("%", "").replace(",", "").strip()
        if text in {"", "-", "False", "None"}:
            return None
        try:
            return float(text)
        except Exception:
            return None
    try:
        return float(value)  # type: ignore[arg-type]
    except Exception:
        return None


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
    p.add_argument(
        "--mode",
        type=str,
        default="full_combo",
        choices=["baseline", "buffer_only", "buffer_risk", "full_combo", "sharpe_booster", "all"],
    )
    p.add_argument("--hold-rank-cutoff", type=int, default=30)
    p.add_argument("--hold-value-quantile", type=float, default=0.6)
    p.add_argument("--prefetch-rank", type=int, default=80)
    p.add_argument("--financial-max-workers", type=int, default=6)
    p.add_argument("--refresh-financial-cache", action="store_true")
    p.add_argument("--refresh-risk-cache", action="store_true")
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
        "--financial-cache",
        type=Path,
        default=Path("data/cache/small_cap_financial_quality.parquet"),
    )
    p.add_argument(
        "--risk-cache",
        type=Path,
        default=Path("data/cache/small_cap_risk_flags.parquet"),
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


def estimate_announcement_date(report_period: object) -> pd.Timestamp:
    period = to_ts(report_period)
    if period.month == 3 and period.day == 31:
        lag_days = 45
    elif period.month == 6 and period.day == 30:
        lag_days = 60
    elif period.month == 9 and period.day == 30:
        lag_days = 45
    elif period.month == 12 and period.day == 31:
        lag_days = 120
    else:
        lag_days = 90
    return period + pd.Timedelta(days=lag_days)


def normalize_ak_symbol(symbol: str) -> str:
    return symbol.replace(".SH", "").replace(".SZ", "").replace(".BJ", "")


def normalize_financial_quality_rows(symbol: str, raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df is None or raw_df.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "report_period",
                "announcement_date",
                "roe",
                "source",
            ]
        )

    report_col = first_existing(list(raw_df.columns), ["报告期", "report_period"])
    roe_col = first_existing(list(raw_df.columns), ["净资产收益率", "roe"])
    if report_col is None or roe_col is None:
        return pd.DataFrame(
            columns=[
                "symbol",
                "report_period",
                "announcement_date",
                "roe",
                "source",
            ]
        )

    out = pd.DataFrame(
        {
            "symbol": symbol,
            "report_period": pd.to_datetime(raw_df[report_col], errors="coerce"),
            "roe": raw_df[roe_col].map(parse_percent),
            "source": "stock_financial_abstract_ths",
        }
    )
    out = out.dropna(subset=["report_period", "roe"]).copy()
    if out.empty:
        return out
    out["announcement_date"] = out["report_period"].map(estimate_announcement_date)
    return out.sort_values(["report_period", "announcement_date"]).reset_index(drop=True)


def merge_visible_financial_quality(
    snapshot_df: pd.DataFrame,
    quality_df: pd.DataFrame,
) -> pd.DataFrame:
    if snapshot_df.empty:
        return snapshot_df.copy()
    out = snapshot_df.copy()
    out["signal_date"] = pd.to_datetime(out["signal_date"], errors="coerce")
    if quality_df.empty:
        out["roe"] = pd.NA
        return out

    quality = quality_df.copy()
    quality["symbol"] = quality["symbol"].astype(str)
    quality["announcement_date"] = pd.to_datetime(
        quality["announcement_date"], errors="coerce"
    )
    quality = quality.dropna(subset=["announcement_date", "symbol"]).sort_values(
        ["symbol", "announcement_date"]
    )
    out["symbol"] = out["symbol"].astype(str)

    merged_parts: list[pd.DataFrame] = []
    for symbol, group in out.groupby("symbol", sort=False):
        left = group.sort_values("signal_date")
        right = quality[quality["symbol"] == symbol].sort_values("announcement_date")
        if right.empty:
            left = left.copy()
            left["roe"] = pd.NA
            merged_parts.append(left)
            continue
        merged = pd.merge_asof(
            left,
            right[["announcement_date", "roe"]].sort_values("announcement_date"),
            left_on="signal_date",
            right_on="announcement_date",
            direction="backward",
        )
        merged_parts.append(merged.drop(columns=["announcement_date"]))
    return pd.concat(merged_parts, ignore_index=True)


def build_risk_flags_from_sz_name_changes(
    change_df: pd.DataFrame,
    signal_dates: Iterable[pd.Timestamp],
    symbols: list[str],
) -> pd.DataFrame:
    signal_index = pd.DatetimeIndex(pd.to_datetime(list(signal_dates))).normalize().sort_values()
    rows: list[dict[str, object]] = []
    work = change_df.copy()
    if work.empty:
        for symbol in symbols:
            for date in signal_index:
                rows.append({"symbol": symbol, "date": date, "is_st_like": False})
        return pd.DataFrame(rows)

    work["变更日期"] = pd.to_datetime(work["变更日期"], errors="coerce")
    work["证券代码"] = work["证券代码"].astype(str).str.zfill(6)
    work = work.dropna(subset=["变更日期"]).sort_values(["证券代码", "变更日期"])

    for symbol in symbols:
        code = normalize_ak_symbol(symbol).zfill(6)
        symbol_changes = work[work["证券代码"] == code].copy()
        current_state = False
        ptr = 0
        records = symbol_changes.to_dict("records")
        for signal_date in signal_index:
            while ptr < len(records) and pd.Timestamp(records[ptr]["变更日期"]).normalize() <= signal_date:
                after_name = str(records[ptr].get("变更后简称", "") or "")
                current_state = any(token in after_name for token in ["ST", "*ST", "PT", "退"])
                ptr += 1
            rows.append(
                {
                    "symbol": symbol,
                    "date": pd.Timestamp(signal_date).normalize(),
                    "is_st_like": current_state,
                }
            )
    return pd.DataFrame(rows)


def resolve_mode_configs(mode: str) -> dict[str, dict[str, object]]:
    configs: dict[str, dict[str, object]] = {
        "baseline": {
            "use_buffer": False,
            "use_risk_filters": False,
            "require_momentum": False,
            "entry_buffer": 0.0,
            "exit_buffer": 0.0,
            "min_raw_close": 0.0,
            "low_vol_exclude_pct": 0.0,
        },
        "buffer_only": {
            "use_buffer": True,
            "use_risk_filters": False,
            "require_momentum": False,
            "entry_buffer": 0.0,
            "exit_buffer": 0.0,
            "min_raw_close": 0.0,
            "low_vol_exclude_pct": 0.0,
        },
        "buffer_risk": {
            "use_buffer": True,
            "use_risk_filters": True,
            "require_momentum": False,
            "entry_buffer": 0.0,
            "exit_buffer": 0.0,
            "min_raw_close": 0.0,
            "low_vol_exclude_pct": 0.0,
        },
        "full_combo": {
            "use_buffer": True,
            "use_risk_filters": True,
            "require_momentum": True,
            "entry_buffer": 0.015,
            "exit_buffer": 0.015,
            "min_raw_close": 0.0,
            "low_vol_exclude_pct": 0.0,
        },
        "sharpe_booster": {
            "use_buffer": True,
            "use_risk_filters": True,
            "require_momentum": True,
            "entry_buffer": 0.015,
            "exit_buffer": 0.015,
            "min_raw_close": 2.5,
            "low_vol_exclude_pct": 0.2,
        },
    }
    if mode == "all":
        return configs
    if mode not in configs:
        raise ValueError(f"unsupported mode: {mode}")
    return {mode: configs[mode]}


def load_parquet_cache(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    con = duckdb.connect()
    try:
        return con.execute("SELECT * FROM read_parquet(?)", [path.as_posix()]).df()
    finally:
        con.close()


def save_parquet_cache(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    con = duckdb.connect()
    try:
        con.register("cache_df", df)
        con.execute(f"COPY cache_df TO '{tmp_path.as_posix()}' (FORMAT PARQUET)")
    finally:
        con.close()
    if path.exists():
        path.unlink()
    tmp_path.rename(path)


def fetch_financial_quality_for_symbol(symbol: str) -> pd.DataFrame:
    try:
        raw = ak.stock_financial_abstract_ths(
            symbol=normalize_ak_symbol(symbol).zfill(6),
            indicator="按报告期",
        )
    except Exception:
        return pd.DataFrame(
            columns=[
                "symbol",
                "report_period",
                "announcement_date",
                "roe",
                "source",
            ]
        )
    return normalize_financial_quality_rows(symbol, raw)


def ensure_financial_quality_cache(
    symbols: list[str],
    cache_path: Path,
    refresh: bool = False,
    max_workers: int = 6,
) -> pd.DataFrame:
    if not symbols:
        return pd.DataFrame(
            columns=[
                "symbol",
                "report_period",
                "announcement_date",
                "roe",
                "source",
            ]
        )

    cached = pd.DataFrame() if refresh else load_parquet_cache(cache_path)
    existing_symbols = set(cached["symbol"].astype(str)) if not cached.empty and "symbol" in cached.columns else set()
    missing = [str(s) for s in symbols if str(s) not in existing_symbols]

    frames: list[pd.DataFrame] = []
    if missing:
        ensure_proxy()
        with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as executor:
            future_map = {
                executor.submit(fetch_financial_quality_for_symbol, symbol): symbol
                for symbol in missing
            }
            for future in as_completed(future_map):
                frame = future.result()
                if frame is not None and not frame.empty:
                    frames.append(frame)

    if frames:
        fetched = pd.concat(frames, ignore_index=True)
        cached = pd.concat([cached, fetched], ignore_index=True)
        cached = cached.drop_duplicates(
            subset=["symbol", "report_period"], keep="last"
        ).reset_index(drop=True)
        save_parquet_cache(cache_path, cached)

    if cached.empty:
        return cached
    cached["symbol"] = cached["symbol"].astype(str)
    cached["report_period"] = pd.to_datetime(cached["report_period"], errors="coerce")
    cached["announcement_date"] = pd.to_datetime(
        cached["announcement_date"], errors="coerce"
    )
    return cached[cached["symbol"].isin(symbols)].reset_index(drop=True)


def build_delist_risk_flags(
    delist_df: pd.DataFrame,
    signal_dates: Iterable[pd.Timestamp],
    symbols: list[str],
    code_col: str,
    date_col: str,
    suffix: str,
) -> pd.DataFrame:
    if delist_df is None or delist_df.empty:
        return pd.DataFrame(columns=["symbol", "date", "is_st_like"])

    work = delist_df.copy()
    work[code_col] = work[code_col].astype(str).str.zfill(6)
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    work = work.dropna(subset=[date_col])
    requested = {normalize_ak_symbol(s).zfill(6): s for s in symbols if s.endswith(suffix)}
    if not requested:
        return pd.DataFrame(columns=["symbol", "date", "is_st_like"])

    signal_index = pd.DatetimeIndex(pd.to_datetime(list(signal_dates))).normalize().sort_values()
    rows: list[dict[str, object]] = []
    for code, symbol in requested.items():
        matched = work[work[code_col] == code]
        if matched.empty:
            continue
        cutoff = pd.Timestamp(matched[date_col].iloc[0]).normalize()
        for signal_date in signal_index:
            rows.append(
                {
                    "symbol": symbol,
                    "date": pd.Timestamp(signal_date).normalize(),
                    "is_st_like": bool(signal_date >= cutoff),
                }
            )
    return pd.DataFrame(rows)


def ensure_risk_flag_cache(
    symbols: list[str],
    signal_dates: Iterable[pd.Timestamp],
    cache_path: Path,
    refresh: bool = False,
) -> pd.DataFrame:
    requested_symbols = [str(s) for s in symbols]
    requested_dates = pd.DatetimeIndex(pd.to_datetime(list(signal_dates))).normalize()
    if not requested_symbols or requested_dates.empty:
        return pd.DataFrame(columns=["symbol", "date", "is_st_like"])

    if not refresh and cache_path.exists():
        cached = load_parquet_cache(cache_path)
        if not cached.empty:
            cached["symbol"] = cached["symbol"].astype(str)
            cached["date"] = pd.to_datetime(cached["date"], errors="coerce")
            cached = cached.dropna(subset=["date"])
            cached["date"] = pd.DatetimeIndex(cached["date"]).normalize()
            requested = pd.MultiIndex.from_product(
                [requested_symbols, requested_dates], names=["symbol", "date"]
            )
            cached_index = pd.MultiIndex.from_frame(cached[["symbol", "date"]])
            if requested.isin(cached_index).all():
                return cached[
                    cached["symbol"].isin(requested_symbols)
                    & cached["date"].isin(requested_dates)
                ].reset_index(drop=True)

    ensure_proxy()
    sz_changes = ak.stock_info_sz_change_name(symbol="简称变更")
    sh_delist = ak.stock_info_sh_delist(symbol="全部")
    sz_delist = ak.stock_info_sz_delist(symbol="终止上市公司")

    risk_frames = [
        build_risk_flags_from_sz_name_changes(
            change_df=sz_changes,
            signal_dates=requested_dates,
            symbols=[s for s in requested_symbols if s.endswith(".SZ")],
        ),
        build_delist_risk_flags(
            delist_df=sh_delist,
            signal_dates=requested_dates,
            symbols=requested_symbols,
            code_col="公司代码",
            date_col="暂停上市日期",
            suffix=".SH",
        ),
        build_delist_risk_flags(
            delist_df=sz_delist,
            signal_dates=requested_dates,
            symbols=requested_symbols,
            code_col="证券代码",
            date_col="终止上市日期",
            suffix=".SZ",
        ),
    ]
    risk_frames = [frame for frame in risk_frames if frame is not None and not frame.empty]

    if risk_frames:
        risk_df = pd.concat(risk_frames, ignore_index=True)
        risk_df["date"] = pd.to_datetime(risk_df["date"], errors="coerce")
        risk_df = risk_df.dropna(subset=["date"])
        risk_df["date"] = pd.DatetimeIndex(risk_df["date"]).normalize()
        risk_df["is_st_like"] = risk_df["is_st_like"].fillna(False).astype(bool)
        risk_df = (
            risk_df.groupby(["symbol", "date"], as_index=False)["is_st_like"]
            .max()
            .reset_index(drop=True)
        )
    else:
        risk_df = pd.DataFrame(columns=["symbol", "date", "is_st_like"])

    complete_index = pd.MultiIndex.from_product(
        [requested_symbols, requested_dates], names=["symbol", "date"]
    )
    if risk_df.empty:
        risk_df = complete_index.to_frame(index=False)
        risk_df["is_st_like"] = False
    else:
        risk_df = (
            complete_index.to_frame(index=False)
            .merge(risk_df, on=["symbol", "date"], how="left")
            .fillna({"is_st_like": False})
        )
        risk_df["is_st_like"] = risk_df["is_st_like"].astype(bool)

    save_parquet_cache(cache_path, risk_df)
    return risk_df.reset_index(drop=True)


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


def rank_signal_snapshot(
    snapshot_df: pd.DataFrame,
    signal_date: pd.Timestamp,
    min_list_days: int = 250,
    buy_value_quantile: float = 0.4,
    hold_value_quantile: float = 0.6,
    require_momentum: bool = False,
    require_positive_roe: bool = False,
    exclude_st: bool = False,
    min_raw_close: float = 0.0,
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
    work["market_cap_rank"] = work["total_mv_10k"].rank(
        method="first", ascending=True
    )

    if {"close", "ma20_stock"}.issubset(work.columns):
        work["close"] = pd.to_numeric(work["close"], errors="coerce")
        work["ma20_stock"] = pd.to_numeric(work["ma20_stock"], errors="coerce")
        work["momentum_ok"] = work["close"] > work["ma20_stock"]
    else:
        work["momentum_ok"] = True

    if "raw_close" in work.columns:
        work["raw_close"] = pd.to_numeric(work["raw_close"], errors="coerce")
        work["price_ok"] = work["raw_close"] >= float(min_raw_close)
    else:
        work["price_ok"] = True

    if "roe" in work.columns:
        work["roe"] = pd.to_numeric(work["roe"], errors="coerce")
        work["roe_ok"] = work["roe"] > 0
    else:
        work["roe_ok"] = True

    if "is_st_like" in work.columns:
        work["is_st_like"] = work["is_st_like"].fillna(False).astype(bool)
        work["risk_ok"] = ~work["is_st_like"]
    else:
        work["risk_ok"] = True

    buy_ok = (work["pe_rank"] <= float(buy_value_quantile)) & (
        work["pb_rank"] <= float(buy_value_quantile)
    )
    hold_ok = (work["pe_rank"] <= float(hold_value_quantile)) & (
        work["pb_rank"] <= float(hold_value_quantile)
    )

    if require_momentum:
        buy_ok &= work["momentum_ok"]
        hold_ok &= work["momentum_ok"]
    if require_positive_roe:
        buy_ok &= work["roe_ok"]
        hold_ok &= work["roe_ok"]
    if exclude_st:
        buy_ok &= work["risk_ok"]
        hold_ok &= work["risk_ok"]
    if float(min_raw_close) > 0:
        buy_ok &= work["price_ok"]
        hold_ok &= work["price_ok"]

    work["buy_candidate"] = buy_ok
    work["hold_candidate"] = hold_ok
    return work.reset_index(drop=True)


def select_small_cap_candidates(
    snapshot_df: pd.DataFrame,
    signal_date: pd.Timestamp,
    stock_num: int = 10,
    value_quantile: float = 0.4,
    min_list_days: int = 250,
    require_momentum: bool = False,
    require_positive_roe: bool = False,
    exclude_st: bool = False,
    min_raw_close: float = 0.0,
    low_vol_exclude_pct: float = 0.0,
) -> pd.DataFrame:
    ranked = rank_signal_snapshot(
        snapshot_df=snapshot_df,
        signal_date=signal_date,
        min_list_days=min_list_days,
        buy_value_quantile=value_quantile,
        hold_value_quantile=max(0.6, float(value_quantile)),
        require_momentum=require_momentum,
        require_positive_roe=require_positive_roe,
        exclude_st=exclude_st,
    )
    if ranked.empty:
        return ranked
    work = ranked[ranked["buy_candidate"]].copy()
    if work.empty:
        return work

    if "raw_close" in work.columns and float(min_raw_close) > 0:
        work["raw_close"] = pd.to_numeric(work["raw_close"], errors="coerce")
        work = work[work["raw_close"] >= float(min_raw_close)]
        if work.empty:
            return work

    if "vol20" in work.columns and float(low_vol_exclude_pct) > 0:
        work["vol20"] = pd.to_numeric(work["vol20"], errors="coerce")
        keep_cutoff = max(0.0, 1.0 - float(low_vol_exclude_pct))
        valid = work["vol20"].notna()
        if valid.any():
            work.loc[valid, "vol20_rank"] = work.loc[valid, "vol20"].rank(
                pct=True, ascending=True, method="average"
            )
            work = work[(~valid) | (work["vol20_rank"] <= keep_cutoff)]
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
        min_signal = min(to_ts(x) for x in signal_dates) - pd.Timedelta(days=40)
        max_signal = max(to_ts(x) for x in signal_dates)
        df = con.execute(
            """
            WITH base AS (
              SELECT
                d.date,
                d.symbol,
                d.name,
                d.industry,
                d.pe_ttm,
                d.pb,
                d.total_mv_10k,
                d.list_date,
                d.close,
                r.close AS raw_close
              FROM v_daily_hfq_w_ind_dim d
              LEFT JOIN v_bar_daily_raw r
                ON d.symbol = r.symbol AND d.date = r.date
              WHERE d.date BETWEEN ? AND ?
            ),
            with_ret AS (
              SELECT
                date,
                symbol,
                name,
                industry,
                pe_ttm,
                pb,
                total_mv_10k,
                list_date,
                close,
                raw_close,
                LAG(close) OVER (
                  PARTITION BY symbol
                  ORDER BY date
                ) AS prev_close,
                AVG(close) OVER (
                  PARTITION BY symbol
                  ORDER BY date
                  ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
                ) AS ma20_stock
              FROM base
            ),
            ret AS (
              SELECT
                date,
                symbol,
                name,
                industry,
                pe_ttm,
                pb,
                total_mv_10k,
                list_date,
                close,
                raw_close,
                ma20_stock,
                CASE
                  WHEN prev_close IS NULL OR prev_close = 0 THEN NULL
                  ELSE close / prev_close - 1
                END AS ret_1d
              FROM with_ret
            ),
            vol AS (
              SELECT
                date,
                symbol,
                name,
                industry,
                pe_ttm,
                pb,
                total_mv_10k,
                list_date,
                close,
                raw_close,
                ma20_stock,
                STDDEV_SAMP(ret_1d) OVER (
                  PARTITION BY symbol
                  ORDER BY date
                  ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
                ) AS vol20
              FROM ret
            )
            SELECT
              m.date AS signal_date,
              m.symbol,
              m.name,
              m.industry,
              m.pe_ttm,
              m.pb,
              m.total_mv_10k,
              m.list_date,
              m.close,
              m.raw_close,
              m.ma20_stock,
              m.vol20
            FROM vol m
            INNER JOIN tmp_signal_dates d
              ON m.date = d.signal_date
            WHERE m.pe_ttm IS NOT NULL
              AND m.pb IS NOT NULL
              AND m.total_mv_10k IS NOT NULL
              AND m.list_date IS NOT NULL
              AND m.industry IS NOT NULL
            ORDER BY m.date, m.symbol
            """,
            [min_signal.date(), max_signal.date()],
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


def build_prefetch_symbol_universe(
    snapshot_df: pd.DataFrame,
    prefetch_rank: int,
    min_list_days: int,
    value_quantile: float,
) -> list[str]:
    if snapshot_df.empty:
        return []

    symbols: set[str] = set()
    for signal_date, group in snapshot_df.groupby("signal_date", sort=True):
        ranked = rank_signal_snapshot(
            snapshot_df=group,
            signal_date=to_ts(signal_date),
            min_list_days=min_list_days,
            buy_value_quantile=value_quantile,
            hold_value_quantile=value_quantile,
            require_momentum=False,
            require_positive_roe=False,
            exclude_st=False,
        )
        if ranked.empty:
            continue
        prefetch = ranked[
            (ranked["market_cap_rank"] <= float(prefetch_rank))
            & (ranked["pe_rank"] <= float(value_quantile))
            & (ranked["pb_rank"] <= float(value_quantile))
        ]
        symbols.update(str(x) for x in prefetch["symbol"].tolist())
    return sorted(symbols)


def build_timing_state(
    timing_frame: pd.DataFrame,
    entry_buffer: float = 0.0,
    exit_buffer: float = 0.0,
    initial_state: bool = False,
) -> pd.DataFrame:
    work = timing_frame.copy().sort_values("date").reset_index(drop=True)
    states: list[bool] = []
    current_state = bool(initial_state)
    for _, row in work.iterrows():
        close = row.get("close")
        ma = row.get("ma")
        if pd.isna(close) or pd.isna(ma):
            current_state = False
        else:
            upper = float(ma) * (1.0 + float(entry_buffer))
            lower = float(ma) * (1.0 - float(exit_buffer))
            if float(close) > upper:
                current_state = True
            elif float(close) < lower:
                current_state = False
        states.append(current_state)
    work["safe_market"] = states
    return work


def build_timing_flags(
    index_series: pd.Series,
    signal_dates: Iterable[pd.Timestamp],
    ma_window: int = 20,
    entry_buffer: float = 0.0,
    exit_buffer: float = 0.0,
    initial_state: bool = False,
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
    merged = build_timing_state(
        merged,
        entry_buffer=entry_buffer,
        exit_buffer=exit_buffer,
        initial_state=initial_state,
    )
    return merged[["date", "close", "ma", "safe_market"]]


def update_buffered_holdings(
    previous_holdings: list[str],
    ranked_snapshot: pd.DataFrame,
    stock_num: int = 10,
    buy_rank_cutoff: int = 10,
    hold_rank_cutoff: int = 30,
    hold_value_quantile: float = 0.6,
) -> list[str]:
    if ranked_snapshot.empty:
        return []

    ranked = ranked_snapshot.copy()
    ranked["market_cap_rank"] = pd.to_numeric(
        ranked["market_cap_rank"], errors="coerce"
    )
    ranked["pe_rank"] = pd.to_numeric(ranked["pe_rank"], errors="coerce")
    ranked["pb_rank"] = pd.to_numeric(ranked["pb_rank"], errors="coerce")
    ranked["buy_candidate"] = ranked["buy_candidate"].fillna(False).astype(bool)
    if "hold_candidate" in ranked.columns:
        ranked["hold_candidate"] = ranked["hold_candidate"].fillna(False).astype(bool)
    else:
        ranked["hold_candidate"] = (
            (ranked["pe_rank"] <= float(hold_value_quantile))
            & (ranked["pb_rank"] <= float(hold_value_quantile))
        )

    keep_df = ranked[
        ranked["symbol"].isin(previous_holdings)
        & (ranked["market_cap_rank"] <= float(hold_rank_cutoff))
        & (ranked["pe_rank"] <= float(hold_value_quantile))
        & (ranked["pb_rank"] <= float(hold_value_quantile))
        & ranked["hold_candidate"]
    ].sort_values(["market_cap_rank", "symbol"])
    kept = [str(x) for x in keep_df["symbol"].tolist()]

    buy_df = ranked[
        ranked["buy_candidate"] & (ranked["market_cap_rank"] <= float(buy_rank_cutoff))
    ].sort_values(["market_cap_rank", "symbol"])
    buy_list = [str(x) for x in buy_df["symbol"].tolist() if str(x) not in kept]

    return (kept + buy_list)[: max(1, int(stock_num))]


def merge_signal_risk_flags(
    snapshot_df: pd.DataFrame,
    risk_df: pd.DataFrame,
) -> pd.DataFrame:
    out = snapshot_df.copy()
    out["signal_date"] = pd.to_datetime(out["signal_date"], errors="coerce")
    if risk_df.empty:
        out["is_st_like"] = False
        return out

    risk = risk_df.copy()
    risk["date"] = pd.to_datetime(risk["date"], errors="coerce")
    risk = risk.dropna(subset=["date"])
    risk["date"] = pd.DatetimeIndex(risk["date"]).normalize()
    out = out.merge(
        risk.rename(columns={"date": "signal_date"}),
        on=["symbol", "signal_date"],
        how="left",
    )
    out["is_st_like"] = out["is_st_like"].fillna(False).astype(bool)
    return out


def build_ranked_weekly_snapshots(
    snapshot_df: pd.DataFrame,
    mode_cfg: dict[str, object],
    min_list_days: int,
    buy_value_quantile: float,
    hold_value_quantile: float,
) -> pd.DataFrame:
    if snapshot_df.empty:
        return snapshot_df.iloc[0:0].copy()

    ranked_parts: list[pd.DataFrame] = []
    for signal_date, group in snapshot_df.groupby("signal_date", sort=True):
        ranked = rank_signal_snapshot(
            snapshot_df=group,
            signal_date=to_ts(signal_date),
            min_list_days=min_list_days,
            buy_value_quantile=buy_value_quantile,
            hold_value_quantile=hold_value_quantile,
            require_momentum=bool(mode_cfg["require_momentum"]),
            require_positive_roe=bool(mode_cfg["use_risk_filters"]),
            exclude_st=bool(mode_cfg["use_risk_filters"]),
            min_raw_close=float(mode_cfg.get("min_raw_close", 0.0)),
        )
        low_vol_exclude_pct = float(mode_cfg.get("low_vol_exclude_pct", 0.0))
        if (
            not ranked.empty
            and low_vol_exclude_pct > 0
            and "vol20" in ranked.columns
            and ranked["buy_candidate"].fillna(False).astype(bool).any()
        ):
            buy_mask = ranked["buy_candidate"].fillna(False).astype(bool)
            vol_mask = buy_mask & ranked["vol20"].notna()
            if vol_mask.any():
                ranked.loc[vol_mask, "vol20_rank"] = ranked.loc[vol_mask, "vol20"].rank(
                    pct=True, ascending=True, method="average"
                )
                ranked.loc[vol_mask, "buy_candidate"] = (
                    ranked.loc[vol_mask, "vol20_rank"]
                    <= max(0.0, 1.0 - low_vol_exclude_pct)
                )
        if not ranked.empty:
            ranked_parts.append(ranked)
    if not ranked_parts:
        return snapshot_df.iloc[0:0].copy()
    return pd.concat(ranked_parts, ignore_index=True)


def build_mode_rebalance_plan(
    schedule_df: pd.DataFrame,
    ranked_weekly_df: pd.DataFrame,
    timing_df: pd.DataFrame,
    stock_num: int,
    mode_cfg: dict[str, object],
    hold_rank_cutoff: int,
    hold_value_quantile: float,
) -> tuple[dict[pd.Timestamp, list[str]], pd.DataFrame]:
    ranked_map: dict[pd.Timestamp, pd.DataFrame] = {}
    if not ranked_weekly_df.empty:
        work = ranked_weekly_df.copy()
        work["signal_date"] = pd.to_datetime(work["signal_date"], errors="coerce")
        work = work.dropna(subset=["signal_date"])
        for signal_date, group in work.groupby("signal_date", sort=True):
            ranked_map[to_ts(signal_date)] = group.copy()

    timing_map: dict[pd.Timestamp, bool] = {}
    if not timing_df.empty:
        work = timing_df.copy()
        work["date"] = pd.to_datetime(work["date"], errors="coerce")
        work = work.dropna(subset=["date"])
        for _, row in work.iterrows():
            timing_map[to_ts(row["date"])] = bool(row["safe_market"])

    plan: dict[pd.Timestamp, list[str]] = {}
    detail_parts: list[pd.DataFrame] = []
    current_holdings: list[str] = []

    for _, row in schedule_df.sort_values("signal_date").iterrows():
        signal_date = to_ts(row["signal_date"])
        rebalance_date = to_ts(row["rebalance_date"])
        ranked = ranked_map.get(signal_date, pd.DataFrame())
        safe_market = timing_map.get(signal_date, False)

        if not safe_market:
            current_holdings = []
        else:
            if bool(mode_cfg["use_buffer"]):
                current_holdings = update_buffered_holdings(
                    previous_holdings=current_holdings,
                    ranked_snapshot=ranked,
                    stock_num=stock_num,
                    buy_rank_cutoff=stock_num,
                    hold_rank_cutoff=hold_rank_cutoff,
                    hold_value_quantile=hold_value_quantile,
                )
            else:
                buy_df = ranked[ranked["buy_candidate"]].copy() if not ranked.empty else ranked
                buy_df = buy_df.sort_values(["total_mv_10k", "symbol"], ascending=[True, True])
                current_holdings = [
                    str(x) for x in buy_df["symbol"].head(stock_num).tolist()
                ]

        plan[rebalance_date] = list(current_holdings)

        detail = ranked.copy() if not ranked.empty else pd.DataFrame(columns=["symbol"])
        if detail.empty:
            detail = pd.DataFrame(
                {
                    "signal_date": [signal_date],
                    "rebalance_date": [rebalance_date],
                    "safe_market": [safe_market],
                    "selected": [False],
                }
            )
        else:
            detail["rebalance_date"] = rebalance_date
            detail["safe_market"] = safe_market
            detail["selected"] = detail["symbol"].isin(current_holdings)
            detail = detail[detail["buy_candidate"] | detail["selected"]]
        detail_parts.append(detail)

    weekly_out = (
        pd.concat(detail_parts, ignore_index=True)
        if detail_parts
        else pd.DataFrame()
    )
    return plan, weekly_out


def make_mode_output_path(path: Path, mode_name: str, multi_mode: bool) -> Path:
    if not multi_mode and mode_name == "full_combo":
        return path
    return path.with_name(f"{path.stem}_{mode_name}{path.suffix}")


def result_to_summary_row(mode_name: str, metrics_df: pd.DataFrame) -> dict[str, object]:
    metric_map = dict(zip(metrics_df["metric"], metrics_df["value"]))
    return {
        "mode": mode_name,
        "start_time": metric_map.get("start_time"),
        "end_time": metric_map.get("end_time"),
        "initial_market_value": metric_map.get("initial_market_value"),
        "end_market_value": metric_map.get("end_market_value"),
        "total_return_pct": metric_map.get("total_return_pct"),
        "annualized_return": metric_map.get("annualized_return"),
        "max_drawdown_pct": metric_map.get("max_drawdown_pct"),
        "sharpe_ratio": metric_map.get("sharpe_ratio"),
        "trade_count": metric_map.get("trade_count"),
        "win_rate": metric_map.get("win_rate"),
        "total_commission": metric_map.get("total_commission"),
    }


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

    mode_configs = resolve_mode_configs(args.mode)
    multi_mode = len(mode_configs) > 1

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
        index_series = fetch_index_price(INDEX_CODE, args.index_cache)

        risk_symbols: list[str] = []
        quality_df = pd.DataFrame()
        risk_df = pd.DataFrame()
        if any(bool(cfg["use_risk_filters"]) for cfg in mode_configs.values()):
            risk_symbols = build_prefetch_symbol_universe(
                snapshot_df=snapshot_df,
                prefetch_rank=max(int(args.prefetch_rank), int(args.hold_rank_cutoff)),
                min_list_days=max(0, int(args.min_list_days)),
                value_quantile=float(args.hold_value_quantile),
            )
            print(f"[INFO] prefetch symbol universe for risk filters: {len(risk_symbols)}")
            quality_df = ensure_financial_quality_cache(
                symbols=risk_symbols,
                cache_path=args.financial_cache,
                refresh=bool(args.refresh_financial_cache),
                max_workers=max(1, int(args.financial_max_workers)),
            )
            risk_df = ensure_risk_flag_cache(
                symbols=risk_symbols,
                signal_dates=signal_dates,
                cache_path=args.risk_cache,
                refresh=bool(args.refresh_risk_cache),
            )
    finally:
        con.close()

    enriched_snapshot_base = snapshot_df.copy()
    if not quality_df.empty:
        enriched_snapshot_base = merge_visible_financial_quality(
            enriched_snapshot_base, quality_df
        )
    else:
        enriched_snapshot_base["roe"] = pd.NA
    if not risk_df.empty:
        enriched_snapshot_base = merge_signal_risk_flags(enriched_snapshot_base, risk_df)
    else:
        enriched_snapshot_base["is_st_like"] = False

    summary_rows: list[dict[str, object]] = []

    for mode_name, mode_cfg in mode_configs.items():
        timing_df = build_timing_flags(
            index_series,
            signal_dates=signal_dates,
            ma_window=max(1, int(args.timing_ma)),
            entry_buffer=float(mode_cfg["entry_buffer"]),
            exit_buffer=float(mode_cfg["exit_buffer"]),
        )

        ranked_weekly_df = build_ranked_weekly_snapshots(
            snapshot_df=enriched_snapshot_base,
            mode_cfg=mode_cfg,
            min_list_days=max(0, int(args.min_list_days)),
            buy_value_quantile=float(args.value_quantile),
            hold_value_quantile=float(args.hold_value_quantile),
        )
        rebalance_plan, weekly_out = build_mode_rebalance_plan(
            schedule_df=schedule_df,
            ranked_weekly_df=ranked_weekly_df,
            timing_df=timing_df,
            stock_num=max(1, int(args.stock_num)),
            mode_cfg=mode_cfg,
            hold_rank_cutoff=max(1, int(args.hold_rank_cutoff)),
            hold_value_quantile=float(args.hold_value_quantile),
        )
        daily_plan = build_daily_plan(trading_dates, rebalance_plan)

        symbols = sorted({s for syms in daily_plan.values() for s in syms})
        if not symbols:
            print(f"[WARN] mode={mode_name}: no symbols selected, skip")
            continue

        con = duckdb.connect(args.db.as_posix())
        try:
            market_data = load_price_data_for_symbols(
                con,
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
            )
        finally:
            con.close()
        if not market_data:
            print(f"[WARN] mode={mode_name}: no market data loaded, skip")
            continue

        valid_symbols = set(market_data.keys())
        rebalance_plan = {
            date: [s for s in symbols_for_day if s in valid_symbols]
            for date, symbols_for_day in rebalance_plan.items()
        }
        daily_plan = build_daily_plan(trading_dates, rebalance_plan)
        if not weekly_out.empty and "symbol" in weekly_out.columns:
            weekly_out = weekly_out[weekly_out["symbol"].isin(valid_symbols)].copy()

        plan_days, non_empty_days, avg_n, min_n, max_n, union_n = summarize_plan(daily_plan)
        print(
            f"[INFO] mode={mode_name} | backtest window: {start_date.date()} -> {end_date.date()} | "
            f"stock_num={max(1, int(args.stock_num))} value_quantile={float(args.value_quantile):.2f}"
        )
        print(
            f"[INFO] mode={mode_name} daily plan: days={plan_days}, non_empty_days={non_empty_days}, "
            f"avg/min/max per day={avg_n:.2f}/{min_n}/{max_n}, symbol_coverage={union_n}"
        )
        print(f"[INFO] mode={mode_name} loading symbols: {len(valid_symbols)}")

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
        print(f"mode={mode_name} total_return_pct={result.metrics.total_return_pct:.2f}")
        print(f"mode={mode_name} sharpe_ratio={result.metrics.sharpe_ratio:.2f}")
        print(f"mode={mode_name} max_drawdown_pct={result.metrics.max_drawdown_pct:.2f}")

        out_metrics = make_mode_output_path(args.out_metrics, mode_name, multi_mode)
        out_trades = make_mode_output_path(args.out_trades, mode_name, multi_mode)
        out_weekly = make_mode_output_path(args.out_weekly_picks, mode_name, multi_mode)
        out_equity = make_mode_output_path(args.out_equity, mode_name, multi_mode)
        out_metrics.parent.mkdir(parents=True, exist_ok=True)
        out_trades.parent.mkdir(parents=True, exist_ok=True)
        out_weekly.parent.mkdir(parents=True, exist_ok=True)
        out_equity.parent.mkdir(parents=True, exist_ok=True)

        metrics_out = metrics_to_frame(result.metrics_df.copy())
        metrics_out.to_csv(out_metrics, index=False, encoding="utf-8-sig")
        result.trades_df.to_csv(out_trades, index=False, encoding="utf-8-sig")
        weekly_out.to_csv(out_weekly, index=False, encoding="utf-8-sig")

        equity_df = extract_equity_frame(result)
        if equity_df.empty:
            pd.DataFrame(
                {
                    "message": [
                        "AKQuant result did not expose an equity dataframe in a known attribute"
                    ]
                }
            ).to_csv(out_equity, index=False, encoding="utf-8-sig")
        else:
            equity_df.to_csv(out_equity, index=False, encoding="utf-8-sig")

        summary_rows.append(result_to_summary_row(mode_name, metrics_out))
        print(f"[DONE] mode={mode_name} metrics     -> {out_metrics}")
        print(f"[DONE] mode={mode_name} trades      -> {out_trades}")
        print(f"[DONE] mode={mode_name} weekly picks -> {out_weekly}")
        print(f"[DONE] mode={mode_name} equity      -> {out_equity}")

    if summary_rows:
        summary_path = args.out_metrics.with_name(
            f"{args.out_metrics.stem}_summary.csv"
        )
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False, encoding="utf-8-sig")
        print(f"[DONE] summary     -> {summary_path}")


if __name__ == "__main__":
    main()
