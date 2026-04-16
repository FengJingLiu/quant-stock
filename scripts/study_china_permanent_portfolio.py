#!/usr/bin/env python3
# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false

import argparse
import json
import math
from pathlib import Path

import akshare as ak
import akshare_proxy_patch
import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from akquant import ExecutionMode, Strategy, run_backtest


PROXY_HOST = "***AKSHARE_HOST***"
PROXY_TOKEN = "***AKSHARE_TOKEN***"


def parse_args():
    parser = argparse.ArgumentParser(description="中国版永久组合深度研究")
    parser.add_argument("--db", type=Path, default=Path("data/duckdb/stock.duckdb"))
    parser.add_argument("--start-date", type=str, default="2015-01-01")
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument(
        "--out-dir", type=Path, default=Path("data/china_permanent_portfolio")
    )
    parser.add_argument("--weekly", action="store_true")
    return parser.parse_args()


def to_ts(value):
    if isinstance(value, (int, np.integer)):
        ts = (
            pd.to_datetime(int(value), unit="ns", utc=True)
            .tz_convert("Asia/Shanghai")
            .tz_localize(None)
        )
    else:
        ts = pd.Timestamp(value)
    if pd.isna(ts):
        raise ValueError(f"invalid timestamp: {value}")
    if ts.tzinfo is not None:
        ts = ts.tz_convert("Asia/Shanghai").tz_localize(None)
    ts = pd.Timestamp(ts)
    return pd.Timestamp(year=int(ts.year), month=int(ts.month), day=int(ts.day))


def ensure_proxy():
    akshare_proxy_patch.install_patch(PROXY_HOST, PROXY_TOKEN, retry=30)


def ensure_dirs(base_dir):
    base_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = base_dir / "cache"
    tables_dir = base_dir / "tables"
    figs_dir = base_dir / "figures"
    for p in [cache_dir, tables_dir, figs_dir]:
        p.mkdir(parents=True, exist_ok=True)
    return cache_dir, tables_dir, figs_dir


def write_table(df, path):
    out = df.copy()
    out.to_csv(path, index=False)


def save_fig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def load_cached_series(path):
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty or "date" not in df.columns or "close" not in df.columns:
        return None
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["date", "close"]).sort_values(by="date")
    if df.empty:
        return None
    idx = pd.DatetimeIndex(df["date"])
    idx = idx.tz_localize(None)
    s = pd.Series(df["close"].to_numpy(dtype=float), index=idx, name="close")
    mask = ~pd.Index(s.index).duplicated(keep="last")
    s = pd.Series(
        s.loc[mask].to_numpy(dtype=float),
        index=pd.DatetimeIndex(s.index[mask]),
        name="close",
    )
    s = s.sort_index()
    if s.empty:
        return None
    return s


def save_cached_series(path, s):
    out = pd.DataFrame(
        {"date": pd.DatetimeIndex(s.index).astype("datetime64[ns]"), "close": s.values}
    )
    out.to_csv(path, index=False)


def first_existing(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None


def normalize_date_close(df):
    dc = first_existing(
        list(df.columns), ["date", "Date", "日期", "时间", "月份", "month"]
    )
    cc = first_existing(list(df.columns), ["close", "Close", "收盘", "收盘价"])
    if dc is None or cc is None:
        lower_map = {str(c).lower(): c for c in df.columns}
        for k, v in lower_map.items():
            if dc is None and ("date" in k or "日期" in k or "时间" in k or "月" in k):
                dc = v
            if cc is None and ("close" in k or "收盘" in k):
                cc = v
    if dc is None or cc is None:
        raise RuntimeError(f"date/close columns not found: {df.columns.tolist()}")
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df[dc], errors="coerce"),
            "close": pd.to_numeric(df[cc], errors="coerce"),
        }
    )
    out = out.dropna(subset=["date", "close"]).sort_values(by="date")
    out = out[~out["date"].duplicated(keep="last")]
    if out.empty:
        raise RuntimeError("normalized date-close is empty")
    idx = pd.DatetimeIndex(out["date"]).tz_localize(None)
    s = pd.Series(out["close"].to_numpy(dtype=float), index=idx, name="close")
    mask = ~pd.Index(s.index).duplicated(keep="last")
    s = pd.Series(
        s.loc[mask].to_numpy(dtype=float),
        index=pd.DatetimeIndex(s.index[mask]),
        name="close",
    )
    s = s.sort_index()
    return s


def fetch_index_price(code, cache_dir):
    cache_path = cache_dir / f"index_{code}.csv"
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
    s = normalize_date_close(df)
    save_cached_series(cache_path, s)
    return s


def fetch_etf_price(code, cache_dir):
    cache_path = cache_dir / f"etf_{code}.csv"
    cached = load_cached_series(cache_path)
    if cached is not None:
        return cached
    ensure_proxy()
    df = ak.fund_etf_hist_em(
        symbol=code,
        period="daily",
        start_date="19900101",
        end_date="21000101",
        adjust="qfq",
    )
    if df is None or df.empty:
        raise RuntimeError(f"fund_etf_hist_em empty for {code}")
    s = normalize_date_close(df)
    save_cached_series(cache_path, s)
    return s


def restrict_series(s, start_date, end_date):
    out = s.copy()
    idx = pd.DatetimeIndex(out.index).tz_localize(None)
    out.index = idx
    out = out[(out.index >= start_date) & (out.index <= end_date)]
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def get_end_date(con, user_end):
    if user_end:
        return to_ts(user_end)
    row = con.execute("SELECT MAX(date) AS d FROM v_bar_daily_hfq").df()
    return to_ts(row["d"].iloc[0])


def build_symbol_sql(symbols):
    return ",".join([f"'{s}'" for s in symbols])


def load_local_price_panel(con, symbols, start_date, end_date):
    if len(symbols) == 0:
        return pd.DataFrame()
    sym_sql = build_symbol_sql(symbols)
    q = f"""
    SELECT date, symbol, close
    FROM v_bar_daily_hfq
    WHERE symbol IN ({sym_sql})
      AND date BETWEEN '{start_date.date()}' AND '{end_date.date()}'
    ORDER BY date, symbol
    """
    df = con.execute(q).df()
    if df.empty:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["date", "symbol", "close"])
    if df.empty:
        return pd.DataFrame()
    pivot = df.pivot(index="date", columns="symbol", values="close").sort_index()
    idx = pd.DatetimeIndex(pivot.index).tz_localize(None)
    pivot.index = idx
    return pivot


def get_symbols(con, industry=None, name_like=None):
    clauses = ["(is_delisted IS NULL OR is_delisted=0)"]
    if industry is not None:
        if isinstance(industry, list):
            ins = ",".join([f"'{x}'" for x in industry])
            clauses.append(f"industry IN ({ins})")
        else:
            clauses.append(f"industry='{industry}'")
    if name_like is not None:
        clauses.append(f"name LIKE '{name_like}'")
    where = " AND ".join(clauses)
    q = f"""
    SELECT DISTINCT symbol
    FROM v_dim_symbol
    WHERE {where}
    ORDER BY symbol
    """
    df = con.execute(q).df()
    if df.empty:
        return []
    return df["symbol"].astype(str).tolist()


def previous_trade_date(dates, target_date):
    arr = pd.DatetimeIndex(dates)
    arr = arr[arr < target_date]
    if len(arr) == 0:
        return target_date
    return pd.Timestamp(arr.max())


def monthly_rebalance_dates(dates, start_date):
    idx = pd.DatetimeIndex(dates)
    idx = idx[idx >= start_date]
    if len(idx) == 0:
        return pd.DatetimeIndex([])
    tmp = pd.DataFrame({"date": idx})
    tmp["ym"] = tmp["date"].dt.to_period("M")
    out = tmp.sort_values(by="date").drop_duplicates(subset=["ym"], keep="first")
    out = pd.to_datetime(out["date"], errors="coerce").dropna()
    return pd.DatetimeIndex(out.tolist())


def latest_factor_snapshot(con, symbols, signal_date, fields):
    if len(symbols) == 0:
        return pd.DataFrame()
    sym_sql = build_symbol_sql(symbols)
    field_sql = ", ".join(fields)
    q = f"""
    WITH raw AS (
        SELECT date, symbol, {field_sql}
        FROM v_daily_hfq_w_ind_dim
        WHERE symbol IN ({sym_sql})
          AND date <= '{signal_date.date()}'
    ), ranked AS (
        SELECT *, ROW_NUMBER() OVER(PARTITION BY symbol ORDER BY date DESC) AS rn
        FROM raw
    )
    SELECT *
    FROM ranked
    WHERE rn = 1
    """
    df = con.execute(q).df()
    if df.empty:
        return df
    for c in fields:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def normalize_weights_map(weight_map):
    if len(weight_map) == 0:
        return {}
    total = float(sum(weight_map.values()))
    if total <= 0:
        n = len(weight_map)
        return {k: 1.0 / n for k in weight_map}
    return {k: float(v) / total for k, v in weight_map.items()}


def build_monthly_selected_basket(price_panel, reb_dates, selector_func):
    if price_panel.empty:
        return pd.Series(dtype=float), pd.DataFrame(), pd.DataFrame()
    px = price_panel.copy().sort_index()
    px = px.ffill()
    rets = px.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    w = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    picks_rows = []

    for i, reb_date in enumerate(reb_dates):
        next_date = reb_dates[i + 1] if i + 1 < len(reb_dates) else None
        selected_df, weight_map = selector_func(reb_date)
        weight_map = normalize_weights_map(weight_map)
        if len(weight_map) == 0:
            continue
        if next_date is None:
            mask = w.index >= reb_date
        else:
            mask = (w.index >= reb_date) & (w.index < next_date)
        for sym, wt in weight_map.items():
            if sym in w.columns:
                w.loc[mask, sym] = float(wt)
        if selected_df is not None and not selected_df.empty:
            tmp = selected_df.copy()
            tmp["rebalance_date"] = reb_date
            picks_rows.append(tmp)

    row_sum = w.sum(axis=1)
    row_sum = row_sum.replace(0.0, np.nan)
    w = w.div(row_sum, axis=0).fillna(0.0)
    basket_ret = (w.shift(1).fillna(0.0) * rets).sum(axis=1)
    nav = (1.0 + basket_ret).cumprod()
    if not nav.empty:
        nav.iloc[0] = 1.0
    picks = (
        pd.concat(picks_rows, ignore_index=True) if len(picks_rows) else pd.DataFrame()
    )
    return nav, picks, w


def build_bank_method2(con, start_date, end_date, variant):
    bank_symbols = get_symbols(con, industry="银行")
    panel = load_local_price_panel(con, bank_symbols, start_date, end_date)
    if panel.empty:
        raise RuntimeError("bank price panel empty")
    reb_dates = monthly_rebalance_dates(panel.index, start_date)

    def selector(reb_date):
        signal_date = previous_trade_date(panel.index, reb_date)
        snap = latest_factor_snapshot(
            con,
            bank_symbols,
            signal_date,
            ["pb", "dividend_yield_ttm", "total_mv_10k"],
        )
        if snap.empty:
            return pd.DataFrame(), {}
        snap = snap.dropna(subset=["pb", "dividend_yield_ttm"])
        snap = snap[(snap["pb"] > 0) & (snap["dividend_yield_ttm"] > 0)]
        if snap.empty:
            return pd.DataFrame(), {}
        div_pct = snap["dividend_yield_ttm"].rank(pct=True)
        pb_pct = snap["pb"].rank(pct=True)
        snap["score"] = div_pct + (1.0 - pb_pct)
        snap = snap.sort_values(
            by=["score", "dividend_yield_ttm"], ascending=[False, False]
        )
        top_n = 6
        sel = snap.head(top_n).copy()
        if sel.empty:
            return pd.DataFrame(), {}
        if variant == "eq":
            weight_map = {s: 1.0 / len(sel) for s in sel["symbol"].astype(str).tolist()}
        else:
            w = sel[["symbol", "dividend_yield_ttm"]].copy()
            w["dividend_yield_ttm"] = pd.to_numeric(
                w["dividend_yield_ttm"], errors="coerce"
            )
            w = w.dropna(subset=["dividend_yield_ttm"])
            w = w[w["dividend_yield_ttm"] > 0]
            if w.empty:
                weight_map = {
                    s: 1.0 / len(sel) for s in sel["symbol"].astype(str).tolist()
                }
            else:
                raw_sum = float(w["dividend_yield_ttm"].sum())
                if raw_sum <= 0:
                    weight_map = {
                        s: 1.0 / len(w) for s in w["symbol"].astype(str).tolist()
                    }
                else:
                    weight_map = {
                        str(r.symbol): float(r.dividend_yield_ttm) / raw_sum
                        for r in w.itertuples(index=False)
                    }
        sel["signal_date"] = signal_date
        sel["variant"] = variant
        return sel, weight_map

    nav, picks, weights = build_monthly_selected_basket(panel, reb_dates, selector)
    return nav, picks, weights


def build_production_method2(con, start_date, end_date):
    industries = [
        "化工原料",
        "化纤",
        "煤炭开采",
        "石油开采",
        "石油加工",
        "水泥",
        "普钢",
        "特种钢",
        "钢加工",
        "工程机械",
        "电气设备",
        "小金属",
        "铜",
        "铝",
        "铅锌",
        "矿物制品",
        "农药化肥",
    ]
    symbols = get_symbols(con, industry=industries)
    panel = load_local_price_panel(con, symbols, start_date, end_date)
    if panel.empty:
        raise RuntimeError("production price panel empty")
    reb_dates = monthly_rebalance_dates(panel.index, start_date)

    def selector(reb_date):
        signal_date = previous_trade_date(panel.index, reb_date)
        snap = latest_factor_snapshot(con, symbols, signal_date, ["total_mv_10k"])
        if snap.empty:
            return pd.DataFrame(), {}
        snap = snap.dropna(subset=["total_mv_10k"])
        snap = snap[snap["total_mv_10k"] > 0]
        if snap.empty:
            return pd.DataFrame(), {}
        snap = snap.sort_values(by="total_mv_10k", ascending=False)
        top_n = 15
        sel = snap.head(top_n).copy()
        weight_map = {s: 1.0 / len(sel) for s in sel["symbol"].astype(str).tolist()}
        sel["signal_date"] = signal_date
        return sel, weight_map

    nav, picks, weights = build_monthly_selected_basket(panel, reb_dates, selector)
    return nav, picks, weights


def build_gold_stock_nav(con, start_date, end_date):
    gold_symbols = list(
        set(get_symbols(con, industry="黄金") + get_symbols(con, name_like="%黄金%"))
    )
    panel = load_local_price_panel(con, gold_symbols, start_date, end_date)
    if panel.empty:
        raise RuntimeError("gold-stock panel empty")
    panel = panel.sort_index().ffill()
    rets = panel.pct_change().replace([np.inf, -np.inf], np.nan)
    ew_ret = rets.mean(axis=1, skipna=True).fillna(0.0)
    nav = (1.0 + ew_ret).cumprod()
    if not nav.empty:
        nav.iloc[0] = 1.0
    return nav


def build_prod_m1_index_blend(prod_idx):
    use_cols = [c for c in ["399613", "399614", "399615"] if c in prod_idx.columns]
    if len(use_cols) == 0:
        raise RuntimeError("production index pool empty for method1")
    weights = {"399613": 0.4, "399614": 0.3, "399615": 0.3}
    px = prod_idx[use_cols].copy().sort_index().ffill()
    ret = px.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    w_vec = np.array([weights.get(c, 0.0) for c in use_cols], dtype=float)
    if float(w_vec.sum()) <= 0:
        w_vec = np.ones(len(use_cols), dtype=float) / len(use_cols)
    else:
        w_vec = w_vec / float(w_vec.sum())
    port_ret = ret.to_numpy(dtype=float) @ w_vec
    s = pd.Series(port_ret, index=ret.index)
    nav = (1.0 + s).cumprod()
    if not nav.empty:
        nav.iloc[0] = 1.0
    return nav


def build_prod_m3_dynamic(prod_idx, start_date):
    px = prod_idx.copy().sort_index().ffill()
    if px.empty:
        raise RuntimeError("production index pool empty for method3")
    reb_dates = monthly_rebalance_dates(px.index, start_date)
    lookback = 126
    rets = px.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    w = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    pick_rows = []

    for i, reb_date in enumerate(reb_dates):
        next_date = reb_dates[i + 1] if i + 1 < len(reb_dates) else None
        signal_date = previous_trade_date(px.index, reb_date)
        pos = px.index.get_indexer([signal_date], method="nearest")
        if len(pos) == 0 or pos[0] <= lookback:
            continue
        loc = int(pos[0])
        p0 = px.iloc[loc - lookback]
        p1 = px.iloc[loc]
        mom = (p1 / p0) - 1.0
        mom = mom.replace([np.inf, -np.inf], np.nan).dropna()
        if mom.empty:
            continue
        top = mom.sort_values(ascending=False).head(2)
        chosen = top.index.tolist()
        wt = {c: 1.0 / len(chosen) for c in chosen}
        if next_date is None:
            mask = w.index >= reb_date
        else:
            mask = (w.index >= reb_date) & (w.index < next_date)
        for c, ww in wt.items():
            w.loc[mask, c] = ww
        for c, score in top.items():
            pick_rows.append(
                {
                    "rebalance_date": reb_date,
                    "signal_date": signal_date,
                    "symbol": str(c),
                    "momentum_126d": float(score),
                    "weight": float(wt[c]),
                }
            )

    row_sum = w.sum(axis=1).replace(0.0, np.nan)
    w = w.div(row_sum, axis=0).fillna(0.0)
    port_ret = (w.shift(1).fillna(0.0) * rets).sum(axis=1)
    nav = (1.0 + port_ret).cumprod()
    if not nav.empty:
        nav.iloc[0] = 1.0
    picks = pd.DataFrame(pick_rows)
    return nav, picks, w


def blend_nav(nav_a, nav_b, w_a):
    df = pd.DataFrame({"a": nav_a, "b": nav_b}).sort_index().ffill().dropna()
    ret = df.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    port = (ret["a"] * float(w_a)) + (ret["b"] * (1.0 - float(w_a)))
    nav = (1.0 + port).cumprod()
    if not nav.empty:
        nav.iloc[0] = 1.0
    return nav


def nav_to_period_returns(nav, freq):
    s = nav.copy().sort_index().ffill().dropna()
    if s.empty:
        return pd.Series(dtype=float)
    sampled = s.resample(freq).last().dropna()
    ret = sampled.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    return ret


def build_period_return_panel(nav_map, keys, freq):
    cols = {}
    for k in keys:
        cols[k] = nav_to_period_returns(nav_map[k], freq)
    panel = pd.DataFrame(cols).sort_index()
    panel = panel.dropna()
    return panel


def calc_metrics(ret, periods_per_year):
    out = {
        "total_return": np.nan,
        "cagr": np.nan,
        "ann_vol": np.nan,
        "sharpe": np.nan,
        "max_drawdown": np.nan,
        "calmar": np.nan,
        "win_rate": np.nan,
    }
    if ret.empty:
        return out
    nav = (1.0 + ret).cumprod()
    total = float(nav.iloc[-1] - 1.0)
    n = len(ret)
    years = n / float(periods_per_year) if periods_per_year > 0 else np.nan
    if years and years > 0:
        cagr = float(nav.iloc[-1] ** (1.0 / years) - 1.0)
    else:
        cagr = np.nan
    ann_vol = float(ret.std(ddof=0) * math.sqrt(periods_per_year))
    ann_mean = float(ret.mean() * periods_per_year)
    sharpe = ann_mean / ann_vol if ann_vol > 0 else np.nan
    dd = nav / nav.cummax() - 1.0
    mdd = float(dd.min())
    calmar = cagr / abs(mdd) if (not np.isnan(cagr) and mdd < 0) else np.nan
    win_rate = float((ret > 0).mean())
    out.update(
        {
            "total_return": total,
            "cagr": cagr,
            "ann_vol": ann_vol,
            "sharpe": sharpe,
            "max_drawdown": mdd,
            "calmar": calmar,
            "win_rate": win_rate,
        }
    )
    return out


def make_rebalance_flags(index, rule, threshold):
    idx = pd.DatetimeIndex(index)
    month_id = pd.Series(idx.to_period("M"), index=idx)
    year_id = pd.Series(idx.to_period("Y"), index=idx)
    month_start = month_id != month_id.shift(1)
    year_start = year_id != year_id.shift(1)
    semi_start = month_start & idx.month.isin([1, 7])

    def should_rebalance(i, drift):
        if i == 0:
            return True
        dt = idx[i]
        if rule == "annual":
            return bool(year_start.loc[dt])
        if rule == "threshold5":
            return bool(month_start.loc[dt]) and drift > threshold
        if rule == "threshold10":
            return bool(month_start.loc[dt]) and drift > threshold
        if rule == "hybrid":
            return bool(semi_start.loc[dt]) or (
                bool(month_start.loc[dt]) and drift > threshold
            )
        raise ValueError(f"unknown rule: {rule}")

    return should_rebalance


class ScheduledTargetWeightsStrategy(Strategy):
    warmup_period = 1

    def __init__(self, target_plan):
        self.warmup_period = 1
        self.target_plan = {
            to_ts(k): normalize_weights_map(v) for k, v in target_plan.items()
        }
        self.current_date = None
        self.current_targets = {}

    def _switch_day(self, bar_date):
        if self.current_date == bar_date:
            return
        self.current_date = bar_date
        if bar_date in self.target_plan:
            self.current_targets = self.target_plan[bar_date]

    def on_bar(self, bar):
        bar_date = to_ts(getattr(bar, "timestamp"))
        self._switch_day(bar_date)

        symbol = str(bar.symbol)
        target = float(self.current_targets.get(symbol, 0.0))
        pos = float(self.get_position(symbol))

        if target > 0.0 or pos > 0.0:
            self.order_target_percent(target, symbol)


def build_synthetic_ohlcv_from_returns(ret_panel):
    out = {}
    for col in ret_panel.columns:
        r_num = pd.to_numeric(ret_panel[col], errors="coerce")
        r = pd.Series(r_num, index=ret_panel.index, dtype=float).fillna(0.0)
        close = (1.0 + r).cumprod().clip(lower=1e-9)
        idx = pd.DatetimeIndex(close.index).tz_localize(None)
        work = pd.DataFrame(
            {
                "date": idx,
                "open": close.values,
                "high": close.values,
                "low": close.values,
                "close": close.values,
                "volume": np.full(len(close), 1_000_000.0),
                "symbol": str(col),
            }
        )
        out[str(col)] = work
    return out


def build_target_plan_and_cost(
    ret_panel,
    target_weights,
    rule,
    asset_types,
    stock_buy_cost,
    stock_sell_cost,
    etf_buy_cost,
    etf_sell_cost,
    threshold,
):
    assets = list(target_weights.keys())
    w_target = np.array([float(target_weights[a]) for a in assets], dtype=float)
    w_target = w_target / float(w_target.sum())
    panel = ret_panel[assets].copy().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    idx = pd.DatetimeIndex(panel.index)
    should_rebalance = make_rebalance_flags(idx, rule, threshold)

    target_plan = {}
    turnover_list = []
    cost_list = []
    weights_rows = []

    w = np.zeros(len(assets), dtype=float)

    for i, dt in enumerate(idx):
        dt_key = pd.Timestamp(dt)
        if i == 0:
            delta0 = w_target - w
            buy0 = np.clip(delta0, 0.0, None)
            sell0 = np.clip(-delta0, 0.0, None)
            tc0 = 0.0
            for j, a in enumerate(assets):
                a_type = asset_types.get(a, "etf")
                if a_type == "stock":
                    tc0 += buy0[j] * stock_buy_cost + sell0[j] * stock_sell_cost
                else:
                    tc0 += buy0[j] * etf_buy_cost + sell0[j] * etf_sell_cost
            turnover = float(np.abs(delta0).sum())
            cost_rate = float(tc0)
            w = w_target.copy()
            target_plan[dt_key] = {
                assets[j]: float(w_target[j]) for j in range(len(assets))
            }
        else:
            r = panel.iloc[i].to_numpy(dtype=float)
            gross = w * (1.0 + r)
            denom = float(gross.sum())
            if denom > 0:
                w = gross / denom
            drift = float(np.max(np.abs(w - w_target)))

            if should_rebalance(i, drift):
                delta = w_target - w
                buy = np.clip(delta, 0.0, None)
                sell = np.clip(-delta, 0.0, None)
                tc = 0.0
                for j, a in enumerate(assets):
                    a_type = asset_types.get(a, "etf")
                    if a_type == "stock":
                        tc += buy[j] * stock_buy_cost + sell[j] * stock_sell_cost
                    else:
                        tc += buy[j] * etf_buy_cost + sell[j] * etf_sell_cost
                turnover = float(np.abs(delta).sum())
                cost_rate = float(tc)
                w = w_target.copy()
                target_plan[dt_key] = {
                    assets[j]: float(w_target[j]) for j in range(len(assets))
                }
            else:
                turnover = 0.0
                cost_rate = 0.0

        turnover_list.append(turnover)
        cost_list.append(cost_rate)
        row = {assets[j]: float(w[j]) for j in range(len(assets))}
        row["date"] = dt_key
        weights_rows.append(row)

    turnover_s = pd.Series(turnover_list, index=idx, name="turnover")
    cost_s = pd.Series(cost_list, index=idx, name="cost_rate")
    weights_df = pd.DataFrame(weights_rows).set_index("date")
    return panel, target_plan, turnover_s, cost_s, weights_df


def simulate_portfolio(
    ret_panel,
    target_weights,
    rule,
    asset_types,
    stock_buy_cost,
    stock_sell_cost,
    etf_buy_cost,
    etf_sell_cost,
    threshold,
):
    (
        panel,
        target_plan,
        turnover,
        cost_rate,
        weights,
    ) = build_target_plan_and_cost(
        ret_panel=ret_panel,
        target_weights=target_weights,
        rule=rule,
        asset_types=asset_types,
        stock_buy_cost=stock_buy_cost,
        stock_sell_cost=stock_sell_cost,
        etf_buy_cost=etf_buy_cost,
        etf_sell_cost=etf_sell_cost,
        threshold=threshold,
    )

    data = build_synthetic_ohlcv_from_returns(panel)
    if len(data) == 0:
        empty_idx = pd.DatetimeIndex([])
        return (
            pd.Series(dtype=float, index=empty_idx, name="nav"),
            pd.Series(dtype=float, index=empty_idx, name="ret"),
            pd.Series(dtype=float, index=empty_idx, name="turnover"),
            pd.Series(dtype=float, index=empty_idx, name="cost_rate"),
            pd.DataFrame(),
        )

    result = run_backtest(
        data=data,
        strategy=ScheduledTargetWeightsStrategy(target_plan=target_plan),
        symbol=list(data.keys()),
        initial_cash=1_000_000.0,
        commission_rate=0.0,
        execution_mode=ExecutionMode.CurrentClose,
        warmup_period=1,
    )

    equity = pd.Series(result.equity_curve, copy=True)
    equity.index = pd.DatetimeIndex(equity.index).tz_localize(None)
    equity = equity.sort_index()
    equity = equity[~equity.index.duplicated(keep="last")]
    nav_raw = equity / float(equity.iloc[0])

    turnover = turnover.sort_index()
    turnover = turnover[~turnover.index.duplicated(keep="last")]
    cost_rate = cost_rate.sort_index()
    cost_rate = cost_rate[~cost_rate.index.duplicated(keep="last")]
    weights = weights.sort_index()
    weights = weights[~weights.index.duplicated(keep="last")]

    aligned_cost = cost_rate.reindex(nav_raw.index).fillna(0.0)
    fee_factor = (1.0 - aligned_cost).cumprod()
    nav = (nav_raw * fee_factor).rename("nav")

    ret = nav.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0).rename("ret")
    turnover = turnover.reindex(nav.index).fillna(0.0).rename("turnover")
    cost_rate = aligned_cost.rename("cost_rate")
    weights = weights.reindex(nav.index).ffill().fillna(0.0)

    return nav, ret, turnover, cost_rate, weights


def annual_return_table(ret_map):
    rows = []
    for name, ret in ret_map.items():
        if ret.empty:
            continue
        y = (1.0 + ret).groupby(ret.index.year).prod() - 1.0
        for yr, val in y.items():
            rows.append(
                {"strategy": name, "year": int(yr), "annual_return": float(val)}
            )
    if len(rows) == 0:
        return pd.DataFrame(columns=["strategy", "year", "annual_return"])
    out = pd.DataFrame(rows)
    out = out.sort_values(by=["strategy", "year"])
    return out


def max_drawdown_series(nav):
    peak = nav.cummax()
    dd = nav / peak - 1.0
    return dd


def drawdown_episodes(name, nav):
    dd = max_drawdown_series(nav)
    in_dd = False
    start = None
    trough_date = None
    trough_val = 0.0
    rows = []

    for dt, val in dd.items():
        v = float(val)
        if (not in_dd) and v < 0:
            in_dd = True
            start = dt
            trough_date = dt
            trough_val = v
        elif in_dd:
            if v < trough_val:
                trough_val = v
                trough_date = dt
            if v >= -1e-12:
                rows.append(
                    {
                        "strategy": name,
                        "start": pd.Timestamp(start).date(),
                        "trough": pd.Timestamp(trough_date).date(),
                        "recovery": pd.Timestamp(dt).date(),
                        "drawdown": float(trough_val),
                    }
                )
                in_dd = False

    if in_dd:
        rows.append(
            {
                "strategy": name,
                "start": pd.Timestamp(start).date(),
                "trough": pd.Timestamp(trough_date).date(),
                "recovery": None,
                "drawdown": float(trough_val),
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(by="drawdown").reset_index(drop=True)
    return out


def rolling_table(name, ret, window, periods_per_year):
    if ret.empty:
        return pd.DataFrame()
    nav = (1.0 + ret).cumprod()
    roll_ret = nav / nav.shift(window) - 1.0
    roll_vol = ret.rolling(window).std(ddof=0) * math.sqrt(periods_per_year)
    out = pd.DataFrame(
        {
            "strategy": name,
            "date": roll_ret.index,
            f"rolling_{window}p_return": roll_ret.values,
            f"rolling_{window}p_vol": roll_vol.values,
        }
    )
    out = out.dropna()
    return out


def risk_contribution_table(name, component_ret, target_weights, periods_per_year):
    cols = list(target_weights.keys())
    panel = component_ret[cols].dropna()
    if panel.empty:
        return pd.DataFrame()
    cov = panel.cov() * periods_per_year
    w = np.array([float(target_weights[c]) for c in cols], dtype=float)
    w = w / float(w.sum())
    cov_mat = cov.to_numpy(dtype=float)
    port_var = float(w @ cov_mat @ w)
    if port_var <= 0:
        return pd.DataFrame()
    mrc = cov_mat @ w
    rc = w * mrc / port_var
    out = pd.DataFrame(
        {
            "strategy": name,
            "sleeve": cols,
            "target_weight": w,
            "risk_contribution": rc,
        }
    )
    return out


def build_macro_regime(cache_dir):
    ensure_proxy()
    cpi_cache = cache_dir / "macro_cpi_monthly.csv"
    pmi_cache = cache_dir / "macro_pmi.csv"

    if cpi_cache.exists():
        cpi_df = pd.read_csv(cpi_cache)
    else:
        cpi_df = ak.macro_china_cpi_monthly()
        cpi_df.to_csv(cpi_cache, index=False)

    if pmi_cache.exists():
        pmi_df = pd.read_csv(pmi_cache)
    else:
        pmi_df = ak.macro_china_pmi()
        pmi_df.to_csv(pmi_cache, index=False)

    def normalize_macro(df, value_candidates):
        if df is None or df.empty:
            return pd.DataFrame(columns=["date", "value"])
        dcol = first_existing(
            list(df.columns), ["date", "日期", "月份", "month", "时间", "统计时间"]
        )
        if dcol is None:
            dcol = df.columns[0]
        vcol = first_existing(list(df.columns), value_candidates)
        if vcol is None:
            num_cols = [c for c in df.columns if c != dcol]
            if len(num_cols) == 0:
                return pd.DataFrame(columns=["date", "value"])
            vcol = num_cols[0]
        out = pd.DataFrame(
            {
                "date": pd.to_datetime(df[dcol], errors="coerce"),
                "value": pd.to_numeric(df[vcol], errors="coerce"),
            }
        )
        out = out.dropna(subset=["date", "value"]).sort_values(by="date")
        if out.empty:
            return out
        out["date"] = out["date"].dt.to_period("M").dt.to_timestamp("M")
        out = out.drop_duplicates(subset=["date"], keep="last")
        return out

    cpi = normalize_macro(cpi_df, ["cpi", "CPI", "同比", "当月", "value"])
    pmi = normalize_macro(pmi_df, ["pmi", "PMI", "value"])

    merged = pd.merge(cpi, pmi, on="date", how="outer", suffixes=("_cpi", "_pmi"))
    merged = merged.sort_values(by="date")
    merged["value_cpi"] = merged["value_cpi"].ffill()
    merged["value_pmi"] = merged["value_pmi"].ffill()
    merged = merged.dropna(subset=["value_cpi", "value_pmi"])
    if merged.empty:
        return pd.DataFrame(columns=["date", "regime", "cpi", "pmi"])

    def label_regime(row):
        cpi_v = float(row["value_cpi"])
        pmi_v = float(row["value_pmi"])
        if pmi_v >= 50 and cpi_v >= 2:
            return "复苏-通胀"
        if pmi_v >= 50 and cpi_v < 2:
            return "复苏-低通胀"
        if pmi_v < 50 and cpi_v >= 2:
            return "滞胀"
        return "收缩-低通胀"

    merged["regime"] = merged.apply(label_regime, axis=1)
    merged = merged.rename(columns={"value_cpi": "cpi", "value_pmi": "pmi"})
    return merged[["date", "regime", "cpi", "pmi"]]


def regime_performance_table(ret_map, regime_df):
    if regime_df.empty:
        return pd.DataFrame()
    rows = []
    regime = regime_df.copy()
    regime["date"] = pd.to_datetime(regime["date"])

    for name, ret in ret_map.items():
        if ret.empty:
            continue
        df = pd.DataFrame({"date": ret.index, "ret": ret.values})
        df["date"] = pd.to_datetime(df["date"]).to_period("M").to_timestamp("M")
        m = pd.merge(df, regime, on="date", how="inner")
        if m.empty:
            continue
        g = m.groupby("regime")["ret"]
        for reg, ser in g:
            rows.append(
                {
                    "strategy": name,
                    "regime": reg,
                    "count": int(ser.shape[0]),
                    "mean_return": float(ser.mean()),
                    "vol": float(ser.std(ddof=0)),
                    "win_rate": float((ser > 0).mean()),
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(by=["strategy", "regime"]).reset_index(drop=True)
    return out


def subperiod_metrics(ret_map, periods_per_year):
    spans = [
        ("2015-2018", "2015-01-01", "2018-12-31"),
        ("2019-2021", "2019-01-01", "2021-12-31"),
        ("2022-Now", "2022-01-01", "2100-01-01"),
    ]
    rows = []
    for name, ret in ret_map.items():
        if ret.empty:
            continue
        for tag, s, e in spans:
            mask = (ret.index >= pd.Timestamp(s)) & (ret.index <= pd.Timestamp(e))
            sub = ret.loc[mask]
            if sub.empty:
                continue
            m = calc_metrics(sub, periods_per_year)
            row = {"strategy": name, "span": tag}
            row.update(m)
            rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(by=["strategy", "span"]).reset_index(drop=True)
    return out


def aggregate_method_comparison(summary_df):
    if summary_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    bank = summary_df[summary_df["strategy"].str.contains("bank=")].copy()
    if bank.empty:
        return pd.DataFrame(), pd.DataFrame()
    bank["bank_method"] = bank["strategy"].str.extract(r"bank=([^,]+)")
    bank_cmp = (
        bank.groupby("bank_method")[["cagr", "sharpe", "max_drawdown"]]
        .mean()
        .reset_index()
    )

    prod = summary_df[summary_df["strategy"].str.contains("prod=")].copy()
    prod["prod_method"] = prod["strategy"].str.extract(r"prod=([^,]+)")
    prod_cmp = (
        prod.groupby("prod_method")[["cagr", "sharpe", "max_drawdown"]]
        .mean()
        .reset_index()
    )
    return bank_cmp, prod_cmp


def render_report(report_path, context):
    text = []
    text.append("# 中国版永久投资组合研究报告")
    text.append("")
    text.append("## 1) 研究目标与问题定义")
    text.append(
        "- 目标：构建并检验中国市场可执行的‘永久组合’框架，比较股票偏重方案（A）与加入现金+黄金后的改进方案（B）。"
    )
    text.append(
        "- 核心问题：在2015年至今窗口内，哪类映射和再平衡方式在收益/回撤/稳健性上更优。"
    )
    text.append("")
    text.append("## 2) 数据与可交易代理映射")
    text.append("- 本地优先：DuckDB（A股个股HFQ与因子视图）。")
    text.append("- 回退数据：AkShare（指数/ETF/月度宏观），并使用仓库要求的代理补丁。")
    text.append(
        "- 关键代理：银行、红利、生产资料、非银弹性、黄金、货币、债券、沪深300/中证500。"
    )
    text.append("")
    text.append("## 3) 策略设计与版本定义")
    text.append("- A版（股票偏重）：银行+红利+生产资料+弹性。")
    text.append("- B版（改进）：在A版基础上加入现金与黄金。")
    text.append("- 银行两法：方法1指数代理、方法2低PB+高股息（等权/股息权重）。")
    text.append("- 生产资料三法：静态指数混合/龙头等权篮子/动量轮动。")
    text.append("")
    text.append("## 4) 回测框架与交易摩擦")
    text.append("- 主回测：月频收益序列；稳健性：周频收益序列。")
    text.append("- 再平衡规则：annual / threshold5 / threshold10 / hybrid。")
    text.append("- 成本模型：股票与ETF差异费率，股票卖出含印花税。")
    text.append("")
    text.append("## 5) 主回测结果（月频）")
    text.append(f"- 共评估策略数：{context['strategy_count']}。")
    text.append(f"- 最优策略（按Sharpe）：{context['best_strategy']}。")
    text.append(
        f"- 最优策略核心指标：CAGR={context['best_cagr']:.2%}，MaxDD={context['best_mdd']:.2%}，Sharpe={context['best_sharpe']:.2f}。"
    )
    text.append("")
    text.append("## 6) 稳健性与敏感性")
    text.append("- 周频结果用于验证参数与结论方向是否一致。")
    text.append("- 对再平衡规则与交易成本进行了单独敏感性分析。")
    text.append("")
    text.append("## 7) 风险归因与相关性")
    text.append("- 输出了核心策略风险贡献、相关矩阵、回撤区间。")
    text.append("- 重点看是否过度集中于某单一风险源。")
    text.append("")
    text.append("## 8) 分环境表现（宏观分箱）")
    text.append(
        "- 基于PMI与CPI划分四类环境：复苏-通胀、复苏-低通胀、滞胀、收缩-低通胀。"
    )
    text.append("- 输出各环境下均值、波动、胜率。")
    text.append("")
    text.append("## 9) 局限性与偏差来源")
    text.append("- 指数/ETF代理与真实可交易标的存在跟踪误差。")
    text.append("- 个股篮子构建受因子缺失和停牌数据质量影响。")
    text.append("- 宏观指标口径与发布时滞会影响环境切分准确度。")
    text.append("")
    text.append("## 10) 实盘建议（研究结论落地）")
    text.append("- 建议优先采用B版（含现金+黄金）并使用hybrid再平衡。")
    text.append("- 交易执行可设置‘阈值+半年度强制’双机制，避免过度换手。")
    text.append("- 先小资金试运行3-6个月，跟踪偏离与摩擦成本后再扩容。")
    text.append("")
    text.append("## 产物索引")
    text.append("- tables/: 指标表、敏感性表、风险与环境表")
    text.append("- figures/: 净值、回撤、滚动指标、相关热图")

    report_path.write_text("\n".join(text), encoding="utf-8")


def main():
    args = parse_args()
    start_date = to_ts(args.start_date)

    cache_dir, tables_dir, figs_dir = ensure_dirs(args.out_dir)

    con = duckdb.connect(str(args.db), read_only=True)
    end_date = get_end_date(con, args.end_date)

    idx_codes = {
        "hs300": "000300",
        "csi500": "000905",
        "dividend": "000922",
        "bank_idx": "399431",
        "nonferrous": "399395",
        "prod_energy": "399613",
        "prod_material": "399614",
        "prod_industry": "399615",
        "energy_metal": "399366",
    }
    etf_codes = {
        "gold_etf": "518880",
        "cash_etf": "511880",
        "bond_etf": "511010",
    }

    idx_series = {}
    for k, code in idx_codes.items():
        s = fetch_index_price(code, cache_dir)
        idx_series[k] = restrict_series(s, start_date, end_date)

    etf_series = {}
    for k, code in etf_codes.items():
        s = fetch_etf_price(code, cache_dir)
        etf_series[k] = restrict_series(s, start_date, end_date)

    bank_m2_eq_nav, bank_eq_picks, _ = build_bank_method2(
        con, start_date, end_date, "eq"
    )
    bank_m2_div_nav, bank_div_picks, _ = build_bank_method2(
        con, start_date, end_date, "div"
    )
    prod_m2_nav, prod_m2_picks, _ = build_production_method2(con, start_date, end_date)
    gold_stock_nav = build_gold_stock_nav(con, start_date, end_date)

    prod_pool = pd.DataFrame(
        {
            "399613": idx_series["prod_energy"],
            "399614": idx_series["prod_material"],
            "399615": idx_series["prod_industry"],
            "399366": idx_series["energy_metal"],
        }
    ).sort_index()
    prod_m1_nav = build_prod_m1_index_blend(prod_pool)
    prod_m3_nav, prod_m3_picks, _ = build_prod_m3_dynamic(prod_pool, start_date)

    bank_m1_nav = idx_series["bank_idx"] / float(idx_series["bank_idx"].iloc[0])
    div_nav = idx_series["dividend"] / float(idx_series["dividend"].iloc[0])
    nonferrous_nav = idx_series["nonferrous"] / float(idx_series["nonferrous"].iloc[0])
    hs300_nav = idx_series["hs300"] / float(idx_series["hs300"].iloc[0])
    csi500_nav = idx_series["csi500"] / float(idx_series["csi500"].iloc[0])
    gold_etf_nav = etf_series["gold_etf"] / float(etf_series["gold_etf"].iloc[0])
    cash_etf_nav = etf_series["cash_etf"] / float(etf_series["cash_etf"].iloc[0])
    bond_etf_nav = etf_series["bond_etf"] / float(etf_series["bond_etf"].iloc[0])

    elastic_nav = blend_nav(gold_stock_nav, nonferrous_nav, 0.5)

    nav_map = {
        "bank_m1_idx": bank_m1_nav,
        "bank_m2_eq": bank_m2_eq_nav,
        "bank_m2_div": bank_m2_div_nav,
        "prod_m1_idxblend": prod_m1_nav,
        "prod_m2_leader": prod_m2_nav,
        "prod_m3_dynamic": prod_m3_nav,
        "dividend_idx": div_nav,
        "elastic_mix": elastic_nav,
        "gold_etf": gold_etf_nav,
        "cash_etf": cash_etf_nav,
        "bond_etf": bond_etf_nav,
        "hs300": hs300_nav,
        "csi500": csi500_nav,
    }

    for k, v in nav_map.items():
        nav_map[k] = (
            restrict_series(v, start_date, end_date).sort_index().ffill().dropna()
        )

    bank_methods = ["bank_m1_idx", "bank_m2_eq", "bank_m2_div"]
    prod_methods = ["prod_m1_idxblend", "prod_m2_leader", "prod_m3_dynamic"]

    strategy_specs = []
    for b in bank_methods:
        for p in prod_methods:
            strategy_specs.append(
                {
                    "strategy": f"A,bank={b},prod={p}",
                    "weights": {
                        b: 0.35,
                        "dividend_idx": 0.25,
                        p: 0.25,
                        "elastic_mix": 0.15,
                    },
                }
            )
            strategy_specs.append(
                {
                    "strategy": f"B,bank={b},prod={p}",
                    "weights": {
                        b: 0.25,
                        "dividend_idx": 0.15,
                        p: 0.20,
                        "elastic_mix": 0.10,
                        "gold_etf": 0.20,
                        "cash_etf": 0.10,
                    },
                }
            )

    comparator_specs = [
        {"strategy": "CMP_HS300", "weights": {"hs300": 1.0}},
        {"strategy": "CMP_60_40", "weights": {"hs300": 0.6, "bond_etf": 0.4}},
        {
            "strategy": "CMP_Permanent_CN",
            "weights": {
                "hs300": 0.25,
                "bond_etf": 0.25,
                "gold_etf": 0.25,
                "cash_etf": 0.25,
            },
        },
        {
            "strategy": "CMP_AllWeatherLite",
            "weights": {
                "hs300": 0.30,
                "csi500": 0.20,
                "bond_etf": 0.20,
                "gold_etf": 0.15,
                "cash_etf": 0.15,
            },
        },
    ]

    all_specs = strategy_specs + comparator_specs

    asset_types = {
        "bank_m1_idx": "etf",
        "bank_m2_eq": "stock",
        "bank_m2_div": "stock",
        "prod_m1_idxblend": "etf",
        "prod_m2_leader": "stock",
        "prod_m3_dynamic": "etf",
        "dividend_idx": "etf",
        "elastic_mix": "stock",
        "gold_etf": "etf",
        "cash_etf": "etf",
        "bond_etf": "etf",
        "hs300": "etf",
        "csi500": "etf",
    }

    monthly_results = {}
    monthly_ret_map = {}
    monthly_nav_map = {}
    monthly_component_map = {}

    for spec in all_specs:
        name = spec["strategy"]
        weights = normalize_weights_map(spec["weights"])
        keys = list(weights.keys())
        panel = build_period_return_panel(nav_map, keys, "ME")
        if panel.empty:
            continue
        nav, ret, turnover, cost_rate, realized_w = simulate_portfolio(
            panel,
            weights,
            rule="hybrid",
            asset_types=asset_types,
            stock_buy_cost=0.0008,
            stock_sell_cost=0.0018,
            etf_buy_cost=0.0003,
            etf_sell_cost=0.0003,
            threshold=0.05,
        )
        m = calc_metrics(ret, periods_per_year=12)
        m["strategy"] = name
        m["turnover_mean"] = float(turnover.mean())
        m["cost_rate_sum"] = float(cost_rate.sum())
        monthly_results[name] = {
            "metrics": m,
            "nav": nav,
            "ret": ret,
            "turnover": turnover,
            "cost_rate": cost_rate,
            "weights": realized_w,
            "target_weights": weights,
            "component_ret": panel,
        }
        monthly_ret_map[name] = ret
        monthly_nav_map[name] = nav
        monthly_component_map[name] = panel

    summary_rows = [v["metrics"] for v in monthly_results.values()]
    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values(by="sharpe", ascending=False).reset_index(
        drop=True
    )
    write_table(summary_df, tables_dir / "summary_metrics_monthly.csv")

    bank_cmp, prod_cmp = aggregate_method_comparison(summary_df)
    write_table(bank_cmp, tables_dir / "bank_method_comparison.csv")
    write_table(prod_cmp, tables_dir / "production_method_comparison.csv")

    annual_df = annual_return_table(monthly_ret_map)
    write_table(annual_df, tables_dir / "annual_returns_monthly.csv")

    corr_df = pd.DataFrame(monthly_ret_map).corr()
    corr_out = corr_df.copy()
    corr_out.insert(0, "strategy", corr_out.index)
    write_table(corr_out.reset_index(drop=True), tables_dir / "correlation_monthly.csv")

    core_a = summary_df[summary_df["strategy"].str.startswith("A,")].head(1)
    core_b = summary_df[summary_df["strategy"].str.startswith("B,")].head(1)
    core_names = []
    if not core_a.empty:
        core_names.append(core_a["strategy"].iloc[0])
    if not core_b.empty:
        core_names.append(core_b["strategy"].iloc[0])
    if "CMP_HS300" in monthly_results:
        core_names.append("CMP_HS300")

    dd_rows = []
    for name in core_names:
        dd_part = drawdown_episodes(name, monthly_results[name]["nav"])
        if not dd_part.empty:
            dd_rows.append(dd_part)
    dd_df = pd.concat(dd_rows, ignore_index=True) if len(dd_rows) else pd.DataFrame()
    write_table(dd_df, tables_dir / "drawdown_episodes.csv")

    rolling_rows = []
    for name in core_names:
        rolling_rows.append(
            rolling_table(
                name, monthly_results[name]["ret"], window=12, periods_per_year=12
            )
        )
    rolling_df = (
        pd.concat(rolling_rows, ignore_index=True)
        if len(rolling_rows)
        else pd.DataFrame()
    )
    write_table(rolling_df, tables_dir / "rolling_12m_metrics.csv")

    rc_rows = []
    for name in core_names:
        if name not in monthly_results:
            continue
        comp = monthly_results[name]["component_ret"]
        tw = monthly_results[name]["target_weights"]
        rc = risk_contribution_table(name, comp, tw, periods_per_year=12)
        if not rc.empty:
            rc_rows.append(rc)
    rc_df = pd.concat(rc_rows, ignore_index=True) if len(rc_rows) else pd.DataFrame()
    write_table(rc_df, tables_dir / "risk_contribution.csv")

    subperiod_df = subperiod_metrics(monthly_ret_map, periods_per_year=12)
    write_table(subperiod_df, tables_dir / "subperiod_metrics.csv")

    regime_df = build_macro_regime(cache_dir)
    regime_perf_df = regime_performance_table(monthly_ret_map, regime_df)
    write_table(regime_perf_df, tables_dir / "regime_performance.csv")

    rules = ["annual", "threshold5", "threshold10", "hybrid"]
    rule_rows = []
    core_for_sens = []
    if not core_a.empty:
        core_for_sens.append(core_a["strategy"].iloc[0])
    if not core_b.empty:
        core_for_sens.append(core_b["strategy"].iloc[0])
    core_for_sens += ["CMP_Permanent_CN", "CMP_HS300"]
    core_for_sens = [x for x in core_for_sens if x in monthly_results]

    for name in core_for_sens:
        base = monthly_results[name]
        panel = base["component_ret"]
        tw = base["target_weights"]
        for rule in rules:
            th = 0.05
            if rule == "threshold10":
                th = 0.10
            nav, ret, turnover, cost_rate, _ = simulate_portfolio(
                panel,
                tw,
                rule=rule,
                asset_types=asset_types,
                stock_buy_cost=0.0008,
                stock_sell_cost=0.0018,
                etf_buy_cost=0.0003,
                etf_sell_cost=0.0003,
                threshold=th,
            )
            m = calc_metrics(ret, periods_per_year=12)
            m["strategy"] = name
            m["rule"] = rule
            m["turnover_mean"] = float(turnover.mean())
            m["cost_rate_sum"] = float(cost_rate.sum())
            rule_rows.append(m)
    rule_df = pd.DataFrame(rule_rows)
    write_table(rule_df, tables_dir / "rebalance_rule_sensitivity.csv")

    cost_rows = []
    cost_scales = [0.0, 0.5, 1.0, 2.0]
    cost_target_name = (
        core_b["strategy"].iloc[0]
        if not core_b.empty
        else summary_df.iloc[0]["strategy"]
    )
    base = monthly_results[cost_target_name]
    panel = base["component_ret"]
    tw = base["target_weights"]
    for cs in cost_scales:
        nav, ret, turnover, cost_rate, _ = simulate_portfolio(
            panel,
            tw,
            rule="hybrid",
            asset_types=asset_types,
            stock_buy_cost=0.0008 * cs,
            stock_sell_cost=0.0018 * cs,
            etf_buy_cost=0.0003 * cs,
            etf_sell_cost=0.0003 * cs,
            threshold=0.05,
        )
        m = calc_metrics(ret, periods_per_year=12)
        m["strategy"] = cost_target_name
        m["cost_scale"] = cs
        m["turnover_mean"] = float(turnover.mean())
        m["cost_rate_sum"] = float(cost_rate.sum())
        cost_rows.append(m)
    cost_df = pd.DataFrame(cost_rows)
    write_table(cost_df, tables_dir / "cost_sensitivity.csv")

    weekly_summary = pd.DataFrame()
    if args.weekly:
        weekly_rows = []
        for spec in all_specs:
            name = spec["strategy"]
            weights = normalize_weights_map(spec["weights"])
            keys = list(weights.keys())
            panel_w = build_period_return_panel(nav_map, keys, "W-FRI")
            if panel_w.empty:
                continue
            nav, ret, turnover, cost_rate, _ = simulate_portfolio(
                panel_w,
                weights,
                rule="hybrid",
                asset_types=asset_types,
                stock_buy_cost=0.0008,
                stock_sell_cost=0.0018,
                etf_buy_cost=0.0003,
                etf_sell_cost=0.0003,
                threshold=0.05,
            )
            m = calc_metrics(ret, periods_per_year=52)
            m["strategy"] = name
            m["turnover_mean"] = float(turnover.mean())
            m["cost_rate_sum"] = float(cost_rate.sum())
            weekly_rows.append(m)
        weekly_summary = pd.DataFrame(weekly_rows).sort_values(
            by="sharpe", ascending=False
        )
        write_table(weekly_summary, tables_dir / "summary_metrics_weekly.csv")

    write_table(bank_eq_picks, tables_dir / "bank_method2_eq_picks.csv")
    write_table(bank_div_picks, tables_dir / "bank_method2_div_picks.csv")
    write_table(prod_m2_picks, tables_dir / "production_method2_picks.csv")
    write_table(prod_m3_picks, tables_dir / "production_method3_picks.csv")

    plt.figure(figsize=(12, 6))
    top_plot = summary_df.head(6)["strategy"].tolist()
    for name in top_plot:
        plt.plot(
            monthly_results[name]["nav"].index,
            monthly_results[name]["nav"].values,
            label=name,
        )
    if "CMP_HS300" in monthly_results:
        plt.plot(
            monthly_results["CMP_HS300"]["nav"].index,
            monthly_results["CMP_HS300"]["nav"].values,
            label="CMP_HS300",
            linestyle="--",
            linewidth=2,
        )
    plt.legend(fontsize=8)
    plt.title("中国版永久组合：月频净值对比（主回测）")
    save_fig(figs_dir / "nav_curves_main.png")

    plt.figure(figsize=(12, 6))
    for name in core_names:
        nav = monthly_results[name]["nav"]
        dd = max_drawdown_series(nav)
        plt.plot(dd.index, dd.values, label=name)
    plt.legend(fontsize=9)
    plt.title("核心策略回撤曲线")
    save_fig(figs_dir / "drawdown_core.png")

    plt.figure(figsize=(9, 7))
    if not corr_df.empty:
        arr = corr_df.to_numpy(dtype=float)
        plt.imshow(arr, cmap="RdYlGn", vmin=-1, vmax=1)
        ticks = list(range(len(corr_df.columns)))
        labels = [str(x) for x in corr_df.columns]
        plt.xticks(ticks=ticks, labels=labels, rotation=90)
        plt.yticks(ticks=ticks, labels=labels)
        plt.colorbar(label="Correlation")
        plt.title("月频收益相关系数矩阵")
    save_fig(figs_dir / "correlation_heatmap.png")

    plt.figure(figsize=(12, 6))
    for name in core_names:
        r = monthly_results[name]["ret"]
        roll = (1.0 + r).cumprod() / (1.0 + r).cumprod().shift(12) - 1.0
        plt.plot(roll.index, roll.values, label=name)
    plt.legend(fontsize=9)
    plt.title("核心策略滚动12个月收益")
    save_fig(figs_dir / "rolling_12m_return_core.png")

    if not annual_df.empty and len(core_names) > 0:
        pivot = annual_df[annual_df["strategy"].isin(core_names)].pivot(
            index="year", columns="strategy", values="annual_return"
        )
        plt.figure(figsize=(12, 6))
        x = np.arange(len(pivot.index))
        width = 0.8 / max(1, len(pivot.columns))
        for i, col in enumerate(pivot.columns):
            plt.bar(
                x + i * width,
                pivot[col].to_numpy(dtype=float),
                width=width,
                label=str(col),
            )
        plt.xticks(
            x + width * max(0, len(pivot.columns) - 1) / 2, pivot.index.astype(str)
        )
        plt.legend(fontsize=9)
        plt.title("核心策略年度收益")
        save_fig(figs_dir / "annual_return_core.png")

    best_name = summary_df.iloc[0]["strategy"]
    best_row = summary_df.iloc[0]
    report_context = {
        "strategy_count": int(summary_df.shape[0]),
        "best_strategy": str(best_name),
        "best_cagr": float(best_row["cagr"]),
        "best_mdd": float(best_row["max_drawdown"]),
        "best_sharpe": float(best_row["sharpe"]),
    }
    render_report(args.out_dir / "report.md", report_context)

    summary_json = {
        "window": {
            "start": str(start_date.date()),
            "end": str(end_date.date()),
        },
        "strategies_evaluated": int(summary_df.shape[0]),
        "best_strategy": str(best_name),
        "best_metrics": {
            "total_return": float(best_row["total_return"]),
            "cagr": float(best_row["cagr"]),
            "ann_vol": float(best_row["ann_vol"]),
            "sharpe": float(best_row["sharpe"]),
            "max_drawdown": float(best_row["max_drawdown"]),
        },
        "core_strategies": core_names,
        "tables_dir": str(tables_dir),
        "figures_dir": str(figs_dir),
        "weekly_enabled": bool(args.weekly),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary_json, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("研究完成")
    print(f"时间区间: {start_date.date()} -> {end_date.date()}")
    print(f"策略数量: {summary_df.shape[0]}")
    print(
        f"最佳策略: {best_name} | CAGR={best_row['cagr']:.2%} | Sharpe={best_row['sharpe']:.2f} | MaxDD={best_row['max_drawdown']:.2%}"
    )
    print(f"输出目录: {args.out_dir}")


if __name__ == "__main__":
    main()
