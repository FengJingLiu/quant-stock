from __future__ import annotations

import argparse
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, cast

import akshare as ak
import duckdb
import numpy as np
import pandas as pd
from akquant import ExecutionMode, Strategy, run_backtest

BANK_INDUSTRY = "银行"
METAL_BOARD_CODE = "BK0478"
METAL_BOARD_NAME = "有色金属"
OTHER_BOARD_KEYWORDS_DEFAULT = ("科技", "消费", "化工")
from src.config import AKSHARE_PROXY_HOST as DEFAULT_AKSHARE_PROXY_HOST
from src.data_clients import ensure_akshare_proxy_patch
from src.config import AKSHARE_PROXY_TOKEN as DEFAULT_AKSHARE_PROXY_TOKEN
SLEEVE_WEIGHTS = {"bank": 0.50, "metal": 0.25, "other": 0.25}
BANK_REBALANCE_DIVIDEND_FLOOR = 5.0
NONBANK_MIN_CIRC_MV_10K = 2_000_000.0

TRIGGER_BANK_INITIAL_BUILD = "银行初始建仓"
TRIGGER_BANK_DIVIDEND_REBALANCE = "银行股息率<5%调仓"
TRIGGER_BANK_WEIGHT_TRIM = "银行仓位>60%回调至50%"
TRIGGER_MONTHLY_SELECTION_ENTRY = "非银初始建仓"
TRIGGER_MONTHLY_REBALANCE_EXIT = "非银调仓卖出"
TRIGGER_TAKE_PROFIT = "止盈30%"
TRIGGER_STOP_LOSS = "止损20%"
TRIGGER_TARGET_ADJUSTMENT_ENTRY = "目标仓位调整买入"
TRIGGER_TARGET_ADJUSTMENT_EXIT = "目标仓位调整卖出或期末平仓"

OUTPUT_COLUMN_NAME_MAPS: dict[str, dict[str, str]] = {
    "picks": {
        "rebalance_date": "调仓日",
        "signal_date": "信号日",
        "sleeve": "策略分仓",
        "symbol": "股票代码",
        "name": "股票名称",
        "industry": "行业",
        "close": "收盘价",
        "close_adj": "复权收盘价",
        "ma120": "120日均线",
        "ma120_adj": "复权120日均线",
        "pe_ttm": "市盈率TTM",
        "pb": "市净率",
        "dividend_yield": "股息率",
        "amount_ma20": "20日平均成交额",
        "circ_mv_10k": "流通市值(万元)",
        "hfq_factor": "后复权因子",
        "qfq_factor": "前复权因子",
    },
    "trace": {
        "date": "日期",
        "bank_count": "银行持仓数",
        "metal_count": "小金属持仓数",
        "other_count": "其他持仓数",
        "bank_weight_target": "银行目标仓位",
        "metal_weight": "小金属仓位",
        "other_weight": "其他仓位",
        "cash_weight": "现金仓位",
    },
    "targets": {
        "date": "日期",
        "symbol": "股票代码",
        "target_weight": "目标仓位",
    },
    "metrics": {
        "metric": "指标",
        "value": "数值",
        "index": "索引",
    },
    "trades": {
        "symbol_code": "股票代码",
        "name_cn": "股票名称",
        "symbol": "股票代码|名称",
        "entry_time": "开仓时间",
        "exit_time": "平仓时间",
        "entry_price": "开仓价",
        "exit_price": "平仓价",
        "quantity": "成交数量",
        "side": "方向",
        "pnl": "盈亏",
        "net_pnl": "净盈亏",
        "return_pct": "收益率%",
        "commission": "手续费",
        "duration_bars": "持有K线数",
        "duration": "持有时长",
        "mae": "最大不利变动%",
        "mfe": "最大有利变动%",
        "entry_tag": "开仓标签",
        "exit_tag": "平仓标签",
        "entry_portfolio_value": "开仓组合净值",
        "max_drawdown_pct": "最大回撤%",
        "entry_trigger": "开仓触发条件",
        "exit_trigger": "平仓触发条件",
    },
}


@dataclass(frozen=True)
class RebalanceSelection:
    rebalance_date: pd.Timestamp
    signal_date: pd.Timestamp
    bank_symbols: list[str]
    metal_symbols: list[str]
    other_symbols: list[str]


@dataclass(frozen=True)
class BoardUniverses:
    bank_symbols: set[str]
    metal_symbols: set[str]
    other_symbols: set[str]
    symbol_name_map: dict[str, str]
    bank_board_code: str
    bank_board_name: str
    metal_board_code: str
    metal_board_name: str
    other_board_codes: list[str]
    other_board_names: list[str]


def normalize_ts(value: Any) -> pd.Timestamp:
    if isinstance(value, (int, np.integer)):
        ts = (
            pd.to_datetime(int(value), unit="ns", utc=True)
            .tz_convert("Asia/Shanghai")
            .tz_localize(None)
        )
    else:
        ts = pd.Timestamp(value)
        if pd.isna(ts):
            raise ValueError(f"Invalid timestamp: {value}")
        if ts.tzinfo is not None:
            ts = ts.tz_convert("Asia/Shanghai").tz_localize(None)
    ts_checked = cast(pd.Timestamp, ts)
    return cast(
        pd.Timestamp,
        pd.Timestamp(
            year=int(ts_checked.year),
            month=int(ts_checked.month),
            day=int(ts_checked.day),
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="A-share 50:25:25 bank/metal/other allocation backtest"
    )
    parser.add_argument("--db", type=Path, default=Path("data/duckdb/stock.duckdb"))
    parser.add_argument("--start-date", type=str, default="2010-01-01")
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--initial-cash", type=float, default=200_000.0)
    parser.add_argument("--commission-rate", type=float, default=0.0003)
    parser.add_argument("--bank-top-n", type=int, default=2)
    parser.add_argument("--metal-top-n", type=int, default=1)
    parser.add_argument("--other-top-n", type=int, default=1)
    parser.add_argument(
        "--akshare-proxy-host",
        type=str,
        default=DEFAULT_AKSHARE_PROXY_HOST,
    )
    parser.add_argument(
        "--akshare-token",
        type=str,
        default=DEFAULT_AKSHARE_PROXY_TOKEN,
    )
    parser.add_argument("--akshare-proxy-retry", type=int, default=30)
    parser.add_argument("--board-tries", type=int, default=4)
    parser.add_argument("--board-sleep", type=float, default=0.8)
    parser.add_argument(
        "--other-board-keywords",
        type=str,
        default=",".join(OTHER_BOARD_KEYWORDS_DEFAULT),
        help="Comma-separated board-name keywords for other sleeve, e.g. 科技,消费,化工",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/backtest_bank_metal_balance_502525"),
    )
    return parser.parse_args()


def resolve_end_date(
    con: duckdb.DuckDBPyConnection, end_date: str | None
) -> pd.Timestamp:
    if end_date:
        return normalize_ts(end_date)
    row = con.execute("SELECT MAX(date) FROM v_bar_daily_raw").fetchone()
    if row is None or row[0] is None:
        raise RuntimeError("v_bar_daily_raw has no data")
    return normalize_ts(row[0])


def load_trading_dates(
    con: duckdb.DuckDBPyConnection, start_date: pd.Timestamp, end_date: pd.Timestamp
) -> list[pd.Timestamp]:
    df = con.execute(
        """
        SELECT DISTINCT date
        FROM v_bar_daily_raw
        WHERE date BETWEEN ? AND ?
        ORDER BY date
        """,
        [start_date.date(), end_date.date()],
    ).df()
    if df.empty:
        return []
    return [normalize_ts(x) for x in df["date"].tolist()]


def load_previous_trade_date(
    con: duckdb.DuckDBPyConnection, before_date: pd.Timestamp
) -> pd.Timestamp | None:
    row = con.execute(
        "SELECT MAX(date) FROM v_bar_daily_raw WHERE date < ?",
        [before_date.date()],
    ).fetchone()
    if row is None or row[0] is None:
        return None
    return normalize_ts(row[0])


def first_trade_day_per_month(trading_dates: list[pd.Timestamp]) -> list[pd.Timestamp]:
    if not trading_dates:
        return []
    frame = pd.DataFrame({"date": pd.DatetimeIndex(trading_dates)})
    frame["month_key"] = frame["date"].dt.to_period("M")
    firsts = frame.sort_values("date").drop_duplicates("month_key", keep="first")
    return [normalize_ts(x) for x in firsts["date"].tolist()]


def previous_trade_date(
    trading_dates: list[pd.Timestamp], rebalance_date: pd.Timestamp
) -> pd.Timestamp | None:
    prev = [d for d in trading_dates if d < rebalance_date]
    if not prev:
        return None
    return prev[-1]


def load_signal_snapshot(
    con: duckdb.DuckDBPyConnection,
    signal_dates: list[pd.Timestamp],
    start_hist: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    if not signal_dates:
        return pd.DataFrame()

    signals_df = pd.DataFrame(
        {"signal_date": [d.date().isoformat() for d in signal_dates]}
    )
    con.register("tmp_signal_dates", signals_df)
    try:
        df = con.execute(
            """
            WITH base AS (
              SELECT
                b.date,
                b.symbol,
                b.close,
                b.close * COALESCE(f.hfq_factor, 1.0) AS close_adj,
                b.amount,
                i.pe_ttm,
                i.pb,
                COALESCE(i.dividend_yield_ttm, i.dividend_yield) AS dividend_yield,
                i.dividend_yield_ttm,
                i.dividend_yield,
                i.circ_mv_10k,
                i.total_mv_10k,
                d.name,
                d.industry,
                COALESCE(d.is_delisted, 0) AS is_delisted,
                f.hfq_factor,
                f.qfq_factor
              FROM v_bar_daily_raw b
              LEFT JOIN v_indicator_daily i USING (symbol, date)
              LEFT JOIN v_dim_symbol d USING (symbol)
              LEFT JOIN v_adj_factor_daily f USING (symbol, date)
              WHERE b.date BETWEEN ? AND ?
            ),
            enriched AS (
              SELECT
                *,
                AVG(close) OVER (
                  PARTITION BY symbol
                  ORDER BY date
                  ROWS BETWEEN 119 PRECEDING AND CURRENT ROW
                ) AS ma120,
                AVG(close_adj) OVER (
                  PARTITION BY symbol
                  ORDER BY date
                  ROWS BETWEEN 119 PRECEDING AND CURRENT ROW
                ) AS ma120_adj,
                AVG(amount) OVER (
                  PARTITION BY symbol
                  ORDER BY date
                  ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
                ) AS amount_ma20
              FROM base
            )
            SELECT
              e.date,
              e.symbol,
              e.close,
              e.close_adj,
              e.amount,
              e.pe_ttm,
              e.pb,
              e.dividend_yield,
              e.dividend_yield_ttm,
              e.dividend_yield AS dividend_yield_raw,
              e.ma120,
              e.ma120_adj,
              e.amount_ma20,
              e.circ_mv_10k,
              e.total_mv_10k,
              e.name,
              e.industry,
              e.is_delisted,
              e.hfq_factor,
              e.qfq_factor
            FROM enriched e
            INNER JOIN tmp_signal_dates s
              ON e.date = CAST(s.signal_date AS DATE)
            WHERE
              COALESCE(e.is_delisted, 0) = 0
              AND UPPER(COALESCE(e.name, '')) NOT LIKE '%ST%'
              AND COALESCE(e.name, '') NOT LIKE '%退%'
            ORDER BY e.date, e.symbol
            """,
            [start_hist.date(), end_date.date()],
        ).df()
    finally:
        con.unregister("tmp_signal_dates")

    if df.empty:
        return df

    num_cols = [
        "close",
        "close_adj",
        "amount",
        "pe_ttm",
        "pb",
        "dividend_yield",
        "ma120",
        "ma120_adj",
        "amount_ma20",
        "circ_mv_10k",
        "total_mv_10k",
        "hfq_factor",
        "qfq_factor",
    ]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.dropna(subset=["date", "symbol"]).copy()


def filter_value_dividend_ma(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    pe_ttm = cast(pd.Series, pd.to_numeric(df["pe_ttm"], errors="coerce"))
    dividend_yield = cast(
        pd.Series, pd.to_numeric(df["dividend_yield"], errors="coerce")
    )
    ma120_adj = cast(pd.Series, pd.to_numeric(df["ma120_adj"], errors="coerce"))
    close_adj = cast(pd.Series, pd.to_numeric(df["close_adj"], errors="coerce"))
    circ_mv = cast(pd.Series, pd.to_numeric(df["circ_mv_10k"], errors="coerce"))
    mask = (
        (pe_ttm > 0)
        & (pe_ttm < 20)
        & (dividend_yield > 4.0)
        & ma120_adj.notna()
        & (ma120_adj > 0)
        & (close_adj > 0)
        & (close_adj <= ma120_adj * 0.88)
        & (circ_mv >= NONBANK_MIN_CIRC_MV_10K)
    )
    return cast(pd.DataFrame, df.loc[mask].copy())


def parse_board_keywords(raw: str) -> list[str]:
    words = [x.strip() for x in raw.split(",") if x.strip()]
    if words:
        return words
    return list(OTHER_BOARD_KEYWORDS_DEFAULT)


def retry_call(func: Callable[[], Any], tries: int, sleep_base: float) -> Any:
    last_error: Exception | None = None
    for i in range(max(1, int(tries))):
        try:
            return func()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if i + 1 < max(1, int(tries)):
                time.sleep(max(0.0, float(sleep_base)) * float(i + 1))
    if last_error is not None:
        raise last_error
    raise RuntimeError("retry_call failed unexpectedly")


def normalize_symbol_code(raw: Any) -> str:
    sym = str(raw).strip().upper()
    if not sym:
        return ""
    if "." in sym:
        return sym
    if not sym.isdigit() or len(sym) != 6:
        return sym
    if sym.startswith(("4", "8")):
        return f"{sym}.BJ"
    if sym.startswith(("5", "6", "9")):
        return f"{sym}.SH"
    return f"{sym}.SZ"


def load_board_universes(
    *,
    proxy_host: str,
    token: str,
    proxy_retry: int,
    tries: int,
    sleep: float,
    other_board_keywords: list[str],
) -> BoardUniverses:
    if not token:
        raise ValueError("akshare token is required for board constituent queries")

    ensure_akshare_proxy_patch(
        proxy_host=proxy_host,
        token=token,
        retry=max(1, int(proxy_retry)),
    )

    board_names = retry_call(
        lambda: ak.stock_board_industry_name_em(),
        tries=max(1, int(tries)),
        sleep_base=float(sleep),
    )
    board_df = cast(pd.DataFrame, board_names)
    if board_df is None or board_df.empty:
        raise RuntimeError("AkShare board list is empty")
    if "板块代码" not in board_df.columns or "板块名称" not in board_df.columns:
        raise RuntimeError(
            "AkShare board list missing required columns: 板块代码/板块名称"
        )

    board_work = cast(
        pd.DataFrame,
        board_df.loc[:, ["板块代码", "板块名称"]].copy(),
    )
    board_code_col = cast(pd.Series, board_work["板块代码"].astype(str))
    board_name_col = cast(pd.Series, board_work["板块名称"].astype(str))
    board_work["板块代码"] = board_code_col.str.upper().str.strip()
    board_work["板块名称"] = board_name_col.str.strip()

    board_code_col = cast(pd.Series, board_work["板块代码"])
    board_name_col = cast(pd.Series, board_work["板块名称"])

    bank_rows = cast(
        pd.DataFrame,
        board_work.loc[board_name_col == BANK_INDUSTRY].copy(),
    )
    if bank_rows.empty:
        raise RuntimeError("AkShare board list does not contain 银行 board")
    bank_row = bank_rows.iloc[0]
    bank_code = str(bank_row["板块代码"])
    bank_name = str(bank_row["板块名称"])

    metal_rows = cast(
        pd.DataFrame,
        board_work.loc[board_code_col == METAL_BOARD_CODE].copy(),
    )
    if metal_rows.empty:
        metal_rows = cast(
            pd.DataFrame,
            board_work.loc[board_name_col == METAL_BOARD_NAME].copy(),
        )
    if metal_rows.empty:
        raise RuntimeError(
            f"AkShare board list does not contain {METAL_BOARD_CODE}/{METAL_BOARD_NAME}"
        )
    metal_row = metal_rows.iloc[0]
    metal_code = str(metal_row["板块代码"])
    metal_name = str(metal_row["板块名称"])

    escaped_keywords = [re.escape(k) for k in other_board_keywords if k.strip()]
    if not escaped_keywords:
        raise ValueError("other board keywords are empty")
    pattern = "|".join(escaped_keywords)

    match_mask = cast(
        pd.Series,
        cast(pd.Series, board_work["板块名称"])
        .astype(str)
        .str.contains(pattern, regex=True),
    )
    other_mask = (
        (~cast(pd.Series, board_work["板块代码"]).isin([bank_code, metal_code]))
        & (~cast(pd.Series, board_work["板块名称"]).isin([bank_name, metal_name]))
        & match_mask
    )
    other_rows = cast(pd.DataFrame, board_work.loc[other_mask].copy())
    other_rows = cast(
        pd.DataFrame,
        other_rows.sort_values(["板块名称", "板块代码"]).drop_duplicates(
            subset=["板块代码"], keep="first"
        ),
    )
    if other_rows.empty:
        raise RuntimeError(
            f"No AkShare boards matched other keywords: {','.join(other_board_keywords)}"
        )

    symbol_name_map: dict[str, str] = {}

    def fetch_constituents(board_symbol: str) -> set[str]:
        raw_df = retry_call(
            lambda: ak.stock_board_industry_cons_em(symbol=board_symbol),
            tries=max(1, int(tries)),
            sleep_base=float(sleep),
        )
        cons_df = cast(pd.DataFrame, raw_df)
        if cons_df is None or cons_df.empty:
            return set()
        if "代码" not in cons_df.columns:
            return set()

        symbols: set[str] = set()
        for rec in cons_df.itertuples(index=False):
            code = normalize_symbol_code(getattr(rec, "代码", ""))
            if not code:
                continue
            symbols.add(code)
            nm = str(getattr(rec, "名称", "") or "").strip()
            if nm:
                symbol_name_map[code] = nm
        return symbols

    bank_symbols = fetch_constituents(bank_code)
    metal_symbols = fetch_constituents(metal_code)

    other_symbols: set[str] = set()
    for row in other_rows.itertuples(index=False):
        code = str(getattr(row, "板块代码")).strip().upper()
        if not code:
            continue
        other_symbols |= fetch_constituents(code)

    other_symbols -= bank_symbols
    other_symbols -= metal_symbols

    if not bank_symbols:
        raise RuntimeError("Bank board constituents empty from AkShare")
    if not metal_symbols:
        raise RuntimeError(
            f"Metal board ({METAL_BOARD_CODE}/{METAL_BOARD_NAME}) constituents empty"
        )
    if not other_symbols:
        raise RuntimeError("Other board constituents empty after exclusion")

    return BoardUniverses(
        bank_symbols=bank_symbols,
        metal_symbols=metal_symbols,
        other_symbols=other_symbols,
        symbol_name_map=symbol_name_map,
        bank_board_code=bank_code,
        bank_board_name=bank_name,
        metal_board_code=metal_code,
        metal_board_name=metal_name,
        other_board_codes=other_rows["板块代码"].astype(str).tolist(),
        other_board_names=other_rows["板块名称"].astype(str).tolist(),
    )


def select_bank(
    snapshot: pd.DataFrame,
    top_n: int,
    board_universes: BoardUniverses,
) -> pd.DataFrame:
    bank = cast(
        pd.DataFrame,
        snapshot.loc[
            snapshot["symbol"].isin(sorted(board_universes.bank_symbols))
        ].copy(),
    )
    if bank.empty:
        return bank

    amount_ma20 = cast(pd.Series, pd.to_numeric(bank["amount_ma20"], errors="coerce"))
    dividend_yield = cast(
        pd.Series, pd.to_numeric(bank["dividend_yield"], errors="coerce")
    )
    pb = cast(pd.Series, pd.to_numeric(bank["pb"], errors="coerce"))
    pe_ttm = cast(pd.Series, pd.to_numeric(bank["pe_ttm"], errors="coerce"))
    base_mask = (
        (amount_ma20.fillna(0) > 0)
        & (dividend_yield.fillna(0) > 0)
        & (pb.fillna(0) > 0)
        & (pe_ttm.fillna(0) > 0)
    )
    bank = cast(pd.DataFrame, bank.loc[base_mask].copy())
    if bank.empty:
        return bank

    amount_ma20 = cast(pd.Series, pd.to_numeric(bank["amount_ma20"], errors="coerce"))
    dividend_yield = cast(
        pd.Series, pd.to_numeric(bank["dividend_yield"], errors="coerce")
    )
    pb = cast(pd.Series, pd.to_numeric(bank["pb"], errors="coerce"))
    pe_ttm = cast(pd.Series, pd.to_numeric(bank["pe_ttm"], errors="coerce"))

    cashflow_score = cast(pd.Series, amount_ma20.rank(pct=True))
    dividend_score = cast(pd.Series, dividend_yield.rank(pct=True))
    pb_score = cast(pd.Series, (-pb).rank(pct=True))
    pe_score = cast(pd.Series, (-pe_ttm).rank(pct=True))

    bank["bank_score"] = cashflow_score + dividend_score + pb_score + pe_score
    bank = cast(
        pd.DataFrame,
        bank.sort_values(
            by=["bank_score", "dividend_yield", "pb", "symbol"],
            ascending=[False, False, True, True],
        ),
    )
    bank_limit = max(1, min(2, int(top_n)))
    return cast(pd.DataFrame, bank.head(bank_limit).copy())


def pick_industry_leaders(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    if df.empty:
        return df
    leaders = cast(
        pd.DataFrame,
        df.sort_values(
            by=["industry", "circ_mv_10k", "total_mv_10k", "symbol"],
            ascending=[True, False, False, True],
        )
        .drop_duplicates(subset=["industry"], keep="first")
        .copy(),
    )
    leaders = cast(
        pd.DataFrame,
        leaders.sort_values(
            by=["dividend_yield", "pe_ttm", "circ_mv_10k", "symbol"],
            ascending=[False, True, False, True],
        ),
    )
    return cast(pd.DataFrame, leaders.head(max(1, int(top_n))).copy())


def select_metal(
    snapshot: pd.DataFrame,
    top_n: int,
    board_universes: BoardUniverses,
) -> pd.DataFrame:
    metal = cast(
        pd.DataFrame,
        snapshot.loc[
            snapshot["symbol"].isin(sorted(board_universes.metal_symbols))
        ].copy(),
    )
    metal = filter_value_dividend_ma(metal)
    return pick_industry_leaders(metal, top_n=max(1, min(1, int(top_n))))


def select_other(
    snapshot: pd.DataFrame,
    top_n: int,
    board_universes: BoardUniverses,
) -> pd.DataFrame:
    mask = (
        snapshot["symbol"].isin(sorted(board_universes.other_symbols))
        & (~snapshot["symbol"].isin(sorted(board_universes.bank_symbols)))
        & (~snapshot["symbol"].isin(sorted(board_universes.metal_symbols)))
    )
    other = cast(pd.DataFrame, snapshot.loc[mask].copy())
    other = filter_value_dividend_ma(other)
    return pick_industry_leaders(other, top_n=max(1, min(1, int(top_n))))


def build_rebalance_selections(
    signal_snapshot: pd.DataFrame,
    rebalance_dates: list[pd.Timestamp],
    signal_by_rebalance: dict[pd.Timestamp, pd.Timestamp],
    board_universes: BoardUniverses,
    bank_top_n: int,
    metal_top_n: int,
    other_top_n: int,
) -> tuple[dict[pd.Timestamp, RebalanceSelection], pd.DataFrame]:
    selections: dict[pd.Timestamp, RebalanceSelection] = {}
    rows: list[dict[str, Any]] = []

    for reb_date in rebalance_dates:
        signal_date = signal_by_rebalance.get(reb_date)
        if signal_date is None:
            continue
        signal_mask = signal_snapshot["date"] == pd.Timestamp(signal_date)
        snap = cast(pd.DataFrame, signal_snapshot.loc[signal_mask].copy())
        if snap.empty:
            continue

        bank_df = select_bank(snap, top_n=bank_top_n, board_universes=board_universes)
        metal_df = select_metal(
            snap,
            top_n=metal_top_n,
            board_universes=board_universes,
        )
        other_df = select_other(
            snap,
            top_n=other_top_n,
            board_universes=board_universes,
        )

        bank_symbols = bank_df["symbol"].astype(str).tolist()
        metal_symbols = metal_df["symbol"].astype(str).tolist()
        other_symbols = other_df["symbol"].astype(str).tolist()

        selections[reb_date] = RebalanceSelection(
            rebalance_date=reb_date,
            signal_date=signal_date,
            bank_symbols=bank_symbols,
            metal_symbols=metal_symbols,
            other_symbols=other_symbols,
        )

        for sleeve, part in (
            ("bank", bank_df),
            ("metal", metal_df),
            ("other", other_df),
        ):
            if part.empty:
                continue
            for _, rec in part.iterrows():
                rows.append(
                    {
                        "rebalance_date": reb_date.date().isoformat(),
                        "signal_date": signal_date.date().isoformat(),
                        "sleeve": sleeve,
                        "symbol": str(rec["symbol"]),
                        "name": rec.get("name"),
                        "industry": rec.get("industry"),
                        "close": rec.get("close"),
                        "close_adj": rec.get("close_adj"),
                        "ma120": rec.get("ma120"),
                        "ma120_adj": rec.get("ma120_adj"),
                        "pe_ttm": rec.get("pe_ttm"),
                        "pb": rec.get("pb"),
                        "dividend_yield": rec.get("dividend_yield"),
                        "amount_ma20": rec.get("amount_ma20"),
                        "circ_mv_10k": rec.get("circ_mv_10k"),
                        "hfq_factor": rec.get("hfq_factor"),
                        "qfq_factor": rec.get("qfq_factor"),
                    }
                )

    picks_df = pd.DataFrame(rows)
    return selections, picks_df


def build_bank_dividend_lookup(
    signal_snapshot: pd.DataFrame,
    board_universes: BoardUniverses,
) -> dict[tuple[pd.Timestamp, str], float]:
    if signal_snapshot.empty:
        return {}

    bank_rows = cast(
        pd.DataFrame,
        signal_snapshot.loc[
            signal_snapshot["symbol"].isin(sorted(board_universes.bank_symbols)),
            ["date", "symbol", "dividend_yield"],
        ].copy(),
    )
    if bank_rows.empty:
        return {}

    bank_rows["date"] = pd.to_datetime(bank_rows["date"], errors="coerce")
    bank_rows["dividend_yield"] = pd.to_numeric(
        bank_rows["dividend_yield"], errors="coerce"
    )
    bank_rows = cast(
        pd.DataFrame,
        bank_rows.dropna(subset=["date", "symbol", "dividend_yield"]),
    )
    if bank_rows.empty:
        return {}

    lookup: dict[tuple[pd.Timestamp, str], float] = {}
    for rec in bank_rows.itertuples(index=False):
        dt = normalize_ts(getattr(rec, "date"))
        sym = str(getattr(rec, "symbol"))
        dy = float(getattr(rec, "dividend_yield"))
        if not np.isfinite(dy):
            continue
        lookup[(dt, sym)] = dy
    return lookup


def load_close_lookup(
    con: duckdb.DuckDBPyConnection,
    symbols: list[str],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> dict[tuple[pd.Timestamp, str], float]:
    if not symbols:
        return {}
    sym_df = pd.DataFrame({"symbol": sorted(set(symbols))})
    con.register("tmp_symbols", sym_df)
    try:
        df = con.execute(
            """
            SELECT date, symbol, close
            FROM v_bar_daily_raw
            INNER JOIN tmp_symbols USING (symbol)
            WHERE date BETWEEN ? AND ?
            ORDER BY date, symbol
            """,
            [start_date.date(), end_date.date()],
        ).df()
    finally:
        con.unregister("tmp_symbols")

    if df.empty:
        return {}

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["date", "symbol", "close"])
    lookup: dict[tuple[pd.Timestamp, str], float] = {}
    for rec in df.itertuples(index=False):
        dt = normalize_ts(getattr(rec, "date"))
        sym = str(getattr(rec, "symbol"))
        close_px = float(getattr(rec, "close"))
        lookup[(dt, sym)] = close_px
    return lookup


def build_daily_targets_with_take_profit(
    trading_dates: list[pd.Timestamp],
    rebalance_dates: list[pd.Timestamp],
    selections: dict[pd.Timestamp, RebalanceSelection],
    close_lookup: dict[tuple[pd.Timestamp, str], float],
    bank_dividend_lookup: dict[tuple[pd.Timestamp, str], float] | None = None,
    bank_dividend_floor: float = BANK_REBALANCE_DIVIDEND_FLOOR,
) -> tuple[
    dict[pd.Timestamp, dict[str, float]],
    dict[pd.Timestamp, list[str]],
    set[pd.Timestamp],
    dict[pd.Timestamp, str],
    dict[tuple[pd.Timestamp, str], str],
    pd.DataFrame,
]:
    if not trading_dates:
        return {}, {}, set(), {}, {}, pd.DataFrame()

    rebalance_dates = [d for d in rebalance_dates if d in selections]
    if not rebalance_dates:
        return {}, {}, set(), {}, {}, pd.DataFrame()

    bank_dividend_lookup = bank_dividend_lookup or {}

    active: dict[str, list[str]] = {"bank": [], "metal": [], "other": []}
    entry_price: dict[str, dict[str, float]] = {"metal": {}, "other": {}}

    daily_targets: dict[pd.Timestamp, dict[str, float]] = {}
    daily_bank_symbols: dict[pd.Timestamp, list[str]] = {}
    bank_rebalance_days: set[pd.Timestamp] = set()
    bank_rebalance_reasons: dict[pd.Timestamp, str] = {}
    nonbank_exit_events: dict[tuple[pd.Timestamp, str], str] = {}
    trace_rows: list[dict[str, Any]] = []

    for day in trading_dates:
        bank_rebalance_today = False
        bank_rebalance_reason = ""
        if day in selections:
            pick = selections[day]

            if not active["bank"]:
                active["bank"] = list(pick.bank_symbols)
                bank_rebalance_today = bool(active["bank"])
                if bank_rebalance_today:
                    bank_rebalance_reason = TRIGGER_BANK_INITIAL_BUILD
            else:
                held_count = max(1, min(2, len(active["bank"])))
                to_replace: set[str] = set()
                for sym in active["bank"]:
                    dy = bank_dividend_lookup.get((day, sym))
                    if dy is None or not np.isfinite(dy):
                        continue
                    if float(dy) < float(bank_dividend_floor):
                        to_replace.add(sym)

                if to_replace:
                    updated_bank = [s for s in active["bank"] if s not in to_replace]
                    high_dividend_candidates = {
                        s
                        for s in pick.bank_symbols
                        if (
                            bank_dividend_lookup.get((day, s)) is not None
                            and np.isfinite(float(bank_dividend_lookup[(day, s)]))
                            and float(bank_dividend_lookup[(day, s)])
                            >= float(bank_dividend_floor)
                        )
                    }
                    replacement_candidates = [
                        s
                        for s in pick.bank_symbols
                        if (
                            s in high_dividend_candidates
                            and s not in to_replace
                            and s not in updated_bank
                        )
                    ]
                    need = max(0, held_count - len(updated_bank))
                    if need > 0:
                        updated_bank.extend(replacement_candidates[:need])
                    updated_bank = updated_bank[:held_count]
                    if updated_bank and updated_bank != active["bank"]:
                        active["bank"] = updated_bank
                        bank_rebalance_today = True
                        bank_rebalance_reason = TRIGGER_BANK_DIVIDEND_REBALANCE

            for sleeve, picked in (
                ("metal", pick.metal_symbols),
                ("other", pick.other_symbols),
            ):
                if active[sleeve]:
                    continue

                initialized_symbol: str | None = None
                initialized_price: float | None = None
                for sym in picked:
                    px = close_lookup.get((day, sym))
                    if px is None or not np.isfinite(px) or px <= 0:
                        continue
                    initialized_symbol = sym
                    initialized_price = float(px)
                    break

                if initialized_symbol is None or initialized_price is None:
                    continue
                active[sleeve] = [initialized_symbol]
                entry_price[sleeve] = {initialized_symbol: initialized_price}

        if bank_rebalance_today:
            bank_rebalance_days.add(day)
            bank_rebalance_reasons[day] = (
                bank_rebalance_reason or TRIGGER_BANK_DIVIDEND_REBALANCE
            )

        for sleeve in ("metal", "other"):
            for sym in list(active[sleeve]):
                ep = entry_price[sleeve].get(sym)
                px = close_lookup.get((day, sym))
                if ep is None or px is None or not np.isfinite(px) or px <= 0:
                    continue
                if float(px) >= ep * 1.30:
                    active[sleeve].remove(sym)
                    entry_price[sleeve].pop(sym, None)
                    nonbank_exit_events[(day, sym)] = TRIGGER_TAKE_PROFIT
                elif float(px) <= ep * 0.80:
                    active[sleeve].remove(sym)
                    entry_price[sleeve].pop(sym, None)
                    nonbank_exit_events[(day, sym)] = TRIGGER_STOP_LOSS

        targets: dict[str, float] = {}
        for sleeve, symbols, sleeve_weight in (
            ("metal", active["metal"], SLEEVE_WEIGHTS["metal"]),
            ("other", active["other"], SLEEVE_WEIGHTS["other"]),
        ):
            sleeve_symbols = symbols[:1]
            if not sleeve_symbols:
                continue
            per_weight = float(sleeve_weight) / float(len(sleeve_symbols))
            for sym in sleeve_symbols:
                targets[sym] = per_weight

        daily_targets[day] = targets
        daily_bank_symbols[day] = list(active["bank"])

        bank_target_weight = SLEEVE_WEIGHTS["bank"] if len(active["bank"]) > 0 else 0.0

        trace_rows.append(
            {
                "date": day.date().isoformat(),
                "bank_count": len(active["bank"]),
                "metal_count": len(active["metal"]),
                "other_count": len(active["other"]),
                "bank_weight_target": bank_target_weight,
                "metal_weight": sum(targets.get(s, 0.0) for s in active["metal"]),
                "other_weight": sum(targets.get(s, 0.0) for s in active["other"]),
                "cash_weight": 1.0 - bank_target_weight - sum(targets.values()),
            }
        )

    return (
        daily_targets,
        daily_bank_symbols,
        bank_rebalance_days,
        bank_rebalance_reasons,
        nonbank_exit_events,
        pd.DataFrame(trace_rows),
    )


def load_market_data(
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
            FROM v_bar_daily_raw
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

    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "symbol", "open", "high", "low", "close", "volume"])

    out: dict[str, pd.DataFrame] = {}
    for sym, group in df.groupby("symbol", sort=True):
        work = group.copy()
        work = work.sort_values("date").drop_duplicates(subset=["date"], keep="last")
        if work.empty:
            continue
        out[str(sym)] = cast(
            pd.DataFrame,
            work[
                ["date", "open", "high", "low", "close", "volume", "symbol"]
            ].reset_index(drop=True),
        )
    return out


class ScheduledTargetWeightsStrategy(Strategy):
    warmup_period = 1

    def __init__(
        self,
        daily_targets: dict[pd.Timestamp, dict[str, float]],
        daily_bank_symbols: dict[pd.Timestamp, list[str]],
        bank_rebalance_days: set[pd.Timestamp],
    ):
        self.warmup_period = 1
        self.daily_targets = {
            normalize_ts(d): {str(k): float(v) for k, v in w.items()}
            for d, w in daily_targets.items()
        }
        self.daily_bank_symbols = {
            normalize_ts(d): [str(s) for s in symbols]
            for d, symbols in daily_bank_symbols.items()
        }
        self.bank_rebalance_days = {normalize_ts(d) for d in bank_rebalance_days}
        self.bank_universe = {
            sym for symbols in self.daily_bank_symbols.values() for sym in symbols
        }
        self.current_date: pd.Timestamp | None = None
        self.current_targets: dict[str, float] = {}
        self.current_bank_symbols: list[str] = []
        self.is_bank_rebalance_day = False
        self.bank_force_trim_today = False
        self.latest_close: dict[str, float] = {}

    def _switch_day(self, bar_date: pd.Timestamp) -> None:
        if self.current_date == bar_date:
            return
        self.current_date = bar_date
        self.current_targets = self.daily_targets.get(bar_date, {})
        self.current_bank_symbols = self.daily_bank_symbols.get(bar_date, [])
        self.is_bank_rebalance_day = bar_date in self.bank_rebalance_days
        self.bank_force_trim_today = self._estimate_bank_weight() > 0.60

    def _estimate_bank_weight(
        self,
        current_symbol: str | None = None,
        current_close: float | None = None,
    ) -> float:
        if not self.current_bank_symbols:
            return 0.0
        equity = float(getattr(self.ctx, "equity", 0.0) or 0.0)
        if equity <= 0:
            return 0.0

        bank_value = 0.0
        for sym in self.current_bank_symbols:
            qty = float(self.get_position(sym))
            if qty <= 0:
                continue
            if (
                current_symbol is not None
                and sym == current_symbol
                and current_close is not None
            ):
                px = float(current_close)
            else:
                px = self.latest_close.get(sym)
                if px is None:
                    continue
            if px <= 0 or not np.isfinite(px):
                continue
            bank_value += qty * float(px)
        return bank_value / equity

    def on_bar(self, bar) -> None:
        bar_date = normalize_ts(getattr(bar, "timestamp"))
        self._switch_day(bar_date)

        symbol = str(bar.symbol)
        close_px = float(bar.close)
        if close_px > 0 and np.isfinite(close_px):
            self.latest_close[symbol] = close_px

        position = float(self.get_position(symbol))

        if symbol in self.bank_universe:
            if self.is_bank_rebalance_day or self.bank_force_trim_today:
                if symbol in self.current_bank_symbols and self.current_bank_symbols:
                    target = SLEEVE_WEIGHTS["bank"] / float(
                        len(self.current_bank_symbols)
                    )
                else:
                    target = 0.0
                if target > 0.0 or position > 0.0:
                    self.order_target_percent(float(target), symbol)
            return

        target = float(self.current_targets.get(symbol, 0.0))

        if target > 0.0 or position > 0.0:
            self.order_target_percent(target, symbol)


def flatten_daily_targets(
    daily_targets: dict[pd.Timestamp, dict[str, float]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for day in sorted(daily_targets.keys()):
        weights = daily_targets[day]
        if not weights:
            rows.append(
                {
                    "date": day.date().isoformat(),
                    "symbol": "",
                    "target_weight": 0.0,
                }
            )
            continue
        for sym, wt in sorted(weights.items()):
            rows.append(
                {
                    "date": day.date().isoformat(),
                    "symbol": sym,
                    "target_weight": float(wt),
                }
            )
    return pd.DataFrame(rows)


def build_symbol_name_lookup(
    snapshot: pd.DataFrame,
    board_universes: BoardUniverses,
) -> dict[str, str]:
    lookup: dict[str, str] = dict(board_universes.symbol_name_map)
    if snapshot.empty:
        return lookup

    symbols = cast(pd.Series, snapshot["symbol"].astype(str))
    names = cast(
        pd.Series,
        snapshot["name"].where(snapshot["name"].notna(), "").astype(str),
    )
    for sym, name in zip(symbols.tolist(), names.tolist()):
        ts_sym = normalize_symbol_code(sym)
        nm = str(name).strip()
        if ts_sym and nm:
            lookup[ts_sym] = nm
    return lookup


def enrich_trades_with_names(
    trades_df: pd.DataFrame,
    symbol_name_lookup: dict[str, str],
) -> pd.DataFrame:
    if trades_df.empty or "symbol" not in trades_df.columns:
        return trades_df

    out = trades_df.copy()
    raw_symbols = cast(pd.Series, out["symbol"].astype(str))
    symbol_codes = [normalize_symbol_code(s) for s in raw_symbols.tolist()]
    name_cn = [symbol_name_lookup.get(s, "") for s in symbol_codes]
    symbol_display = [f"{s}|{n}" if n else s for s, n in zip(symbol_codes, name_cn)]

    out.insert(0, "symbol_code", symbol_codes)
    out.insert(1, "name_cn", name_cn)
    out["symbol"] = symbol_display
    return out


def build_previous_trade_day_map(
    trading_dates: list[pd.Timestamp],
) -> dict[pd.Timestamp, pd.Timestamp]:
    if len(trading_dates) < 2:
        return {}
    return {
        trading_dates[i]: trading_dates[i - 1] for i in range(1, len(trading_dates))
    }


def infer_order_day_from_fill_time(
    fill_time: Any,
    previous_trade_day_map: dict[pd.Timestamp, pd.Timestamp],
) -> pd.Timestamp | None:
    ts = pd.to_datetime(fill_time, errors="coerce")
    if ts is None or pd.isna(ts):
        return None
    fill_day = normalize_ts(ts)
    return previous_trade_day_map.get(fill_day)


def annotate_trades_with_triggers(
    trades_df: pd.DataFrame,
    trading_dates: list[pd.Timestamp],
    daily_targets: dict[pd.Timestamp, dict[str, float]],
    daily_bank_symbols: dict[pd.Timestamp, list[str]],
    bank_rebalance_days: set[pd.Timestamp],
    bank_rebalance_reasons: dict[pd.Timestamp, str],
    nonbank_exit_events: dict[tuple[pd.Timestamp, str], str],
) -> pd.DataFrame:
    if trades_df.empty:
        return trades_df

    out = trades_df.copy()
    previous_trade_day_map = build_previous_trade_day_map(trading_dates)
    bank_universe = {sym for symbols in daily_bank_symbols.values() for sym in symbols}

    if "symbol_code" in out.columns:
        symbol_codes = [
            normalize_symbol_code(str(x))
            for x in cast(pd.Series, out["symbol_code"]).tolist()
        ]
    else:
        symbol_codes = [
            normalize_symbol_code(str(x).split("|")[0])
            for x in cast(pd.Series, out["symbol"]).tolist()
        ]

    entry_times = cast(pd.Series, out.get("entry_time", pd.Series(dtype="object")))
    exit_times = cast(pd.Series, out.get("exit_time", pd.Series(dtype="object")))

    entry_trigger: list[str] = []
    exit_trigger: list[str] = []

    for idx, sym in enumerate(symbol_codes):
        entry_time = entry_times.iloc[idx] if idx < len(entry_times) else None
        exit_time = exit_times.iloc[idx] if idx < len(exit_times) else None

        entry_order_day = infer_order_day_from_fill_time(
            entry_time, previous_trade_day_map
        )
        if entry_order_day is None:
            entry_trigger.append(TRIGGER_TARGET_ADJUSTMENT_ENTRY)
        elif sym in bank_universe:
            if entry_order_day in bank_rebalance_days:
                entry_trigger.append(
                    bank_rebalance_reasons.get(
                        entry_order_day, TRIGGER_BANK_DIVIDEND_REBALANCE
                    )
                )
            else:
                entry_trigger.append(TRIGGER_BANK_WEIGHT_TRIM)
        else:
            target_w = float(daily_targets.get(entry_order_day, {}).get(sym, 0.0))
            if target_w > 0:
                entry_trigger.append(TRIGGER_MONTHLY_SELECTION_ENTRY)
            else:
                entry_trigger.append(TRIGGER_TARGET_ADJUSTMENT_ENTRY)

        exit_order_day = infer_order_day_from_fill_time(
            exit_time, previous_trade_day_map
        )
        if exit_order_day is None:
            exit_trigger.append("")
        elif sym in bank_universe:
            if exit_order_day in bank_rebalance_days:
                exit_trigger.append(
                    bank_rebalance_reasons.get(
                        exit_order_day, TRIGGER_BANK_DIVIDEND_REBALANCE
                    )
                )
            else:
                exit_trigger.append(TRIGGER_BANK_WEIGHT_TRIM)
        else:
            exit_reason = nonbank_exit_events.get((exit_order_day, sym))
            if exit_reason:
                exit_trigger.append(exit_reason)
            else:
                exit_trigger.append(TRIGGER_TARGET_ADJUSTMENT_EXIT)

    out["entry_trigger"] = entry_trigger
    out["exit_trigger"] = exit_trigger
    return out


def rename_output_columns_to_chinese(
    df: pd.DataFrame, output_kind: str
) -> pd.DataFrame:
    mapping = OUTPUT_COLUMN_NAME_MAPS.get(output_kind, {})
    if not mapping or df.empty:
        return df.rename(columns=mapping)
    return df.rename(columns={col: mapping.get(col, col) for col in df.columns})


def main() -> None:
    args = parse_args()
    if not args.db.exists():
        raise FileNotFoundError(f"DuckDB not found: {args.db}")

    con = duckdb.connect(args.db.as_posix(), read_only=True)
    try:
        start_date = normalize_ts(args.start_date)
        end_date = resolve_end_date(con, args.end_date)
        if end_date < start_date:
            raise ValueError(
                f"end_date {end_date.date()} is earlier than start_date {start_date.date()}"
            )

        pre_start_date = load_previous_trade_date(con, start_date)
        if pre_start_date is None:
            raise RuntimeError("No prior trading date before start_date")

        trading_dates_all = load_trading_dates(con, pre_start_date, end_date)
        if not trading_dates_all:
            raise RuntimeError("No trading dates found in selected period")
        trading_dates = [d for d in trading_dates_all if d >= start_date]
        if not trading_dates:
            raise RuntimeError("No in-window trading dates found in selected period")

        rebalance_dates = first_trade_day_per_month(trading_dates)
        signal_by_rebalance: dict[pd.Timestamp, pd.Timestamp] = {}
        for reb_date in rebalance_dates:
            signal = previous_trade_date(trading_dates_all, reb_date)
            if signal is not None:
                signal_by_rebalance[reb_date] = signal

        if not signal_by_rebalance:
            raise RuntimeError("No signal dates found for monthly rebalancing")

        signal_dates = sorted(set(signal_by_rebalance.values()))
        other_keywords = parse_board_keywords(str(args.other_board_keywords))
        board_universes = load_board_universes(
            proxy_host=str(args.akshare_proxy_host),
            token=str(args.akshare_token),
            proxy_retry=int(args.akshare_proxy_retry),
            tries=int(args.board_tries),
            sleep=float(args.board_sleep),
            other_board_keywords=other_keywords,
        )

        history_start = cast(pd.Timestamp, pre_start_date - pd.Timedelta(days=260))
        snapshot = load_signal_snapshot(
            con,
            signal_dates=signal_dates,
            start_hist=history_start,
            end_date=end_date,
        )
        if snapshot.empty:
            raise RuntimeError("Signal snapshot is empty; cannot build selections")

        selections, picks_df = build_rebalance_selections(
            signal_snapshot=snapshot,
            rebalance_dates=rebalance_dates,
            signal_by_rebalance=signal_by_rebalance,
            board_universes=board_universes,
            bank_top_n=max(1, min(2, int(args.bank_top_n))),
            metal_top_n=max(1, min(1, int(args.metal_top_n))),
            other_top_n=max(1, min(1, int(args.other_top_n))),
        )
        if not selections:
            raise RuntimeError("No monthly selections generated")

        bank_dividend_lookup = build_bank_dividend_lookup(snapshot, board_universes)

        selected_symbols = sorted(
            {
                sym
                for pick in selections.values()
                for sym in (pick.bank_symbols + pick.metal_symbols + pick.other_symbols)
            }
        )
        if not selected_symbols:
            raise RuntimeError("Selection symbol set is empty")

        close_lookup = load_close_lookup(
            con,
            symbols=selected_symbols,
            start_date=pre_start_date,
            end_date=end_date,
        )

        selections_by_signal_date = {
            signal_by_rebalance[reb_date]: selections[reb_date]
            for reb_date in rebalance_dates
            if reb_date in selections and reb_date in signal_by_rebalance
        }
        rebalance_signal_dates = sorted(selections_by_signal_date.keys())

        (
            daily_targets,
            daily_bank_symbols,
            bank_rebalance_days,
            bank_rebalance_reasons,
            nonbank_exit_events,
            daily_trace_df,
        ) = build_daily_targets_with_take_profit(
            trading_dates=trading_dates_all,
            rebalance_dates=rebalance_signal_dates,
            selections=selections_by_signal_date,
            close_lookup=close_lookup,
            bank_dividend_lookup=bank_dividend_lookup,
            bank_dividend_floor=BANK_REBALANCE_DIVIDEND_FLOOR,
        )
        if not daily_targets:
            raise RuntimeError("Daily target plan is empty")

        market_data = load_market_data(
            con,
            symbols=selected_symbols,
            start_date=pre_start_date,
            end_date=end_date,
        )
    finally:
        con.close()

    if not market_data:
        raise RuntimeError("No market data loaded for selected symbols")

    valid_symbols = set(market_data.keys())
    daily_targets = {
        d: {s: w for s, w in targets.items() if s in valid_symbols}
        for d, targets in daily_targets.items()
    }
    daily_bank_symbols = {
        d: [s for s in symbols if s in valid_symbols]
        for d, symbols in daily_bank_symbols.items()
    }
    bank_rebalance_days = {d for d in bank_rebalance_days if d in daily_bank_symbols}
    bank_rebalance_reasons = {
        d: reason
        for d, reason in bank_rebalance_reasons.items()
        if d in bank_rebalance_days
    }
    nonbank_exit_events = {
        (d, s): reason
        for (d, s), reason in nonbank_exit_events.items()
        if d in daily_targets and s in valid_symbols
    }
    strategy_symbols = sorted(valid_symbols)

    result = run_backtest(
        data=market_data,
        strategy=ScheduledTargetWeightsStrategy(
            daily_targets=daily_targets,
            daily_bank_symbols=daily_bank_symbols,
            bank_rebalance_days=bank_rebalance_days,
        ),
        symbol=strategy_symbols,
        initial_cash=float(args.initial_cash),
        commission_rate=float(args.commission_rate),
        execution_mode=ExecutionMode.NextOpen,
        warmup_period=1,
    )

    metrics_df = result.metrics_df.copy()
    if metrics_df.shape[1] == 1:
        val_col = metrics_df.columns[0]
        metrics_out = pd.DataFrame(
            {
                "metric": metrics_df.index.astype(str),
                "value": metrics_df[val_col].values,
            }
        )
    else:
        metrics_out = metrics_df.reset_index(drop=False)

    symbol_name_lookup = build_symbol_name_lookup(snapshot, board_universes)
    trades_with_names = enrich_trades_with_names(
        result.trades_df.copy(), symbol_name_lookup
    )
    trades_with_triggers = annotate_trades_with_triggers(
        trades_df=trades_with_names,
        trading_dates=trading_dates_all,
        daily_targets=daily_targets,
        daily_bank_symbols=daily_bank_symbols,
        bank_rebalance_days=bank_rebalance_days,
        bank_rebalance_reasons=bank_rebalance_reasons,
        nonbank_exit_events=nonbank_exit_events,
    )

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    picks_out = out_dir / "monthly_picks.csv"
    trace_out = out_dir / "daily_trace.csv"
    targets_out = out_dir / "daily_targets.csv"
    metrics_out_file = out_dir / "metrics.csv"
    trades_out = out_dir / "trades.csv"

    if picks_df.empty:
        picks_export = pd.DataFrame(
            columns=[
                "rebalance_date",
                "signal_date",
                "sleeve",
                "symbol",
                "name",
                "industry",
                "close",
                "close_adj",
                "ma120",
                "ma120_adj",
                "pe_ttm",
                "pb",
                "dividend_yield",
                "amount_ma20",
                "circ_mv_10k",
                "hfq_factor",
                "qfq_factor",
            ]
        )
    else:
        picks_export = picks_df.copy()
    picks_export = rename_output_columns_to_chinese(picks_export, "picks")
    picks_export.to_csv(picks_out, index=False, encoding="utf-8-sig")

    if daily_trace_df.empty:
        trace_export = pd.DataFrame(
            columns=[
                "date",
                "bank_count",
                "metal_count",
                "other_count",
                "bank_weight_target",
                "metal_weight",
                "other_weight",
                "cash_weight",
            ]
        )
    else:
        trace_export = daily_trace_df.copy()
    trace_export = rename_output_columns_to_chinese(trace_export, "trace")
    trace_export.to_csv(trace_out, index=False, encoding="utf-8-sig")

    targets_export = rename_output_columns_to_chinese(
        flatten_daily_targets(daily_targets), "targets"
    )
    metrics_export = rename_output_columns_to_chinese(metrics_out.copy(), "metrics")
    trades_export = rename_output_columns_to_chinese(
        trades_with_triggers.copy(), "trades"
    )

    targets_export.to_csv(targets_out, index=False, encoding="utf-8-sig")
    metrics_export.to_csv(metrics_out_file, index=False, encoding="utf-8-sig")
    trades_export.to_csv(trades_out, index=False, encoding="utf-8-sig")

    print(
        f"[INFO] backtest window: {start_date.date()} -> {end_date.date()} | "
        f"symbols={len(strategy_symbols)}"
    )
    print(
        "[INFO] target ratios: "
        f"bank={SLEEVE_WEIGHTS['bank']:.0%}, "
        f"metal={SLEEVE_WEIGHTS['metal']:.0%}, "
        f"other={SLEEVE_WEIGHTS['other']:.0%}"
    )
    print(
        "[INFO] board universes: "
        f"bank={board_universes.bank_board_name}({board_universes.bank_board_code}) n={len(board_universes.bank_symbols)} | "
        f"metal={board_universes.metal_board_name}({board_universes.metal_board_code}) n={len(board_universes.metal_symbols)} | "
        f"other_boards={len(board_universes.other_board_codes)} n={len(board_universes.other_symbols)}"
    )
    print(f"[RESULT] total_return_pct={result.metrics.total_return_pct:.2f}")
    print(f"[RESULT] sharpe_ratio={result.metrics.sharpe_ratio:.2f}")
    print(f"[RESULT] max_drawdown_pct={result.metrics.max_drawdown_pct:.2f}")
    print(f"[DONE] picks   -> {picks_out}")
    print(f"[DONE] trace   -> {trace_out}")
    print(f"[DONE] targets -> {targets_out}")
    print(f"[DONE] metrics -> {metrics_out_file}")
    print(f"[DONE] trades  -> {trades_out}")


if __name__ == "__main__":
    main()
