from __future__ import annotations

import bisect
import concurrent.futures as cf
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from heapq import heappop, heappush
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl
import vectorbt as vbt

from src.data_clients import create_clickhouse_http_client, query_clickhouse_arrow_df
from src.data_queries import tick_day_rows_sql, tick_trade_dates_sql


@dataclass(frozen=True)
class TriggerConf:
    comment: str
    hold_cnt: int
    amt: float
    vol: float
    b_upper: float
    b_wait: float
    at_sale: float
    hold_sec: float
    high: float
    low: float
    sec: float
    ra_rate: float
    tn_rate: float
    stock_amt: float
    bond_amt: float


@dataclass
class Trade:
    conf_idx: int
    bond_symbol: str
    stock_symbol: str
    entry_ts: datetime
    exit_ts: datetime
    entry_price: float
    exit_price: float
    qty: float

    @property
    def size(self) -> float:
        # 与旧实现兼容：qty * 10 作为下单单位。
        return self.qty * 10.0

    @property
    def pnl(self) -> float:
        return (self.exit_price - self.entry_price) * self.size


def _load_trigger_config(path: str | Path) -> list[TriggerConf]:
    confs = json.loads(Path(path).read_text(encoding="utf-8"))
    out: list[TriggerConf] = []
    for item in confs:
        out.append(
            TriggerConf(
                comment=str(item.get("comment", "")),
                hold_cnt=int(item.get("holdCnt", 1)),
                amt=float(item.get("amt", 0.0)),
                vol=float(item.get("vol", 0.0)),
                b_upper=float(item.get("bUpper", 0.0)),
                b_wait=float(item.get("bWait", 3.0)),
                at_sale=float(item.get("atSale", 0.0)),
                hold_sec=float(item.get("holdSec", 30.0)),
                high=float(item.get("high", 1.0)),
                low=float(item.get("low", 1.0)),
                sec=float(item.get("sec", 10.0)),
                ra_rate=float(item.get("raRate", 0.125)),
                tn_rate=float(item.get("tnRate", 0.0)),
                stock_amt=float(item.get("stockAmt", 20.0)),
                bond_amt=float(item.get("bondAmt", 2.0)),
            )
        )
    return out


def _load_cb_stock_map(cb_info_csv: str | Path) -> tuple[dict[str, list[str]], dict[str, str]]:
    df = pl.read_csv(cb_info_csv, encoding="utf8-lossy")
    if "转债代码" not in df.columns or "正股代码" not in df.columns:
        raise ValueError("可转债基础信息表缺少 转债代码/正股代码 字段")

    b2s: dict[str, str] = {}
    s2b: dict[str, list[str]] = defaultdict(list)
    for row in df.select(["转债代码", "正股代码"]).iter_rows(named=True):
        b = str(row["转债代码"] or "").strip().upper()
        s = str(row["正股代码"] or "").strip().upper()
        if len(b) < 9 or len(s) < 9 or "." not in b or "." not in s:
            continue
        b2s[b] = s
        s2b[s].append(b)
    return dict(s2b), b2s


def _load_stock_shares(shares_json: str | Path | None) -> dict[str, float]:
    if not shares_json:
        return {}
    p = Path(shares_json)
    if not p.exists():
        return {}
    raw = json.loads(p.read_text(encoding="utf-8"))
    out: dict[str, float] = {}
    for k, v in raw.items():
        code6 = str(k).strip()
        if len(code6) != 6:
            continue
        out[code6] = float(v)
    return out


def _symbol_code6(sym: str) -> str:
    return sym.split(".", 1)[0] if "." in sym else sym[:6]


def _calc_vol(conf: TriggerConf, price: float) -> float:
    if price <= 0:
        return 0.0
    if conf.vol > 0:
        return conf.vol
    return (int(conf.amt / 10.0 / price)) * 10.0


def _aggregate_stock_second(stock_df: pl.DataFrame, shares_map: dict[str, float]) -> pl.DataFrame:
    if stock_df.height == 0:
        return pl.DataFrame()
    sec = (
        stock_df.with_columns(
            pl.col("ts").dt.truncate("1s").alias("ts_sec"),
            (pl.col("price") * pl.col("volume") * 100.0).alias("amount_tick"),
            (pl.col("volume") * 100.0).alias("share_tick"),
        )
        .group_by(["symbol", "ts_sec"])
        .agg(
            pl.col("price").last().alias("price"),
            pl.col("amount_tick").sum().alias("amount"),
            pl.col("share_tick").sum().alias("share_volume"),
        )
        .sort(["symbol", "ts_sec"])
        .with_columns(
            pl.col("amount").cum_sum().over("symbol").alias("cum_amount"),
            pl.col("share_volume").cum_sum().over("symbol").alias("cum_shares"),
        )
        .with_columns((pl.col("cum_amount") / 10000.0).alias("cum_amount_wan"))
    )

    shares_df = pl.DataFrame(
        {
            "symbol": list({s for s in sec["symbol"].to_list()}),
        }
    ).with_columns(pl.col("symbol").map_elements(lambda s: shares_map.get(_symbol_code6(str(s)), 0.0), return_dtype=pl.Float64).alias("shares_total"))

    sec = sec.join(shares_df, on="symbol", how="left").with_columns(
        pl.when(pl.col("shares_total") > 0)
        .then(pl.col("cum_shares") / pl.col("shares_total") * 100.0)
        .otherwise(0.0)
        .alias("turnover")
    )
    return sec


def _aggregate_bond_second(bond_df: pl.DataFrame) -> pl.DataFrame:
    if bond_df.height == 0:
        return pl.DataFrame()
    sec = (
        bond_df.with_columns(
            pl.col("ts").dt.truncate("1s").alias("ts_sec"),
            (pl.col("price") * pl.col("volume") * 10.0).alias("amount_tick"),
        )
        .group_by(["symbol", "ts_sec"])
        .agg(
            pl.col("price").last().alias("price"),
            pl.col("amount_tick").sum().alias("amount"),
        )
        .sort(["symbol", "ts_sec"])
        .with_columns(
            pl.col("amount").cum_sum().over("symbol").alias("cum_amount"),
            pl.col("ts_sec").min().over("symbol").alias("first_ts"),
        )
        .with_columns(
            (pl.col("ts_sec").dt.epoch("s") - pl.col("first_ts").dt.epoch("s"))
            .cast(pl.Float64)
            .clip(lower_bound=1.0)
            .alias("elapsed_sec"),
            (pl.col("cum_amount") / 10000.0).alias("cum_amount_wan"),
        )
        .with_columns((pl.col("cum_amount_wan") / pl.col("elapsed_sec")).alias("amt_per_sec_wan"))
    )
    return sec


def _compute_stock_triggers(stock_sec: pl.DataFrame, conf: TriggerConf) -> pl.DataFrame:
    if stock_sec.height == 0:
        return pl.DataFrame()

    sec_n = max(int(round(conf.sec)), 1)

    base = stock_sec.with_columns(pl.col("ts_sec").dt.epoch("s").cast(pl.Int64).alias("ts_epoch"))

    left = base.with_columns((pl.col("ts_epoch") - sec_n).alias("ts_target_epoch"))
    right = base.select(
        "symbol",
        pl.col("ts_sec").alias("ts_prev"),
        pl.col("ts_epoch").alias("ts_prev_epoch"),
        pl.col("price").alias("price_prev"),
        pl.col("cum_amount_wan").alias("cum_amount_prev"),
        pl.col("turnover").alias("turnover_prev"),
    )

    joined = (
        left.join_asof(
            right.sort(["symbol", "ts_prev_epoch"]),
            left_on="ts_target_epoch",
            right_on="ts_prev_epoch",
            by="symbol",
            strategy="backward",
            tolerance=3,
        )
        .with_columns((pl.col("ts_epoch") - pl.col("ts_prev_epoch")).cast(pl.Float64).alias("dt"))
        .filter(pl.col("dt") > 0)
        .with_columns(
            (((pl.col("price") / pl.col("price_prev")) - 1.0) * 100.0).alias("ra_diff"),
            (pl.col("cum_amount_wan") - pl.col("cum_amount_prev")).alias("amount_diff_wan"),
            (pl.col("turnover") - pl.col("turnover_prev")).alias("turnover_diff"),
        )
        .with_columns(
            (pl.col("ra_diff") / pl.col("dt")).alias("ra_rate"),
            (pl.col("amount_diff_wan") / pl.col("dt")).alias("stock_amt_rate"),
            (pl.col("turnover_diff") / pl.col("dt")).alias("tn_rate"),
        )
        .filter(
            (pl.col("price") > 4)
            & (pl.col("ra_rate") >= conf.ra_rate)
            & (pl.col("tn_rate") >= conf.tn_rate)
            & (pl.col("stock_amt_rate") >= conf.stock_amt)
            & (pl.col("dt") >= conf.sec - 3)
            & (pl.col("dt") <= conf.sec + 3)
        )
        .select(
            pl.col("symbol").alias("stock_symbol"),
            "ts_sec",
            pl.col("price").alias("stock_price"),
        )
        .unique(subset=["stock_symbol", "ts_sec"], keep="first")
        .sort(["ts_sec", "stock_symbol"])
    )
    return joined


def _build_bond_lookup(bond_sec: pl.DataFrame) -> dict[str, tuple[list[datetime], list[float]]]:
    out: dict[str, tuple[list[datetime], list[float]]] = {}
    if bond_sec.height == 0:
        return out
    for sym, grp in bond_sec.group_by("symbol", maintain_order=True):
        g = grp.sort("ts_sec")
        sym_key = sym[0] if isinstance(sym, tuple) else sym
        out[str(sym_key)] = (g["ts_sec"].to_list(), g["price"].cast(pl.Float64).to_list())
    return out


def _find_exit_for_trade(
    *,
    conf: TriggerConf,
    times: list[datetime],
    prices: list[float],
    entry_ts: datetime,
    entry_price: float,
) -> tuple[datetime, float]:
    if not times:
        return entry_ts, entry_price

    idx = bisect.bisect_left(times, entry_ts)
    if idx >= len(times):
        return times[-1], prices[-1]

    target = entry_price * (1.0 + conf.at_sale / 100.0) if conf.at_sale > 0 else 0.0
    start = idx + 1 if idx + 1 < len(times) else idx

    for j in range(start, len(times)):
        ts = times[j]
        px = prices[j]
        hold_secs = (ts - entry_ts).total_seconds()
        if conf.at_sale > 0:
            if px >= target:
                return ts, target
            if hold_secs >= conf.hold_sec:
                return ts, px
        else:
            pnl_pct = (px / entry_price - 1.0) * 100.0 if entry_price > 0 else 0.0
            if pnl_pct >= conf.high or pnl_pct <= -conf.low or hold_secs >= conf.hold_sec:
                return ts, px
    return times[-1], prices[-1]


def _select_bond_events(
    *,
    trigger_df: pl.DataFrame,
    bond_sec: pl.DataFrame,
    stock_to_bonds: dict[str, list[str]],
    conf: TriggerConf,
) -> pl.DataFrame:
    if trigger_df.height == 0 or bond_sec.height == 0:
        return pl.DataFrame()

    map_rows: list[dict[str, str]] = []
    for s, bonds in stock_to_bonds.items():
        for b in bonds:
            map_rows.append({"stock_symbol": s, "bond_symbol": b})
    if not map_rows:
        return pl.DataFrame()
    map_df = pl.DataFrame(map_rows)

    bond_snap = bond_sec.select(
        pl.col("symbol").alias("bond_symbol"),
        "ts_sec",
        pl.col("ts_sec").dt.epoch("s").cast(pl.Int64).alias("ts_epoch"),
        pl.col("price").alias("bond_price"),
        "amt_per_sec_wan",
    )

    cands = (
        trigger_df.with_columns(
            pl.col("ts_sec").dt.epoch("s").cast(pl.Int64).alias("ts_epoch")
        )
        .join(map_df, on="stock_symbol", how="inner")
        .join_asof(
            bond_snap.sort(["bond_symbol", "ts_epoch"]),
            left_on="ts_epoch",
            right_on="ts_epoch",
            by="bond_symbol",
            strategy="backward",
        )
        .drop_nulls(["bond_price", "amt_per_sec_wan"])
        .filter(pl.col("amt_per_sec_wan") >= conf.bond_amt)
        .sort(["stock_symbol", "ts_sec", "amt_per_sec_wan"], descending=[False, False, True])
        .group_by(["stock_symbol", "ts_sec"], maintain_order=True)
        .first()
        .sort(["ts_sec", "stock_symbol"])
    )
    return cands


def _to_vbt_result(trades: list[Trade], initial_cash: float) -> dict[str, Any]:
    if not trades:
        return {
            "final_equity": initial_cash,
            "total_pnl": 0.0,
            "trade_count": 0,
            "win_count": 0,
            "win_rate": 0.0,
            "vectorbt": {
                "total_return": 0.0,
                "total_profit": 0.0,
                "max_drawdown": 0.0,
            },
        }

    syms = sorted({t.bond_symbol for t in trades})
    idx = sorted({t.entry_ts for t in trades} | {t.exit_ts for t in trades})

    close = pd.DataFrame(index=idx, columns=syms, dtype=float)
    size = pd.DataFrame(0.0, index=idx, columns=syms)

    for t in trades:
        close.loc[t.entry_ts, t.bond_symbol] = t.entry_price
        close.loc[t.exit_ts, t.bond_symbol] = t.exit_price
        size.loc[t.entry_ts, t.bond_symbol] += t.size
        size.loc[t.exit_ts, t.bond_symbol] -= t.size

    close = close.ffill().bfill()
    pf = vbt.Portfolio.from_orders(
        close=close,
        size=size,
        init_cash=initial_cash,
        cash_sharing=True,
        fees=0.0,
        slippage=0.0,
        freq="1s",
    )

    vectorbt_total_profit = float(pf.total_profit())
    total_return = float(pf.total_return())
    manual_total_profit = float(sum(t.pnl for t in trades))
    final_equity = float(initial_cash + manual_total_profit)
    wins = sum(1 for t in trades if t.pnl > 0)

    return {
        "final_equity": final_equity,
        "total_pnl": manual_total_profit,
        "trade_count": len(trades),
        "win_count": wins,
        "win_rate": wins / len(trades),
        "vectorbt": {
            "total_return": total_return,
            "total_profit": vectorbt_total_profit,
            "max_drawdown": float(pf.max_drawdown()),
        },
    }


def _resolve_workers(workers: int, total_jobs: int) -> int:
    if total_jobs <= 0:
        return 1
    cpu = os.cpu_count() or 1
    requested = cpu if workers <= 0 else workers
    return max(1, min(requested, total_jobs))


def _simulate_one_day(
    *,
    td: date,
    confs: list[TriggerConf],
    stock_to_bonds: dict[str, list[str]],
    shares_map: dict[str, float],
    stock_syms: tuple[str, ...],
    bond_syms: tuple[str, ...],
    client: Any,
) -> list[Trade]:
    stock_df = query_clickhouse_arrow_df(
        tick_day_rows_sql("tick_stock_cb_underlying"),
        parameters={"d": td, "syms": stock_syms},
        client=client,
    )
    bond_df = query_clickhouse_arrow_df(
        tick_day_rows_sql("tick_cb"),
        parameters={"d": td, "syms": bond_syms},
        client=client,
    )
    if stock_df.height == 0 or bond_df.height == 0:
        return []

    stock_sec = _aggregate_stock_second(stock_df, shares_map)
    bond_sec = _aggregate_bond_second(bond_df)
    if stock_sec.height == 0 or bond_sec.height == 0:
        return []

    bond_lookup = _build_bond_lookup(bond_sec)
    day_trades: list[Trade] = []

    for conf_idx, conf in enumerate(confs):
        trigger_df = _compute_stock_triggers(stock_sec, conf)
        if trigger_df.height == 0:
            continue
        events = _select_bond_events(
            trigger_df=trigger_df,
            bond_sec=bond_sec,
            stock_to_bonds=stock_to_bonds,
            conf=conf,
        )
        if events.height == 0:
            continue

        open_positions: list[dict[str, Any]] = []
        for ev in events.iter_rows(named=True):
            ts = ev["ts_sec"]
            stock_symbol = str(ev["stock_symbol"])
            bond_symbol = str(ev["bond_symbol"])
            bond_price = float(ev["bond_price"])

            # 仅用于同策略日内持仓槽位约束。
            open_positions = [p for p in open_positions if p["exit_ts"] > ts]

            if len(open_positions) >= conf.hold_cnt:
                continue
            if any(pos["bond_symbol"] == bond_symbol for pos in open_positions):
                continue

            entry_price = bond_price * (1.0 + conf.b_upper / 100.0)
            qty = _calc_vol(conf, entry_price)
            if qty <= 0:
                continue

            if bond_symbol not in bond_lookup:
                continue
            times, prices = bond_lookup[bond_symbol]
            exit_ts, exit_price = _find_exit_for_trade(
                conf=conf,
                times=times,
                prices=prices,
                entry_ts=ts,
                entry_price=entry_price,
            )

            day_trades.append(
                Trade(
                    conf_idx=conf_idx,
                    bond_symbol=bond_symbol,
                    stock_symbol=stock_symbol,
                    entry_ts=ts,
                    exit_ts=exit_ts,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    qty=qty,
                )
            )
            open_positions.append({"bond_symbol": bond_symbol, "exit_ts": exit_ts})
    return day_trades


def _apply_cash_constraint(trades: list[Trade], initial_cash: float) -> list[Trade]:
    if not trades:
        return []
    accepted: list[Trade] = []
    cash = float(initial_cash)
    open_heap: list[tuple[datetime, float]] = []

    for t in sorted(trades, key=lambda x: (x.entry_ts, x.exit_ts)):
        while open_heap and open_heap[0][0] <= t.entry_ts:
            _, proceeds = heappop(open_heap)
            cash += proceeds
        cost = t.entry_price * t.size
        if cost > cash:
            continue
        cash -= cost
        accepted.append(t)
        heappush(open_heap, (t.exit_ts, t.exit_price * t.size))
    return accepted


def _process_day_worker(
    td: date,
    confs: list[TriggerConf],
    stock_to_bonds: dict[str, list[str]],
    shares_map: dict[str, float],
    stock_syms: tuple[str, ...],
    bond_syms: tuple[str, ...],
) -> list[Trade]:
    client = create_clickhouse_http_client()
    return _simulate_one_day(
        td=td,
        confs=confs,
        stock_to_bonds=stock_to_bonds,
        shares_map=shares_map,
        stock_syms=stock_syms,
        bond_syms=bond_syms,
        client=client,
    )


def run_backtest(
    *,
    config_path: str | Path,
    cb_info_csv: str | Path,
    start_date: date,
    end_date: date,
    shares_json: str | Path | None = None,
    initial_cash: float = 1_000_000.0,
    workers: int = 0,
) -> dict[str, Any]:
    confs = _load_trigger_config(config_path)
    stock_to_bonds, _ = _load_cb_stock_map(cb_info_csv)
    shares_map = _load_stock_shares(shares_json)

    client = create_clickhouse_http_client()

    stock_syms = sorted(stock_to_bonds.keys())
    bond_syms = sorted({b for arr in stock_to_bonds.values() for b in arr})

    stock_dates_df = query_clickhouse_arrow_df(
        tick_trade_dates_sql("tick_stock_cb_underlying"),
        parameters={"sd": start_date, "ed": end_date},
        client=client,
    )
    bond_dates_df = query_clickhouse_arrow_df(
        tick_trade_dates_sql("tick_cb"),
        parameters={"sd": start_date, "ed": end_date},
        client=client,
    )
    stock_dates = set(stock_dates_df.get_column("trade_date").to_list()) if stock_dates_df.height else set()
    bond_dates = set(bond_dates_df.get_column("trade_date").to_list()) if bond_dates_df.height else set()
    trade_dates = sorted(stock_dates & bond_dates)

    worker_count = _resolve_workers(workers, len(trade_dates))

    candidate_trades: list[Trade] = []
    if worker_count == 1:
        for td in trade_dates:
            candidate_trades.extend(
                _simulate_one_day(
                    td=td,
                    confs=confs,
                    stock_to_bonds=stock_to_bonds,
                    shares_map=shares_map,
                    stock_syms=tuple(stock_syms),
                    bond_syms=tuple(bond_syms),
                    client=client,
                )
            )
    else:
        with cf.ThreadPoolExecutor(max_workers=worker_count) as pool:
            futures = [
                pool.submit(
                    _process_day_worker,
                    td,
                    confs,
                    stock_to_bonds,
                    shares_map,
                    tuple(stock_syms),
                    tuple(bond_syms),
                )
                for td in trade_dates
            ]
            for fut in cf.as_completed(futures):
                candidate_trades.extend(fut.result())

    trades = _apply_cash_constraint(candidate_trades, initial_cash)

    vbt_result = _to_vbt_result(trades, initial_cash)

    by_conf: dict[int, dict[str, Any]] = {}
    for i, conf in enumerate(confs):
        ts = [t for t in trades if t.conf_idx == i]
        pnl = sum(t.pnl for t in ts)
        wins = sum(1 for t in ts if t.pnl > 0)
        by_conf[i] = {
            "comment": conf.comment,
            "trade_count": len(ts),
            "win_count": wins,
            "win_rate": (wins / len(ts)) if ts else 0.0,
            "pnl": pnl,
        }

    return {
        "request_range": {"start": str(start_date), "end": str(end_date)},
        "actual_trade_dates": {
            "count": len(trade_dates),
            "start": str(trade_dates[0]) if trade_dates else None,
            "end": str(trade_dates[-1]) if trade_dates else None,
        },
        "parallel": {
            "workers": worker_count,
            "candidate_trade_count": len(candidate_trades),
        },
        "initial_cash": initial_cash,
        "final_equity": vbt_result["final_equity"],
        "total_pnl": vbt_result["total_pnl"],
        "trade_count": vbt_result["trade_count"],
        "win_count": vbt_result["win_count"],
        "win_rate": vbt_result["win_rate"],
        "vectorbt": vbt_result["vectorbt"],
        "by_conf": by_conf,
    }
