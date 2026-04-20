"""
评估 300ETF 概率信号策略的固定止损方案。

维度:
  - prob 阈值: >10%, >20%, >35%
  - 止盈: 10%, 20%, 30%
  - 止损: 5%, 8%, 10%, 12%
  - 样本: 2016+

输出:
  - stdout 对比表
"""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path
from typing import Any, cast

import pandas as pd
import vectorbt as vbt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.backtest_vbt_tp_matrix import (
    END_DATE,
    ETF_SYMBOL,
    INITIAL_CASH,
    PROB_THRESHOLDS,
    START_DATE,
    TP_TARGETS,
    get_ch,
    load_etf_daily,
    load_signal_dates,
)

SL_TARGETS = [0.05, 0.08, 0.10, 0.12]


def run_combo(
    price_df: pd.DataFrame,
    signal_dates: list[tuple],
    prob_threshold: float,
    tp_pct: float,
    sl_pct: float | None,
) -> dict[str, Any]:
    valid_signals = [d for d, p in signal_dates if p > prob_threshold]
    trade_dates = price_df.index

    entry_dates: set[pd.Timestamp] = set()
    for signal_date in valid_signals:
        signal_ts = pd.Timestamp(signal_date)
        future = trade_dates[trade_dates > signal_ts]
        if len(future) > 0:
            entry_ts = pd.Timestamp(cast(Any, future[0]))
            if not pd.isna(entry_ts):
                entry_dates.add(cast(pd.Timestamp, entry_ts))

    entries = pd.Series(False, index=price_df.index)
    for entry_date in entry_dates:
        if entry_date in entries.index:
            entries.loc[entry_date] = True

    pf = vbt.Portfolio.from_signals(
        close=price_df["close"],
        open=price_df["open"],
        high=price_df["high"],
        low=price_df["low"],
        entries=entries,
        exits=pd.Series(False, index=price_df.index),
        tp_stop=tp_pct,
        sl_stop=sl_pct,
        size=1.0,
        size_type="percent",
        price=price_df["open"],
        init_cash=INITIAL_CASH,
        fees=0.001,
        slippage=0.001,
        accumulate=False,
        freq="1D",
    )

    trades = cast(pd.DataFrame, getattr(pf.trades, "records_readable"))
    stats = cast(Any, pf.stats())
    open_trades = int(getattr(getattr(cast(Any, pf.trades), "open"), "count")())

    return {
        "prob": prob_threshold,
        "tp": tp_pct,
        "sl": sl_pct,
        "total_return": float(cast(Any, pf.total_return())),
        "final_value": float(cast(Any, pf.final_value())),
        "max_dd": float(getattr(pf, "max_drawdown")()),
        "sharpe": float(stats.get("Sharpe Ratio", 0.0) or 0.0),
        "trade_count": len(trades),
        "win_rate": float((trades["PnL"] > 0).mean()) if len(trades) > 0 else 0.0,
        "open_trades": open_trades,
    }


def build_results_df() -> pd.DataFrame:
    signals = [(d, p) for d, p in load_signal_dates() if d >= START_DATE]
    etf = load_etf_daily(get_ch(), ETF_SYMBOL, START_DATE - timedelta(days=30), END_DATE)

    rows: list[dict[str, Any]] = []
    for prob in PROB_THRESHOLDS:
        for tp in TP_TARGETS:
            rows.append(run_combo(etf, signals, prob, tp, None))
            for sl in SL_TARGETS:
                rows.append(run_combo(etf, signals, prob, tp, sl))

    df = pd.DataFrame(rows)
    df["label"] = (
        "p>" + (df["prob"] * 100).round(0).astype(int).astype(str) + "%"
        + " tp" + (df["tp"] * 100).round(0).astype(int).astype(str) + "%"
        + " sl" + df["sl"].map(lambda x: "none" if pd.isna(x) else f"{int(round(x * 100))}%")
    )
    return df.sort_values(["prob", "tp", "sl"], na_position="first").reset_index(drop=True)


def _fmt_pct(value: float) -> str:
    return f"{value:+.2%}"


def _fmt_dd(value: float) -> str:
    return f"{value:.1%}"


def write_report(df: pd.DataFrame) -> None:
    matrix_path = ROOT / "data" / "basket_baseline" / "vbt_tp_sl_matrix.csv"
    report_path = ROOT / "doc" / "vbt_stoploss_matrix_report.md"
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(matrix_path, index=False)

    compare = cast(pd.DataFrame, df[df["sl"].isna() | (df["sl"] == 0.08)].copy())
    sl_series = cast(pd.Series, compare["sl"])
    compare["sl_label"] = cast(Any, sl_series).apply(
        lambda x: "无止损" if pd.isna(x) else f"{int(round(float(x) * 100))}%"
    )
    compare = cast(pd.DataFrame, cast(Any, compare).sort_values(["prob", "tp", "sl"], na_position="first"))

    scored_all = cast(
        pd.DataFrame,
        df.assign(rr=df["total_return"] / df["max_dd"].abs()),
    )
    best_return = cast(dict[str, Any], cast(Any, df.sort_values("total_return", ascending=False).iloc[0]).to_dict())
    best_rr = cast(
        dict[str, Any],
        cast(Any, scored_all.sort_values(["rr", "total_return"], ascending=[False, False]).iloc[0]).to_dict(),
    )
    recommended = cast(
        dict[str, Any],
        cast(Any, df[(df["prob"] == 0.20) & (df["tp"] == 0.30) & (df["sl"] == 0.08)].iloc[0]).to_dict(),
    )
    strong_signal_stop = cast(
        dict[str, Any],
        cast(Any, df[(df["prob"] == 0.35) & (df["tp"] == 0.30) & (df["sl"] == 0.08)].iloc[0]).to_dict(),
    )

    lines = [
        "# 300ETF 止损敏感性报告",
        "",
        "- 标的: 510300.SH",
        f"- 区间: {START_DATE} ~ {END_DATE}",
        f"- 初始资金: {INITIAL_CASH:,.0f}",
        "- 信号: 国家队买入概率触发后 T+1 开盘全仓买入",
        "- 止盈候选: 10% / 20% / 30%",
        "- 止损候选: 无 / 5% / 8% / 10% / 12%",
        "",
        "## 结论",
        "",
        f"- 若追求**绝对收益最大化**，最佳组合仍是 **{best_return['label']}**，总收益 {_fmt_pct(float(best_return['total_return']))}，最大回撤 {_fmt_dd(float(best_return['max_dd']))}。",
        f"- 若需要一条**通用固定止损**，推荐先从 **8%** 开始。代表性组合 **{recommended['label']}** 的总收益 {_fmt_pct(float(recommended['total_return']))}，最大回撤 {_fmt_dd(float(recommended['max_dd']))}，且无未平仓。",
        f"- 若只做**强信号**并且实盘必须带止损，优先看 **{strong_signal_stop['label']}**：总收益 {_fmt_pct(float(strong_signal_stop['total_return']))}，最大回撤 {_fmt_dd(float(strong_signal_stop['max_dd']))}，Sharpe {float(strong_signal_stop['sharpe']):.2f}。",
        f"- 按收益/回撤比排序，最佳风险收益组合是 **{best_rr['label']}**，RR={float(best_rr['rr']):.2f}。",
        "",
        "## 无止损 vs 8% 止损",
        "",
        "| 组合 | 止损 | 总收益 | 最终资产 | 最大回撤 | Sharpe | 交易数 | 未平仓 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in cast(list[dict[str, Any]], compare.to_dict(orient="records")):
        lines.append(
            f"| p>{int(round(float(row['prob']) * 100))}% / TP{int(round(float(row['tp']) * 100))}% "
            f"| {row['sl_label']} "
            f"| {_fmt_pct(float(row['total_return']))} "
            f"| {float(row['final_value']):,.0f} "
            f"| {_fmt_dd(float(row['max_dd']))} "
            f"| {float(row['sharpe']):.2f} "
            f"| {int(row['trade_count'])} "
            f"| {int(row['open_trades'])} |"
        )

    lines.extend([
        "",
        "## Top 10 收益/回撤比",
        "",
        "| 组合 | RR | 总收益 | 最大回撤 | Sharpe | 交易数 | 未平仓 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])

    scored = cast(
        pd.DataFrame,
        cast(Any, scored_all).sort_values(["rr", "total_return"], ascending=[False, False]).head(10),
    )
    for row in cast(list[dict[str, Any]], scored.to_dict(orient="records")):
        lines.append(
            f"| {row['label']} | {float(row['rr']):.2f} "
            f"| {_fmt_pct(float(row['total_return']))} "
            f"| {_fmt_dd(float(row['max_dd']))} "
            f"| {float(row['sharpe']):.2f} "
            f"| {int(row['trade_count'])} "
            f"| {int(row['open_trades'])} |"
        )

    lines.extend([
        "",
        "## 备注",
        "",
        "- 这里是日线 OHLC 级别的 stop/take-profit 模拟，真实盘中跳空会让实际止损价格更差。",
        "- `prob > 35%` 的样本只有 3 次，统计稳定性弱于低阈值组合。",
        f"- 明细矩阵已保存: `{matrix_path.relative_to(ROOT)}`",
        "",
    ])

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    df = build_results_df()

    print(df[[
        "label", "total_return", "final_value", "max_dd",
        "sharpe", "trade_count", "win_rate", "open_trades",
    ]].round(4).to_string(index=False))

    print("\nTop 10 by return / |max_dd|:")
    scored = df.assign(rr=df["total_return"] / df["max_dd"].abs().replace(0, pd.NA))
    scored = scored.sort_values(["rr", "total_return"], ascending=[False, False]).head(10)
    print(scored[[
        "label", "rr", "total_return", "max_dd", "sharpe", "trade_count", "open_trades",
    ]].round(4).to_string(index=False))

    write_report(df)
    print(f"\nSaved report: {ROOT / 'doc' / 'vbt_stoploss_matrix_report.md'}")
    print(f"Saved matrix: {ROOT / 'data' / 'basket_baseline' / 'vbt_tp_sl_matrix.csv'}")


if __name__ == "__main__":
    main()
