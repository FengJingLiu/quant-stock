"""
vectorbt 回测: 不同概率信号强度 × 止盈目标 全仓买入 300ETF

维度:
  - prob 阈值: >10%, >20%, >35%  (信号强度分层)
  - 止盈目标: 10%, 20%, 30%
  - 全仓 (100% 资金)
  - 本金 20 万
  - 时间: 2016+

策略: 信号触发次日开盘全仓买入 510300, 达到止盈目标卖出
"""

from __future__ import annotations

import os
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import vectorbt as vbt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data_clients import create_clickhouse_http_client, query_clickhouse_arrow_df
from src.data_queries import etf_daily_ohlcv_sql

ROOT = Path(__file__).resolve().parents[2]

# ── 参数 ──────────────────────────────────────────────────────────────────

PROB_THRESHOLDS = [0.10, 0.20, 0.35]
TP_TARGETS = [0.10, 0.20, 0.30]
INITIAL_CASH = 200_000.0
START_DATE = date(2016, 1, 1)
END_DATE = date(2026, 4, 16)
ETF_SYMBOL = "510300.SH"

# ── 数据获取 ──────────────────────────────────────────────────────────────


def get_ch() -> Any:
    return create_clickhouse_http_client()


def load_etf_daily(ch: Any, symbol: str, start: date, end: date) -> pd.DataFrame:
    """从 klines_1m_etf 聚合日线数据."""
    df = query_clickhouse_arrow_df(
        etf_daily_ohlcv_sql(),
        parameters={"sym": symbol, "sd": start, "ed": end},
        client=ch,
    )
    df = df.to_pandas()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.set_index("trade_date").sort_index()
    for c in ["open", "high", "low", "close"]:
        df[c] = df[c].astype(float)
    df["volume"] = df["volume"].astype(float)
    return df


def load_signal_dates() -> list[tuple[date, float]]:
    """从已有 events.parquet 读取信号日，避免重复计算."""
    import pyarrow.parquet as pq

    p = ROOT / "data" / "basket_baseline" / "events.parquet"
    if p.exists():
        t = pq.read_table(p)
        df = t.to_pandas()
        pairs = []
        for _, row in df.iterrows():
            sd = row["signal_date"]
            if isinstance(sd, pd.Timestamp):
                sd = sd.date()
            elif isinstance(sd, np.datetime64):
                sd = pd.Timestamp(sd).date()
            pairs.append((sd, float(row["signal_prob"])))
        return sorted(set(pairs))
    else:
        # fallback: 重新计算
        from scripts.backtest_buy_elasticity import detect_signal_dates
        return detect_signal_dates()


# ── 回测核心 ──────────────────────────────────────────────────────────────


def run_tp_backtest(
    price_df: pd.DataFrame,
    signal_dates: list[tuple[date, float]],
    prob_threshold: float,
    tp_pct: float,
) -> dict[str, Any]:
    """
    用 vectorbt 执行单组 (prob_threshold, tp_pct) 回测.

    策略:
      - 筛选 prob > prob_threshold 的信号日
      - 信号日 T+1 开盘买入 (全仓)
      - 持有直到价格达到 entry_price * (1+tp_pct) 卖出
      - 如果持仓期间再出现新信号, 忽略 (已持仓)
      - 无止损 (持有到止盈或一直持有)
    """
    # 筛选符合条件的信号日
    valid_signals = [d for d, p in signal_dates if p > prob_threshold]
    if not valid_signals:
        return {
            "prob_threshold": prob_threshold,
            "tp_pct": tp_pct,
            "n_signals": 0,
            "total_return": 0,
            "final_value": INITIAL_CASH,
        }

    # 入场日 = 信号日 T+1 (在 price_df 的交易日历上)
    trade_dates = price_df.index
    entry_dates = set()
    for sd in valid_signals:
        sd_ts = pd.Timestamp(sd)
        # 找 T+1: 信号日之后的下一个交易日
        future = trade_dates[trade_dates > sd_ts]
        if len(future) > 0:
            entry_dates.add(future[0])

    # 构建 entries 信号
    entries = pd.Series(False, index=price_df.index)
    for ed in entry_dates:
        if ed in entries.index:
            entries.loc[ed] = True

    n_entries = entries.sum()
    if n_entries == 0:
        return {
            "prob_threshold": prob_threshold,
            "tp_pct": tp_pct,
            "n_signals": 0,
            "total_return": 0,
            "final_value": INITIAL_CASH,
        }

    # 用 vectorbt from_signals + tp_stop
    pf = vbt.Portfolio.from_signals(
        close=price_df["close"],
        open=price_df["open"],
        high=price_df["high"],
        low=price_df["low"],
        entries=entries,
        exits=pd.Series(False, index=price_df.index),  # 不用手动退出
        tp_stop=tp_pct,
        size=1.0,           # 全仓
        size_type="percent",
        price=price_df["open"],  # 以开盘价入场
        init_cash=INITIAL_CASH,
        fees=0.001,         # 手续费万分之十 (比较保守)
        slippage=0.001,     # 滑点
        accumulate=False,   # 不累加仓位
        freq="1D",
    )

    # 提取关键指标
    total_return = float(cast(Any, pf.total_return()))
    final_value = float(cast(Any, pf.final_value()))

    stats = cast(Any, pf.stats())

    # 交易记录
    trades = cast(pd.DataFrame, getattr(pf.trades, "records_readable"))
    n_trades = len(trades)
    win_rate = 0.0
    avg_pnl = 0.0
    if n_trades > 0:
        won = trades[trades["PnL"] > 0]
        win_rate = len(won) / n_trades
        avg_pnl = float(cast(Any, trades["PnL"].mean()))

    # 净值曲线
    equity = cast(pd.Series, pf.value())
    max_drawdown = float(getattr(pf, "max_drawdown")())
    sharpe = float(stats.get("Sharpe Ratio", 0.0) or 0.0)

    return {
        "prob_threshold": prob_threshold,
        "tp_pct": tp_pct,
        "n_signals": int(n_entries),
        "n_trades": n_trades,
        "total_return": total_return,
        "final_value": final_value,
        "win_rate": win_rate,
        "avg_pnl": avg_pnl,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
        "equity_curve": equity,
        "portfolio": pf,
    }


# ── 主流程 ────────────────────────────────────────────────────────────────


def main():
    print("=" * 70)
    print("  vectorbt 回测: 概率信号强度 × 止盈目标 → 300ETF 全仓")
    print(f"  本金: {INITIAL_CASH:,.0f}  |  区间: {START_DATE} ~ {END_DATE}")
    print("=" * 70)

    # Step 1: 加载信号日
    print("\n[1] 加载信号日 ...")
    signals = load_signal_dates()
    # 过滤 2016+
    signals = [(d, p) for d, p in signals if d >= START_DATE]
    print(f"  → 2016+ 信号: {len(signals)} 个")
    for d, p in signals[:5]:
        print(f"     {d}  prob={p:.2%}")
    if len(signals) > 5:
        print(f"     ... ({len(signals) - 5} more)")

    # Step 2: 加载 ETF 日线
    print("\n[2] 加载 300ETF 日线 ...")
    ch = get_ch()
    etf = load_etf_daily(ch, ETF_SYMBOL, START_DATE - timedelta(days=30), END_DATE)
    index_values = etf.index.to_list()
    first_trade = pd.Timestamp(index_values[0]).date()
    last_trade = pd.Timestamp(index_values[-1]).date()
    print(f"  → {len(etf)} 交易日, {first_trade} ~ {last_trade}")
    print(f"  → 价格区间: {etf['close'].min():.3f} ~ {etf['close'].max():.3f}")

    # Step 3: 网格回测
    print("\n[3] 网格回测 ...")
    results = []
    for prob_th in PROB_THRESHOLDS:
        for tp in TP_TARGETS:
            print(f"\n  prob>{prob_th:.0%} × TP={tp:.0%} ...", end=" ", flush=True)
            r = run_tp_backtest(etf, signals, prob_th, tp)
            results.append(r)
            print(
                f"trades={r['n_trades']}, "
                f"return={r['total_return']:+.2%}, "
                f"final={r['final_value']:,.0f}, "
                f"win={r['win_rate']:.0%}, "
                f"maxDD={r['max_drawdown']:.1%}"
            )

    # Step 4: 汇总表
    print("\n" + "=" * 70)
    print("  汇总结果矩阵")
    print("=" * 70)

    # 收益率矩阵
    print("\n📊 总收益率 (Total Return)")
    print(f"{'prob \\ TP':>12}", end="")
    for tp in TP_TARGETS:
        print(f"  | TP={tp:.0%}  ", end="")
    print()
    print("-" * 50)
    for prob_th in PROB_THRESHOLDS:
        print(f"  >{prob_th:.0%}     ", end="")
        for tp in TP_TARGETS:
            r = next(
                x for x in results
                if x["prob_threshold"] == prob_th and x["tp_pct"] == tp
            )
            print(f"  | {r['total_return']:+7.2%} ", end="")
        print()

    # 最终资产矩阵
    print(f"\n💰 最终资产 (本金 {INITIAL_CASH:,.0f})")
    print(f"{'prob \\ TP':>12}", end="")
    for tp in TP_TARGETS:
        print(f"  | TP={tp:.0%}    ", end="")
    print()
    print("-" * 55)
    for prob_th in PROB_THRESHOLDS:
        print(f"  >{prob_th:.0%}     ", end="")
        for tp in TP_TARGETS:
            r = next(
                x for x in results
                if x["prob_threshold"] == prob_th and x["tp_pct"] == tp
            )
            print(f"  | {r['final_value']:>9,.0f} ", end="")
        print()

    # 交易统计矩阵
    print(f"\n📈 交易次数 / 胜率 / 最大回撤")
    print(f"{'prob \\ TP':>12}", end="")
    for tp in TP_TARGETS:
        print(f"  | TP={tp:.0%}          ", end="")
    print()
    print("-" * 65)
    for prob_th in PROB_THRESHOLDS:
        print(f"  >{prob_th:.0%}     ", end="")
        for tp in TP_TARGETS:
            r = next(
                x for x in results
                if x["prob_threshold"] == prob_th and x["tp_pct"] == tp
            )
            print(f"  | {r['n_trades']:2d}t/{r['win_rate']:.0%}/{r['max_drawdown']:.1%}", end=" ")
        print()

    # Step 5: 生成报告
    report_path = ROOT / "doc" / "vbt_tp_matrix_report.md"
    _generate_report(results, signals, etf, report_path)
    print(f"\n报告已写入: {report_path}")

    # Step 6: 保存净值曲线
    eq_path = ROOT / "data" / "basket_baseline" / "vbt_tp_equity.parquet"
    _save_equity_curves(results, eq_path)
    print(f"净值曲线已保存: {eq_path}")

    print(f"\n{'='*70}")
    print("  完成!")
    print(f"{'='*70}")


def _generate_report(results, signals, etf, output_path: Path):
    lines = [
        "# vectorbt 回测: 概率信号 × 止盈目标 → 300ETF 全仓",
        "",
        f"- **标的**: 510300.SH (沪深300ETF)",
        f"- **本金**: {INITIAL_CASH:,.0f}",
        f"- **区间**: {START_DATE} ~ {END_DATE}",
        f"- **信号源**: 国家队买入概率 (NT_Buy_Prob)",
        f"- **2016+ 信号数**: {len(signals)}",
        f"- **入场**: 信号触发 T+1 开盘全仓买入",
        f"- **出场**: 达到止盈目标卖出 (无止损)",
        f"- **费用**: 手续费 0.1% + 滑点 0.1%",
        "",
        "## 总收益率矩阵",
        "",
    ]

    # 收益率表
    header = "| prob 阈值 |"
    sep = "|-----------|"
    for tp in TP_TARGETS:
        header += f" TP={tp:.0%} |"
        sep += "--------|"
    lines.append(header)
    lines.append(sep)

    for prob_th in PROB_THRESHOLDS:
        row = f"| >{prob_th:.0%} |"
        for tp in TP_TARGETS:
            r = next(
                x for x in results
                if x["prob_threshold"] == prob_th and x["tp_pct"] == tp
            )
            row += f" {r['total_return']:+.2%} |"
        lines.append(row)

    lines.append("")
    lines.append("## 最终资产矩阵")
    lines.append("")

    header = "| prob 阈值 |"
    sep = "|-----------|"
    for tp in TP_TARGETS:
        header += f" TP={tp:.0%} |"
        sep += "--------|"
    lines.append(header)
    lines.append(sep)

    for prob_th in PROB_THRESHOLDS:
        row = f"| >{prob_th:.0%} |"
        for tp in TP_TARGETS:
            r = next(
                x for x in results
                if x["prob_threshold"] == prob_th and x["tp_pct"] == tp
            )
            row += f" {r['final_value']:,.0f} |"
        lines.append(row)

    lines.append("")
    lines.append("## 详细指标")
    lines.append("")
    lines.append("| prob | TP | 信号数 | 交易数 | 总收益 | 最终资产 | 胜率 | 均PnL | 最大回撤 | Sharpe |")
    lines.append("|------|-----|--------|--------|--------|----------|------|-------|----------|--------|")
    for r in results:
        lines.append(
            f"| >{r['prob_threshold']:.0%} | {r['tp_pct']:.0%} "
            f"| {r['n_signals']} | {r['n_trades']} "
            f"| {r['total_return']:+.2%} | {r['final_value']:,.0f} "
            f"| {r['win_rate']:.0%} | {r['avg_pnl']:+,.0f} "
            f"| {r['max_drawdown']:.1%} | {r['sharpe']:.2f} |"
        )

    lines.append("")
    lines.append("## 信号日明细 (2016+)")
    lines.append("")
    lines.append("| 日期 | 概率 |")
    lines.append("|------|------|")
    for d, p in signals:
        lines.append(f"| {d} | {p:.1%} |")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _save_equity_curves(results, output_path: Path):
    eq_dict = {}
    for r in results:
        if "equity_curve" in r and r["equity_curve"] is not None:
            label = f"p{r['prob_threshold']:.0%}_tp{r['tp_pct']:.0%}"
            eq_dict[label] = r["equity_curve"]
    if eq_dict:
        eq_df = pd.DataFrame(eq_dict)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        eq_df.to_parquet(output_path)


if __name__ == "__main__":
    main()
