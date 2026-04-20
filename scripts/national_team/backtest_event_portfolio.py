"""
事件驱动弹性组合回测主链路

流程:
  1. detect_signal_dates() → 信号日列表
  2. ElasticScorer.score() → 每个信号日的个股弹性打分
  3. PortfolioBuilder.build() → 打分 → 加权组合
  4. ExitEngine.simulate() → 持有 & 退出 → Trade 列表
  5. 汇总: 事件级P&L、归因、多参数组合对比

输出:
  - doc/event_portfolio_report.md
  - data/event_portfolio_trades.parquet
"""

from __future__ import annotations

import itertools
import os
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import clickhouse_connect

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data_clients import create_clickhouse_http_client
from src.national_team.elastic_scorer import ElasticScorer
from src.national_team.portfolio_builder import PortfolioBuilder
from src.national_team.exit_engine import ExitEngine, ExitRule, Trade

# ── 复用 backtest_buy_elasticity 中的工具函数 ─────────────────────────────
from scripts.backtest_buy_elasticity import (
    detect_signal_dates,
    get_hs300_members,
    sym_local_to_tushare,
    sym_tushare_to_local,
    SIGNAL_THRESHOLD,
)

# ── 配置 ──────────────────────────────────────────────────────────────────

@dataclass
class RunConfig:
    """单次回测参数组合。"""
    top_n: int = 5
    weighting: str = "equal"
    exit_rules: tuple[ExitRule, ...] = (ExitRule.FIXED_5,)
    label: str = ""

    def __post_init__(self):
        if not self.label:
            rules_str = "+".join(r.value for r in self.exit_rules)
            self.label = f"top{self.top_n}_{self.weighting}_{rules_str}"


# ── 预定义实验网格 ───────────────────────────────────────────────────────

EXPERIMENT_GRID: list[RunConfig] = [
    RunConfig(top_n=3, weighting="equal", exit_rules=(ExitRule.FIXED_3,)),
    RunConfig(top_n=3, weighting="equal", exit_rules=(ExitRule.FIXED_5,)),
    RunConfig(top_n=3, weighting="equal", exit_rules=(ExitRule.FIXED_10,)),
    RunConfig(top_n=5, weighting="equal", exit_rules=(ExitRule.FIXED_3,)),
    RunConfig(top_n=5, weighting="equal", exit_rules=(ExitRule.FIXED_5,)),
    RunConfig(top_n=5, weighting="equal", exit_rules=(ExitRule.FIXED_10,)),
    RunConfig(top_n=5, weighting="score", exit_rules=(ExitRule.FIXED_5,)),
    RunConfig(top_n=8, weighting="equal", exit_rules=(ExitRule.FIXED_5,)),
    RunConfig(top_n=8, weighting="score", exit_rules=(ExitRule.FIXED_5,)),
    RunConfig(top_n=5, weighting="equal", exit_rules=(ExitRule.SELL_SIGNAL,)),
    RunConfig(top_n=5, weighting="equal", exit_rules=(ExitRule.RELATIVE_WEAKNESS,)),
]


def get_ch() -> clickhouse_connect.driver.Client:
    return create_clickhouse_http_client()


# ── 单事件回测 ────────────────────────────────────────────────────────────

def run_single_event(
    ch: clickhouse_connect.driver.Client,
    scorer: ElasticScorer,
    builder: PortfolioBuilder,
    engine: ExitEngine,
    signal_date: date,
    signal_prob: float,
) -> list[Trade]:
    """对单个事件日执行: 选股 → 构建组合 → 模拟退出。"""

    # 1. 获取候选池
    members = get_hs300_members(ch, signal_date)
    if not members:
        print(f"    {signal_date}: 无成分股，跳过")
        return []

    # 2. 弹性打分
    scored = scorer.score(ch, signal_date, members)
    if scored.height == 0:
        print(f"    {signal_date}: 打分为空，跳过")
        return []

    # 3. 构建组合
    portfolio = builder.build(scored)
    if portfolio.height == 0:
        print(f"    {signal_date}: 组合为空，跳过")
        return []

    # 4. 模拟退出
    trades = engine.simulate(ch, portfolio, signal_date)

    return trades


# ── 事件级汇总 ────────────────────────────────────────────────────────────

def summarize_event(trades: list[Trade], signal_date: date, signal_prob: float) -> dict:
    """汇总单事件的组合表现。"""
    if not trades:
        return {
            "signal_date": signal_date,
            "signal_prob": signal_prob,
            "n_stocks": 0,
            "portfolio_ret": None,
            "avg_holding_days": None,
            "best_stock": None,
            "worst_stock": None,
        }

    portfolio_ret = sum(t.contribution for t in trades)
    avg_hold = sum(t.holding_days for t in trades) / len(trades)
    best = max(trades, key=lambda t: t.ret)
    worst = min(trades, key=lambda t: t.ret)

    return {
        "signal_date": signal_date,
        "signal_prob": signal_prob,
        "n_stocks": len(trades),
        "portfolio_ret": portfolio_ret,
        "avg_holding_days": avg_hold,
        "best_stock": best.symbol,
        "best_ret": best.ret,
        "worst_stock": worst.symbol,
        "worst_ret": worst.ret,
        "avg_stock_ret": sum(t.ret for t in trades) / len(trades),
    }


# ── 多参数实验运行器 ─────────────────────────────────────────────────────

def run_experiment(
    ch: clickhouse_connect.driver.Client,
    config: RunConfig,
    signals: list[tuple[date, float]],
) -> pl.DataFrame:
    """对一组参数配置运行全部信号日回测。"""
    print(f"\n{'='*60}")
    print(f"实验: {config.label}")
    print(f"  top_n={config.top_n}, weighting={config.weighting}, exit={[r.value for r in config.exit_rules]}")
    print(f"{'='*60}")

    scorer = ElasticScorer()
    builder = PortfolioBuilder(top_n=config.top_n, weighting=config.weighting)
    engine = ExitEngine(rules=list(config.exit_rules))

    all_trades: list[Trade] = []
    event_summaries: list[dict] = []

    for sig_date, prob in signals:
        print(f"  {sig_date} (prob={prob:.2%}) ... ", end="", flush=True)
        trades = run_single_event(ch, scorer, builder, engine, sig_date, prob)
        all_trades.extend(trades)

        summary = summarize_event(trades, sig_date, prob)
        event_summaries.append(summary)

        if trades:
            pret = sum(t.contribution for t in trades)
            print(f"{len(trades)} stocks, ret={pret:+.2%}")
        else:
            print("skip")

    if not event_summaries:
        return pl.DataFrame()

    df = pl.DataFrame(event_summaries)
    return df


def compile_trade_records(trades: list[Trade], config_label: str) -> pl.DataFrame:
    """将 Trade 列表转为 DataFrame。"""
    if not trades:
        return pl.DataFrame()
    rows = [
        {
            "config": config_label,
            "symbol": t.symbol,
            "weight": t.weight,
            "entry_date": t.entry_date,
            "entry_price": t.entry_price,
            "exit_date": t.exit_date,
            "exit_price": t.exit_price,
            "exit_rule": t.exit_rule,
            "holding_days": t.holding_days,
            "ret": t.ret,
            "contribution": t.contribution,
        }
        for t in trades
    ]
    return pl.DataFrame(rows)


# ── 报告生成 ──────────────────────────────────────────────────────────────

def generate_report(
    experiment_results: dict[str, pl.DataFrame],
    all_trades_df: pl.DataFrame,
    signals: list[tuple[date, float]],
    output_path: Path,
) -> None:
    """生成 Markdown 归因报告。"""
    lines = [
        "# 事件驱动弹性组合回测报告",
        "",
        f"- **信号因子**: NT_Buy_Prob v2 (threshold > {SIGNAL_THRESHOLD:.0%})",
        f"- **信号日数**: {len(signals)}",
        f"- **实验组数**: {len(experiment_results)}",
        "",
    ]

    # ── 实验对比总览 ─────────────────────────────────────────────────────
    lines.append("## 实验对比总览")
    lines.append("")
    lines.append("| 配置 | 事件数 | 有效 | 平均收益 | 胜率 | sum_ret | compound_ret | ann_ret | max_dd | 持仓天 |")
    lines.append("|------|--------|------|---------|------|---------|-------------|---------|--------|--------|")

    for label, df in experiment_results.items():
        if df.height == 0:
            lines.append(f"| {label} | 0 | 0 | - | - | - | - | - | - | - |")
            continue

        valid = df.filter(pl.col("portfolio_ret").is_not_null())
        n_events = df.height
        n_valid = valid.height
        if n_valid == 0:
            lines.append(f"| {label} | {n_events} | 0 | - | - | - | - | - | - | - |")
            continue

        rets = valid["portfolio_ret"]
        mean_ret = rets.mean()
        win_rate = valid.filter(pl.col("portfolio_ret") > 0).height / n_valid
        sum_ret = float(rets.sum())
        compound_ret = float((1 + rets).product() - 1)
        avg_hold = valid["avg_holding_days"].mean()

        # Annualized: assume ~avg_hold days per event, n_valid events
        total_days = float(avg_hold * n_valid) if avg_hold else n_valid * 5
        total_years = total_days / 252.0
        ann_ret = ((1 + compound_ret) ** (1.0 / total_years) - 1) if total_years > 0 and compound_ret > -1 else 0.0

        # Max drawdown on equity curve
        equity = (1 + rets).cum_prod()
        peak = equity.cum_max()
        dd = (equity - peak) / peak
        max_dd = float(dd.min())

        lines.append(
            f"| {label} | {n_events} | {n_valid} "
            f"| {mean_ret:+.2%} | {win_rate:.0%} "
            f"| {sum_ret:+.2%} | {compound_ret:+.2%} "
            f"| {ann_ret:+.1%} | {max_dd:+.1%} "
            f"| {avg_hold:.1f} |"
        )

    lines.append("")

    # ── Alpha 时间结构分析 ────────────────────────────────────────────────
    lines.append("## Alpha 时间结构 (fixed_3 vs fixed_5 vs fixed_10)")
    lines.append("")
    lines.append("回答: alpha 主要来自事件后的 1-3天，还是 5-10天？")
    lines.append("")

    time_configs = ["top5_equal_fixed_3", "top5_equal_fixed_5", "top5_equal_fixed_10"]
    for label in time_configs:
        if label in experiment_results:
            df = experiment_results[label]
            valid = df.filter(pl.col("portfolio_ret").is_not_null())
            if valid.height > 0:
                mean_ret = valid["portfolio_ret"].mean()
                lines.append(f"- **{label}**: 平均事件收益 = {mean_ret:+.2%}")
    lines.append("")

    # ── 最佳/最差事件 ────────────────────────────────────────────────────
    # 用默认配置 (top5_equal_fixed_5) 显示
    default_label = "top5_equal_fixed_5"
    if default_label in experiment_results:
        df = experiment_results[default_label]
        valid = df.filter(pl.col("portfolio_ret").is_not_null())
        if valid.height > 0:
            lines.append(f"## 事件级明细 ({default_label})")
            lines.append("")
            lines.append("| 信号日 | 概率 | 股票数 | 组合收益 | 平均个股收益 | 最强 | 最弱 |")
            lines.append("|--------|------|--------|---------|-------------|------|------|")

            for row in valid.sort("signal_date").iter_rows(named=True):
                best_info = f"{row.get('best_stock', '-')} ({row.get('best_ret', 0):+.1%})" if row.get("best_stock") else "-"
                worst_info = f"{row.get('worst_stock', '-')} ({row.get('worst_ret', 0):+.1%})" if row.get("worst_stock") else "-"
                lines.append(
                    f"| {row['signal_date']} | {row['signal_prob']:.0%} "
                    f"| {row['n_stocks']} | {row['portfolio_ret']:+.2%} "
                    f"| {row.get('avg_stock_ret', 0):+.2%} "
                    f"| {best_info} | {worst_info} |"
                )
            lines.append("")

    # ── 退出规则对比 ─────────────────────────────────────────────────────
    lines.append("## 退出规则对比 (top5, equal)")
    lines.append("")
    exit_configs = [
        "top5_equal_fixed_5",
        "top5_equal_sell_signal",
        "top5_equal_relative_weakness",
    ]
    for label in exit_configs:
        if label in experiment_results:
            df = experiment_results[label]
            valid = df.filter(pl.col("portfolio_ret").is_not_null())
            if valid.height > 0:
                mean_ret = valid["portfolio_ret"].mean()
                win_rate = valid.filter(pl.col("portfolio_ret") > 0).height / valid.height
                lines.append(f"- **{label}**: mean={mean_ret:+.2%}, win_rate={win_rate:.0%}")
    lines.append("")

    # ── 权重方式对比 ─────────────────────────────────────────────────────
    lines.append("## 权重方式对比 (equal vs score)")
    lines.append("")
    weight_configs = ["top5_equal_fixed_5", "top5_score_fixed_5"]
    for label in weight_configs:
        if label in experiment_results:
            df = experiment_results[label]
            valid = df.filter(pl.col("portfolio_ret").is_not_null())
            if valid.height > 0:
                mean_ret = valid["portfolio_ret"].mean()
                lines.append(f"- **{label}**: mean={mean_ret:+.2%}")
    lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n报告已写入: {output_path}")


# ── 主流程 ────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("事件驱动弹性组合回测")
    print("=" * 60)

    # Step 1: 检测信号日
    print(f"\n[Step 1] 检测信号日 (prob > {SIGNAL_THRESHOLD:.0%}) ...")
    signals = detect_signal_dates()
    print(f"  → 共 {len(signals)} 个信号日")

    if not signals:
        print("No signal dates found!")
        return

    ch = get_ch()

    # Step 2: 运行实验网格
    print(f"\n[Step 2] 运行 {len(EXPERIMENT_GRID)} 组实验 ...")

    experiment_results: dict[str, pl.DataFrame] = {}
    all_trade_dfs: list[pl.DataFrame] = []

    for config in EXPERIMENT_GRID:
        event_df = run_experiment(ch, config, signals)
        experiment_results[config.label] = event_df

    # Step 3: 收集全部交易明细 (用默认配置再跑一遍取 trades)
    print(f"\n[Step 3] 收集交易明细 ...")
    default_config = RunConfig(top_n=5, weighting="equal", exit_rules=(ExitRule.FIXED_5,))
    scorer = ElasticScorer()
    builder = PortfolioBuilder(top_n=default_config.top_n, weighting=default_config.weighting)
    engine = ExitEngine(rules=list(default_config.exit_rules))

    all_trades: list[Trade] = []
    for sig_date, prob in signals:
        trades = run_single_event(ch, scorer, builder, engine, sig_date, prob)
        all_trades.extend(trades)

    all_trades_df = compile_trade_records(all_trades, default_config.label)

    # 保存交易明细
    root = Path(__file__).resolve().parents[2]
    if all_trades_df.height > 0:
        parquet_path = root / "data" / "event_portfolio_trades.parquet"
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        all_trades_df.write_parquet(parquet_path)
        print(f"  交易明细已保存: {parquet_path} ({all_trades_df.height} 条)")

    # Step 4: 生成报告
    print(f"\n[Step 4] 生成报告 ...")
    report_path = root / "doc" / "event_portfolio_report.md"
    generate_report(experiment_results, all_trades_df, signals, report_path)

    print("\n完成!")


if __name__ == "__main__":
    main()
