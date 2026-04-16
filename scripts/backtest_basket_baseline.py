"""
国家队事件篮子策略基线

核心逻辑:
  - 信号触发 (prob > threshold)
  - T1 开盘买入 HS300 全篮子等权 (或权重前 TopN)
  - 固定持有 3 天
  - prob 分层控制仓位 (0% / 30% / 60% / 100%)

输出:
  - 净值曲线 (按时间顺序复利)
  - 分期统计 (全量 / 2016+ / 2020+ / 2023+)
  - 事件平均收益 / IR / t-stat / bootstrap CI
  - doc/basket_baseline_report.md
  - data/basket_baseline/equity.parquet
"""

from __future__ import annotations

import math
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import clickhouse_connect

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.backtest_buy_elasticity import (
    detect_signal_dates,
    get_hs300_members,
    get_stock_daily_agg,
    get_adj_factors,
    sym_local_to_tushare,
    sym_tushare_to_local,
    SIGNAL_THRESHOLD,
)
from src.config import CH_HTTP_KWARGS

ROOT = Path(__file__).resolve().parent.parent

# ── 仓位分层 ──────────────────────────────────────────────────────────────

POSITION_TIERS = [
    (0.10, 0.0),   # prob <= 10%: 不做
    (0.20, 0.30),  # 10% < prob <= 20%: 轻仓
    (0.35, 0.60),  # 20% < prob <= 35%: 中仓
    (1.00, 1.00),  # > 35%: 重仓
]

HOLD_DAYS = 3


def position_size(prob: float) -> float:
    for threshold, size in POSITION_TIERS:
        if prob <= threshold:
            return size
    return 1.0


# ── ClickHouse ────────────────────────────────────────────────────────────


def get_ch() -> clickhouse_connect.driver.Client:
    return clickhouse_connect.get_client(**CH_HTTP_KWARGS)


# ── Helpers ───────────────────────────────────────────────────────────────


def get_trading_dates(ch, start_date: date, end_date: date) -> list[date]:
    r = ch.query(
        """
        SELECT DISTINCT trade_date FROM klines_1m_etf
        WHERE symbol = '510300.SH'
          AND trade_date BETWEEN %(sd)s AND %(ed)s
        ORDER BY trade_date
        """,
        parameters={"sd": start_date, "ed": end_date},
    )
    return [row[0] for row in r.result_rows]


def get_adj_daily(
    ch, symbols: list[str], start_date: date, end_date: date,
) -> pl.DataFrame:
    if not symbols:
        return pl.DataFrame()
    daily = get_stock_daily_agg(ch, symbols, start_date, end_date)
    if daily.height == 0:
        return pl.DataFrame()
    ts_syms = [sym_local_to_tushare(s) for s in symbols]
    adj = get_adj_factors(ch, ts_syms, start_date, end_date)
    if adj.height == 0:
        daily = daily.with_columns(pl.col("daily_close").alias("adj_close"))
        return daily
    adj = adj.with_columns(
        pl.col("symbol").map_elements(
            lambda s: sym_tushare_to_local(s), return_dtype=pl.Utf8,
        ).alias("symbol"),
    )
    daily = daily.join(
        adj.select("symbol", "trade_date", pl.col("factor").alias("adj_factor")),
        on=["symbol", "trade_date"],
        how="left",
    )
    daily = daily.with_columns(
        (pl.col("daily_close") * pl.col("adj_factor").fill_null(1.0)).alias("adj_close"),
    )
    return daily


def get_stock_open_prices(
    ch, symbols: list[str], trade_date: date,
) -> pl.DataFrame:
    if not symbols:
        return pl.DataFrame()
    r = ch.query_arrow(
        """
        SELECT symbol, argMin(open, datetime) AS day_open
        FROM klines_1m_stock
        WHERE trade_date = %(d)s AND symbol IN %(syms)s
        GROUP BY symbol
        """,
        parameters={"d": trade_date, "syms": symbols},
    )
    if r.num_rows == 0:
        return pl.DataFrame(schema={"symbol": pl.Utf8, "day_open": pl.Float64})
    return pl.from_arrow(r).with_columns(pl.col("day_open").cast(pl.Float64))


# ── 单事件回测 ────────────────────────────────────────────────────────────


def run_basket_event(
    ch,
    signal_date: date,
    signal_prob: float,
    hold_days: int = HOLD_DAYS,
    basket_mode: str = "full",  # "full" | "top30" | "top50"
) -> dict | None:
    """
    单事件: T1 open 买入 HS300 篮子, 持有 hold_days 天, 按 T_exit close 卖出.
    Returns event-level stats dict, or None if skipped.
    """
    members = get_hs300_members(ch, signal_date)
    if not members:
        return None

    # 交易日历
    cal = get_trading_dates(
        ch,
        signal_date - timedelta(days=5),
        signal_date + timedelta(days=hold_days * 3 + 10),
    )
    if signal_date not in cal:
        return None
    t0_idx = cal.index(signal_date)
    if t0_idx + 1 + hold_days >= len(cal):
        return None

    t1 = cal[t0_idx + 1]  # T+1 入场日
    t_exit = cal[t0_idx + 1 + hold_days]  # 入场后持有 hold_days 天

    # 获取日线数据 (含复权)
    daily = get_adj_daily(ch, members, t1 - timedelta(days=5), t_exit + timedelta(days=5))
    if daily.height == 0:
        return None

    # T1 开盘价 (复权)
    t1_open_raw = get_stock_open_prices(ch, members, t1)
    t1_adj = daily.filter(pl.col("trade_date") == t1).select(
        "symbol", pl.col("adj_factor").fill_null(1.0).alias("t1_adj"),
    )
    t1_open = (
        t1_open_raw
        .join(t1_adj, on="symbol", how="left")
        .with_columns(
            (pl.col("day_open") * pl.col("t1_adj").fill_null(1.0)).alias("adj_open"),
        )
        .filter(pl.col("adj_open") > 0)
        .select("symbol", "adj_open")
    )

    # T_exit 收盘价 (复权)
    t_exit_close = (
        daily.filter(pl.col("trade_date") == t_exit)
        .filter(pl.col("adj_close") > 0)
        .select("symbol", pl.col("adj_close").alias("exit_close"))
    )

    # 合并 & 算收益
    merged = t1_open.join(t_exit_close, on="symbol", how="inner").with_columns(
        (pl.col("exit_close") / pl.col("adj_open") - 1.0).alias("stock_ret"),
    )

    if merged.height == 0:
        return None

    # 篮子模式
    if basket_mode == "top30":
        merged = merged.head(30)
    elif basket_mode == "top50":
        merged = merged.head(50)

    # 等权篮子收益
    basket_ret = float(merged["stock_ret"].mean())
    pos = position_size(signal_prob)
    portfolio_ret = basket_ret * pos

    return {
        "signal_date": signal_date,
        "signal_prob": signal_prob,
        "position_size": pos,
        "n_stocks": merged.height,
        "basket_ret": basket_ret,
        "portfolio_ret": portfolio_ret,
        "entry_date": t1,
        "exit_date": t_exit,
        "year": signal_date.year,
    }


# ── 统计函数 ──────────────────────────────────────────────────────────────


def compute_stats(df: pl.DataFrame, label: str) -> dict:
    """计算一组事件的统计指标。"""
    rets = df["portfolio_ret"]
    n = rets.len()
    if n == 0:
        return {"label": label, "n": 0}

    mean_ret = float(rets.mean())
    std_ret = float(rets.std()) if n > 1 else 0.0
    win_rate = df.filter(pl.col("portfolio_ret") > 0).height / n

    # 复合累计
    compound_ret = float((1 + rets).product() - 1)

    # 净值曲线 & 最大回撤
    equity = (1 + rets).cum_prod()
    peak = equity.cum_max()
    dd = (equity - peak) / peak
    max_dd = float(dd.min())

    # t-stat
    t_stat = mean_ret / (std_ret / math.sqrt(n)) if std_ret > 0 else 0.0

    # IR (annualized, 假设每事件独立)
    events_per_year = n / max(1, df["year"].n_unique())
    ir = (mean_ret / std_ret * math.sqrt(events_per_year)) if std_ret > 0 else 0.0

    # estimate total calendar time
    basket_rets = df["basket_ret"]
    avg_hold = 3.0  # fixed hold days
    total_days = avg_hold * n
    total_years = total_days / 252.0
    ann_ret = ((1 + compound_ret) ** (1.0 / total_years) - 1) if total_years > 0 and compound_ret > -1 else 0.0

    # bootstrap 95% CI for mean (1000 resamples)
    rng = np.random.default_rng(42)
    arr = rets.to_numpy()
    boot_means = []
    for _ in range(1000):
        sample = rng.choice(arr, size=n, replace=True)
        boot_means.append(np.mean(sample))
    boot_means.sort()
    ci_lo = boot_means[25]
    ci_hi = boot_means[975]

    return {
        "label": label,
        "n": n,
        "mean_ret": mean_ret,
        "median_ret": float(rets.median()),
        "std_ret": std_ret,
        "win_rate": win_rate,
        "sum_ret": float(rets.sum()),
        "compound_ret": compound_ret,
        "ann_ret": ann_ret,
        "max_dd": max_dd,
        "t_stat": t_stat,
        "ir": ir,
        "ci_95_lo": float(ci_lo),
        "ci_95_hi": float(ci_hi),
    }


def format_stats_table(stats_list: list[dict]) -> list[str]:
    """生成 Markdown 统计表。"""
    lines = [
        "| 样本 | N | 平均 | 中位数 | 胜率 | sum | compound | ann% | maxDD | t-stat | IR | 95%CI |",
        "|------|---|------|--------|------|-----|----------|------|-------|--------|----|-------|",
    ]
    for s in stats_list:
        if s["n"] == 0:
            lines.append(f"| {s['label']} | 0 | - | - | - | - | - | - | - | - | - | - |")
            continue
        lines.append(
            f"| {s['label']} | {s['n']} "
            f"| {s['mean_ret']:+.3%} | {s['median_ret']:+.3%} "
            f"| {s['win_rate']:.0%} "
            f"| {s['sum_ret']:+.2%} | {s['compound_ret']:+.2%} "
            f"| {s['ann_ret']:+.1%} | {s['max_dd']:+.1%} "
            f"| {s['t_stat']:+.2f} | {s['ir']:.2f} "
            f"| [{s['ci_95_lo']:+.3%}, {s['ci_95_hi']:+.3%}] |"
        )
    return lines


# ── 生成报告 ──────────────────────────────────────────────────────────────


def generate_report(df: pl.DataFrame, output_path: Path) -> None:
    lines = [
        "# 国家队事件篮子策略基线报告",
        "",
        "策略: 信号触发 → T1 开盘买入 HS300 等权篮子 → 持有 3 天 → prob 分层仓位",
        "",
        "## 仓位分层规则",
        "",
        "| prob 区间 | 仓位比例 |",
        "|-----------|---------|",
    ]
    prev = 0.0
    for threshold, size in POSITION_TIERS:
        lines.append(f"| {prev:.0%}–{threshold:.0%} | {size:.0%} |")
        prev = threshold
    lines.append("")

    # ── 全量 & 分期统计 ──────────────────────────────────────────────────
    lines.append("## 全量 & 分期统计")
    lines.append("")

    periods = [
        ("全量", df),
        ("2016+", df.filter(pl.col("year") >= 2016)),
        ("2018+", df.filter(pl.col("year") >= 2018)),
        ("2020+", df.filter(pl.col("year") >= 2020)),
        ("2023+", df.filter(pl.col("year") >= 2023)),
        ("去2015", df.filter(pl.col("year") != 2015)),
    ]
    stats_list = [compute_stats(sub, label) for label, sub in periods]
    lines.extend(format_stats_table(stats_list))
    lines.append("")

    # ── 不分层 (100% 仓位) 对照 ─────────────────────────────────────────
    lines.append("## 对照: 不分层 (固定 100% 仓位)")
    lines.append("")
    df_full_pos = df.with_columns(pl.col("basket_ret").alias("portfolio_ret"))
    periods_fp = [
        ("全量_100%", df_full_pos),
        ("2016+_100%", df_full_pos.filter(pl.col("year") >= 2016)),
        ("2020+_100%", df_full_pos.filter(pl.col("year") >= 2020)),
        ("2023+_100%", df_full_pos.filter(pl.col("year") >= 2023)),
        ("去2015_100%", df_full_pos.filter(pl.col("year") != 2015)),
    ]
    stats_fp = [compute_stats(sub, label) for label, sub in periods_fp]
    lines.extend(format_stats_table(stats_fp))
    lines.append("")

    # ── 分年统计 ─────────────────────────────────────────────────────────
    lines.append("## 分年统计")
    lines.append("")
    years = sorted(df["year"].unique().to_list())
    year_stats = [compute_stats(df.filter(pl.col("year") == y), str(y)) for y in years]
    lines.extend(format_stats_table(year_stats))
    lines.append("")

    # ── 仓位分层效果 ─────────────────────────────────────────────────────
    lines.append("## 仓位分层效果 (prob 分桶)")
    lines.append("")
    lines.append("| prob 桶 | N | 平均basket | 平均portfolio | 胜率 |")
    lines.append("|---------|---|-----------|--------------|------|")
    prob_bins = [
        ("≤10%", (0.0, 0.10)),
        ("10-20%", (0.10, 0.20)),
        ("20-35%", (0.20, 0.35)),
        (">35%", (0.35, 1.01)),
    ]
    for bin_label, (lo, hi) in prob_bins:
        sub = df.filter((pl.col("signal_prob") > lo) & (pl.col("signal_prob") <= hi))
        n = sub.height
        if n > 0:
            bm = float(sub["basket_ret"].mean())
            pm = float(sub["portfolio_ret"].mean())
            wr = sub.filter(pl.col("basket_ret") > 0).height / n
            lines.append(f"| {bin_label} | {n} | {bm:+.3%} | {pm:+.3%} | {wr:.0%} |")
        else:
            lines.append(f"| {bin_label} | 0 | - | - | - |")
    lines.append("")

    # ── 事件明细 (最近 20 个) ─────────────────────────────────────────────
    lines.append("## 事件明细 (最近 20 个)")
    lines.append("")
    lines.append("| 信号日 | prob | 仓位 | 股数 | basket_ret | portfolio_ret |")
    lines.append("|--------|------|------|------|-----------|--------------|")
    recent = df.sort("signal_date", descending=True).head(20)
    for row in recent.sort("signal_date").iter_rows(named=True):
        lines.append(
            f"| {row['signal_date']} | {row['signal_prob']:.0%} "
            f"| {row['position_size']:.0%} | {row['n_stocks']} "
            f"| {row['basket_ret']:+.3%} | {row['portfolio_ret']:+.3%} |"
        )
    lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n报告已写入: {output_path}")


# ── 主流程 ────────────────────────────────────────────────────────────────


def main() -> None:
    print("=" * 70)
    print("  国家队事件篮子策略基线")
    print("  T1 开盘 → HS300 等权 → 固定 3 天 → prob 分层仓位")
    print("=" * 70)

    # Step 1: 信号日
    print(f"\n[Step 1] 检测信号日 (prob > {SIGNAL_THRESHOLD:.0%}) ...")
    signals = detect_signal_dates()
    print(f"  → 共 {len(signals)} 个信号日")
    if not signals:
        print("No signals!")
        return

    ch = get_ch()

    # Step 2: 逐事件回测
    print(f"\n[Step 2] 逐事件回测 (hold={HOLD_DAYS}d) ...")
    rows = []
    for i, (sig_date, prob) in enumerate(signals):
        pos = position_size(prob)
        result = run_basket_event(ch, sig_date, prob, hold_days=HOLD_DAYS)
        if result is not None:
            rows.append(result)
            print(f"  [{i+1}/{len(signals)}] {sig_date} prob={prob:.1%} pos={pos:.0%} "
                  f"n={result['n_stocks']} basket={result['basket_ret']:+.3%} "
                  f"port={result['portfolio_ret']:+.3%}")
        else:
            print(f"  [{i+1}/{len(signals)}] {sig_date} prob={prob:.1%} → skip")

    if not rows:
        print("No valid events!")
        return

    df = pl.DataFrame(rows)
    print(f"\n  → 有效事件: {df.height}/{len(signals)}")

    # Step 3: 保存
    out_dir = ROOT / "data" / "basket_baseline"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_dir / "events.parquet")
    print(f"  事件数据已保存: {out_dir / 'events.parquet'}")

    # Step 4: 统计 & 报告
    print("\n[Step 3] 生成报告 ...")

    # 总览
    stats = compute_stats(df, "全量")
    print(f"\n{'='*60}")
    print(f"  全量: N={stats['n']}, mean={stats['mean_ret']:+.3%}, "
          f"compound={stats['compound_ret']:+.2%}, "
          f"maxDD={stats['max_dd']:+.1%}, t={stats['t_stat']:.2f}")

    for period_label, start_year in [("2016+", 2016), ("2020+", 2020), ("2023+", 2023)]:
        sub = df.filter(pl.col("year") >= start_year)
        s = compute_stats(sub, period_label)
        print(f"  {period_label}: N={s['n']}, mean={s['mean_ret']:+.3%}, "
              f"compound={s['compound_ret']:+.2%}, "
              f"maxDD={s['max_dd']:+.1%}, t={s['t_stat']:.2f}")

    report_path = ROOT / "doc" / "basket_baseline_report.md"
    generate_report(df, report_path)

    print(f"\n{'='*70}")
    print("  完成!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
