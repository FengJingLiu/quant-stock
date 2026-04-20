"""
分析事件组合买入后能吃到多少收益、用时多久。

不看任何退出信号，纯粹跟踪 T+1 买入后，每只个股在未来 N 个交易日内：
  - 最大涨幅 (前复权)
  - 是否触达 10% / 20% / 30% 收益门槛
  - 触达各门槛所需交易日数

输出:
  - 按 top_n 分组的触达率统计
  - 按信号日的触达分布
  - 控制台表格 + doc/gain_threshold_report.md
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import clickhouse_connect

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data_clients import (
    create_clickhouse_http_client,
    query_clickhouse_arrow_df,
)
from src.data_queries import (
    stock_daily_close_sql,
    stock_qfq_adj_factor_sql,
    trading_dates_sql,
)
from src.national_team.elastic_scorer import ElasticScorer
from scripts.backtest_buy_elasticity import (
    detect_signal_dates,
    get_hs300_members,
    SIGNAL_THRESHOLD,
)

# ── 配置 ──────────────────────────────────────────────────────────────────
MAX_HOLD_DAYS = 40        # 最大跟踪交易日
TOP_N_LIST = [3, 5, 8]    # 分别看 top3/5/8 的结果
THRESHOLDS = [0.10, 0.20, 0.30]  # 收益门槛


def get_ch() -> clickhouse_connect.driver.Client:
    return create_clickhouse_http_client()


def get_trading_dates(ch, start: date, end: date) -> list[date]:
    df = query_clickhouse_arrow_df(
        trading_dates_sql("klines_1m_index"),
        parameters={"sym": "000300", "sd": start, "ed": end},
        client=ch,
    )
    if df.height == 0:
        return []
    return df.with_columns(
        pl.col("trade_date").cast(pl.Date),
    )["trade_date"].to_list()


def get_stock_adj_daily(
    ch, symbols: list[str], start: date, end: date,
) -> pl.DataFrame:
    """获取个股日线 + 前复权收盘价。"""
    daily = query_clickhouse_arrow_df(
        stock_daily_close_sql(),
        parameters={"sd": start, "ed": end, "syms": symbols},
        client=ch,
    )
    if daily.height == 0:
        return pl.DataFrame()
    daily = daily.with_columns(
        pl.col("trade_date").cast(pl.Date),
        pl.col("daily_close").cast(pl.Float64),
    )

    # 前复权因子
    ts_symbols = [f"{s[2:]}.{s[:2].upper()}" for s in symbols]
    adj_df = query_clickhouse_arrow_df(
        stock_qfq_adj_factor_sql("ts_sym"),
        parameters={"sd": start, "ed": end, "syms": ts_symbols},
        client=ch,
    )

    if adj_df.height > 0:
        adj_df = adj_df.with_columns(
            pl.col("trade_date").cast(pl.Date),
            pl.col("factor").cast(pl.Float64),
            pl.col("ts_sym").map_elements(
                lambda s: f"{s.split('.')[1].lower()}{s.split('.')[0]}",
                return_dtype=pl.Utf8,
            ).alias("symbol"),
        ).select("symbol", "trade_date", "factor")

        daily = daily.join(adj_df, on=["symbol", "trade_date"], how="left")
        daily = daily.with_columns(
            (pl.col("daily_close") * pl.col("factor").fill_null(1.0)).alias("adj_close"),
        )
    else:
        daily = daily.with_columns(pl.col("daily_close").alias("adj_close"))

    return daily


def track_single_event(
    ch: clickhouse_connect.driver.Client,
    scorer: ElasticScorer,
    signal_date: date,
    signal_prob: float,
) -> list[dict]:
    """对单个事件日，跟踪所有候选股的前向收益路径。"""

    members = get_hs300_members(ch, signal_date)
    if not members:
        return []

    scored = scorer.score(ch, signal_date, members)
    if scored.height == 0:
        return []

    # 给每只股票打上排名
    scored = scored.with_row_index("rank", offset=1)

    symbols = scored["symbol"].to_list()

    # 获取交易日历
    data_end = signal_date + timedelta(days=MAX_HOLD_DAYS * 2 + 10)
    trading_dates = get_trading_dates(ch, signal_date, data_end)

    if len(trading_dates) < 2:
        return []

    # T+1 买入日
    t0_idx = None
    for i, d in enumerate(trading_dates):
        if d >= signal_date:
            t0_idx = i
            break
    if t0_idx is None or t0_idx + 1 >= len(trading_dates):
        return []
    entry_date = trading_dates[t0_idx + 1]

    # 获取日线数据
    daily = get_stock_adj_daily(
        ch, symbols,
        signal_date - timedelta(days=5),
        data_end,
    )
    if daily.height == 0:
        return []

    results = []

    for row in scored.iter_rows(named=True):
        sym = row["symbol"]
        rank = row["rank"]
        score = row["elastic_score"]

        sym_daily = daily.filter(pl.col("symbol") == sym).sort("trade_date")
        sym_dates = sym_daily["trade_date"].to_list()
        sym_adj = sym_daily["adj_close"].to_list()

        if entry_date not in sym_dates:
            continue

        entry_idx = sym_dates.index(entry_date)
        entry_price = sym_adj[entry_idx]
        if entry_price is None or entry_price <= 0:
            continue

        # 逐日跟踪
        max_gain = 0.0
        max_gain_day = 0
        threshold_days = {}  # threshold -> first day hit

        for offset in range(1, MAX_HOLD_DAYS + 1):
            check_idx = entry_idx + offset
            if check_idx >= len(sym_dates):
                break

            cur_price = sym_adj[check_idx]
            if cur_price is None:
                continue

            gain = cur_price / entry_price - 1.0

            if gain > max_gain:
                max_gain = gain
                max_gain_day = offset

            for thr in THRESHOLDS:
                if thr not in threshold_days and gain >= thr:
                    threshold_days[thr] = offset

        rec = {
            "signal_date": signal_date,
            "signal_prob": signal_prob,
            "symbol": sym,
            "rank": rank,
            "elastic_score": score,
            "entry_date": entry_date,
            "entry_price": entry_price,
            "max_gain": max_gain,
            "max_gain_day": max_gain_day,
        }
        for thr in THRESHOLDS:
            pct = int(thr * 100)
            rec[f"hit_{pct}"] = thr in threshold_days
            rec[f"days_to_{pct}"] = threshold_days.get(thr)

        results.append(rec)

    return results


def main() -> None:
    print("=" * 60)
    print("收益门槛触达分析 (不设退出信号)")
    print(f"跟踪窗口: {MAX_HOLD_DAYS} 交易日")
    print(f"门槛: {[f'{t:.0%}' for t in THRESHOLDS]}")
    print("=" * 60)

    # Step 1: 信号日
    print(f"\n[Step 1] 检测信号日 ...")
    signals = detect_signal_dates()
    print(f"  → {len(signals)} 个信号日")

    if not signals:
        print("No signals!")
        return

    ch = get_ch()
    scorer = ElasticScorer()

    # Step 2: 逐事件跟踪
    print(f"\n[Step 2] 逐事件跟踪前向收益 ...")
    all_records: list[dict] = []

    for i, (sig_date, prob) in enumerate(signals, 1):
        print(f"  [{i}/{len(signals)}] {sig_date} (prob={prob:.2%}) ... ", end="", flush=True)
        recs = track_single_event(ch, scorer, sig_date, prob)
        all_records.extend(recs)
        if recs:
            gains = [r["max_gain"] for r in recs]
            print(f"{len(recs)} stocks, max_gain median={sorted(gains)[len(gains)//2]:+.1%}")
        else:
            print("skip")

    if not all_records:
        print("No data!")
        return

    df = pl.DataFrame(all_records)
    print(f"\n  总计 {df.height} 条记录")

    # Step 3: 按 top_n 分组统计
    print(f"\n{'='*60}")
    print("按 Top N 分组统计")
    print(f"{'='*60}")

    report_lines = [
        "# 收益门槛触达分析",
        "",
        f"- **跟踪窗口**: {MAX_HOLD_DAYS} 交易日 (不设退出信号)",
        f"- **信号日数**: {len(signals)}",
        f"- **门槛**: {', '.join(f'{t:.0%}' for t in THRESHOLDS)}",
        "",
    ]

    for top_n in TOP_N_LIST:
        subset = df.filter(pl.col("rank") <= top_n)
        total = subset.height

        print(f"\n--- Top {top_n} ({total} 条记录) ---")
        report_lines.append(f"## Top {top_n} (共 {total} 条)")
        report_lines.append("")

        # 最大涨幅分布
        max_gains = subset["max_gain"]
        print(f"  最大涨幅: mean={max_gains.mean():+.1%}  median={max_gains.median():+.1%}  "
              f"p75={max_gains.quantile(0.75):+.1%}  p90={max_gains.quantile(0.9):+.1%}  "
              f"max={max_gains.max():+.1%}")

        report_lines.append(f"**最大涨幅分布**: mean={max_gains.mean():+.1%}, "
                           f"median={max_gains.median():+.1%}, "
                           f"p75={max_gains.quantile(0.75):+.1%}, "
                           f"p90={max_gains.quantile(0.9):+.1%}, "
                           f"max={max_gains.max():+.1%}")
        report_lines.append("")

        # 门槛触达率 & 用时
        report_lines.append("| 门槛 | 触达率 | 触达数 | 平均用时 | 中位用时 | 最快 | 最慢(p90) |")
        report_lines.append("|------|--------|--------|---------|---------|------|----------|")

        for thr in THRESHOLDS:
            pct = int(thr * 100)
            hit_col = f"hit_{pct}"
            days_col = f"days_to_{pct}"

            n_hit = subset.filter(pl.col(hit_col) == True).height
            hit_rate = n_hit / total if total > 0 else 0

            hit_subset = subset.filter(pl.col(days_col).is_not_null())
            if hit_subset.height > 0:
                days_vals = hit_subset[days_col].cast(pl.Float64)
                avg_days = days_vals.mean()
                med_days = days_vals.median()
                min_days = days_vals.min()
                p90_days = days_vals.quantile(0.9)
            else:
                avg_days = med_days = min_days = p90_days = None

            days_str = (f"{avg_days:.1f}" if avg_days else "-")
            med_str = (f"{med_days:.0f}" if med_days else "-")
            min_str = (f"{min_days:.0f}" if min_days else "-")
            p90_str = (f"{p90_days:.0f}" if p90_days else "-")

            print(f"  {thr:>5.0%} 触达: {hit_rate:.1%} ({n_hit}/{total})  "
                  f"avg={days_str}d  median={med_str}d  fastest={min_str}d  p90={p90_str}d")

            report_lines.append(
                f"| {thr:.0%} | {hit_rate:.1%} | {n_hit}/{total} "
                f"| {days_str}d | {med_str}d | {min_str}d | {p90_str}d |"
            )

        report_lines.append("")

        # 按排名看: rank 1 vs 2 vs 3 ... 的触达率差异
        if top_n >= 5:
            report_lines.append(f"### 排名 vs 10% 触达率")
            report_lines.append("")
            report_lines.append("| 排名 | 股票数 | 10%触达率 | 20%触达率 | avg最大涨幅 |")
            report_lines.append("|------|--------|----------|----------|-----------|")
            for r in range(1, top_n + 1):
                rank_sub = subset.filter(pl.col("rank") == r)
                if rank_sub.height == 0:
                    continue
                h10 = rank_sub.filter(pl.col("hit_10") == True).height / rank_sub.height
                h20 = rank_sub.filter(pl.col("hit_20") == True).height / rank_sub.height
                avg_mg = rank_sub["max_gain"].mean()
                print(f"    rank {r}: 10%={h10:.0%}  20%={h20:.0%}  avg_max_gain={avg_mg:+.1%}")
                report_lines.append(f"| {r} | {rank_sub.height} | {h10:.0%} | {h20:.0%} | {avg_mg:+.1%} |")
            report_lines.append("")

    # Step 4: 分年份看触达率变化 (top5)
    print(f"\n{'='*60}")
    print("Top 5 分年份触达率")
    print(f"{'='*60}")

    top5 = df.filter(pl.col("rank") <= 5).with_columns(
        pl.col("signal_date").dt.year().alias("year"),
    )

    report_lines.append("## Top 5 分年份触达率")
    report_lines.append("")
    report_lines.append("| 年份 | 事件数 | 股票数 | 10%触达 | 20%触达 | 30%触达 | avg最大涨幅 |")
    report_lines.append("|------|--------|--------|---------|---------|---------|-----------|")

    for year in sorted(top5["year"].unique().to_list()):
        yr_sub = top5.filter(pl.col("year") == year)
        n_events = yr_sub["signal_date"].n_unique()
        n_stocks = yr_sub.height
        h10 = yr_sub.filter(pl.col("hit_10") == True).height / n_stocks if n_stocks > 0 else 0
        h20 = yr_sub.filter(pl.col("hit_20") == True).height / n_stocks if n_stocks > 0 else 0
        h30 = yr_sub.filter(pl.col("hit_30") == True).height / n_stocks if n_stocks > 0 else 0
        avg_mg = yr_sub["max_gain"].mean()

        print(f"  {year}: {n_events}事件 {n_stocks}只  "
              f"10%={h10:.0%}  20%={h20:.0%}  30%={h30:.0%}  avg_max={avg_mg:+.1%}")

        report_lines.append(
            f"| {year} | {n_events} | {n_stocks} "
            f"| {h10:.0%} | {h20:.0%} | {h30:.0%} | {avg_mg:+.1%} |"
        )

    report_lines.append("")

    # Step 5: 触达时间分布直方图 (文本)
    print(f"\n{'='*60}")
    print("10% 触达时间分布 (Top 5)")
    print(f"{'='*60}")

    top5_hit10 = top5.filter(pl.col("hit_10") == True)
    if top5_hit10.height > 0:
        days = top5_hit10["days_to_10"].cast(pl.Int32)
        report_lines.append("## 10% 触达时间分布 (Top 5)")
        report_lines.append("")
        report_lines.append("| 交易日 | 触达数 | 占比 | 累计 |")
        report_lines.append("|--------|--------|------|------|")
        cum = 0
        for d in range(1, MAX_HOLD_DAYS + 1):
            n = days.filter(days == d).len()
            if n == 0 and d > 20:
                continue
            cum += n
            pct = n / top5_hit10.height
            cum_pct = cum / top5_hit10.height
            bar = "█" * int(pct * 50)
            print(f"  T+{d:2d}: {n:3d} ({pct:5.1%}) cum={cum_pct:5.1%}  {bar}")
            if n > 0:
                report_lines.append(f"| T+{d} | {n} | {pct:.1%} | {cum_pct:.1%} |")

    # 保存报告
    root = Path(__file__).resolve().parents[2]
    report_path = root / "doc" / "gain_threshold_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\n报告: {report_path}")

    # 保存明细
    parquet_path = root / "data" / "gain_threshold_detail.parquet"
    df.write_parquet(parquet_path)
    print(f"明细: {parquet_path} ({df.height} 条)")


if __name__ == "__main__":
    main()
