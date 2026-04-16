"""
回测国家队买入信号 v2 (2010-01-01 ~ 2026-04-16)

v2 变更:
  - 使用 NTBuyProb v2: stress_context / vol_shock / absorption / resonance / lead_gap
  - 以 510300.SH ETF 为主目标 (非个股)
  - 自行加载舰队 ETF 数据, 传入 fleet_etf_data

按半年分片处理（数据量大）。

输出：
  - 控制台汇总
  - doc/nt_buy_signals_backtest_v2.md 完整报告
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.national_team.ch_client import get_etf_1m, get_index_1m
from src.national_team.nt_buy_prob import NTBuyProb

# ── 配置 ─────────────────────────────────────────────────────────────────
TARGET_ETF = "510300.SH"
INDEX_SYM = "000300"
START_YEAR = 2010
END_DATE = "2026-04-16"

# 舰队 ETF (用于共振计算)
FLEET_ETFS = ["510300.SH", "510500.SH", "512100.SH", "510050.SH"]

# 信号阈值
PROB_THRESHOLDS = [0.10, 0.20, 0.50]

# ETF 数据可用时间
ETF_RESONANCE_START = date(2013, 4, 1)

# 每次查询跨度：半年
CHUNK_MONTHS = 6


def date_chunks(start: str, end: str, months: int = CHUNK_MONTHS):
    """生成 (start_date, end_date) 的半年切片。"""
    from dateutil.relativedelta import relativedelta
    from datetime import datetime

    sd = datetime.strptime(start, "%Y-%m-%d").date() if isinstance(start, str) else start
    ed = datetime.strptime(end, "%Y-%m-%d").date() if isinstance(end, str) else end

    cur = sd
    while cur < ed:
        chunk_end = min(cur + relativedelta(months=months) - relativedelta(days=1), ed)
        yield cur, chunk_end
        cur = chunk_end + relativedelta(days=1)


def run_chunk(
    sd: date,
    ed: date,
    buy_model: NTBuyProb,
) -> pl.DataFrame | None:
    """处理一个时间片段，返回信号 DataFrame 或 None。"""
    warmup_days = 30
    from dateutil.relativedelta import relativedelta
    warmup_sd = sd - relativedelta(days=warmup_days + 10)

    # 目标 ETF + 指数
    etf_1m = get_etf_1m(TARGET_ETF, warmup_sd, ed)
    index_1m = get_index_1m(INDEX_SYM, warmup_sd, ed)

    if etf_1m.height == 0 or index_1m.height == 0:
        return None

    # 舰队 ETF (用于共振)
    fleet_data: dict[str, pl.DataFrame] = {}
    if ed >= ETF_RESONANCE_START:
        eff_sd = max(warmup_sd, date(2012, 6, 1))
        for sym in FLEET_ETFS:
            try:
                fleet_df = get_etf_1m(sym, eff_sd, ed)
                if fleet_df.height > 0:
                    fleet_data[sym] = fleet_df
            except Exception as e:
                print(f"  [WARN] fleet ETF {sym} failed for {sd}~{ed}: {e}")

    result = buy_model.compute(etf_1m, index_1m, fleet_data or None)

    # 只保留正式区间（去掉 warmup）
    result = result.filter(pl.col("trade_date") >= sd)
    return result


def main() -> None:
    buy_model = NTBuyProb()

    all_signals: list[pl.DataFrame] = []
    start_str = f"{START_YEAR}-01-01"

    chunks = list(date_chunks(start_str, END_DATE))
    total = len(chunks)

    for i, (sd, ed) in enumerate(chunks, 1):
        print(f"[{i}/{total}] Processing {sd} ~ {ed} ...", flush=True)
        result = run_chunk(sd, ed, buy_model)
        if result is not None and result.height > 0:
            all_signals.append(result)
            n_bars = result.height
            for th in PROB_THRESHOLDS:
                n_sig = result.filter(pl.col("nt_buy_prob") > th).height
                if n_sig > 0:
                    print(f"  prob>{th}: {n_sig} signals in {n_bars} bars")

    if not all_signals:
        print("No data found!")
        return

    full = pl.concat(all_signals)
    print(f"\nTotal bars: {full.height}")

    # ── 汇总 ─────────────────────────────────────────────────────────────
    report_lines = []
    report_lines.append("# 国家队买入信号回测报告 v2")
    report_lines.append("")
    report_lines.append(f"- **标的**: {TARGET_ETF} (参考指数: {INDEX_SYM})")
    report_lines.append(f"- **区间**: {START_YEAR}-01-01 ~ {END_DATE}")
    report_lines.append(f"- **总 1 分钟 bar 数**: {full.height:,}")
    report_lines.append(f"- **交易日数**: {full['trade_date'].n_unique()}")
    report_lines.append("- **v2 特征**: stress_context / vol_shock / absorption / resonance / lead_gap")
    report_lines.append("- **组合方法**: 线性 + sigmoid (替代贝叶斯似然比)")
    report_lines.append("")

    # 各阈值统计
    report_lines.append("## 信号统计")
    report_lines.append("")
    report_lines.append("| 阈值 | 触发 bar 数 | 触发交易日数 | 首次触发 | 末次触发 |")
    report_lines.append("|------|------------|------------|---------|---------|")

    for th in PROB_THRESHOLDS:
        sig = full.filter(pl.col("nt_buy_prob") > th)
        n_bars = sig.height
        if n_bars > 0:
            n_days = sig["trade_date"].n_unique()
            first = sig["datetime"].min()
            last = sig["datetime"].max()
            report_lines.append(f"| >{th:.0%} | {n_bars:,} | {n_days} | {first} | {last} |")
        else:
            report_lines.append(f"| >{th:.0%} | 0 | 0 | - | - |")

    # 年度分布 (prob > 10%)
    report_lines.append("")
    report_lines.append("## 年度信号分布 (prob > 10%)")
    report_lines.append("")

    sig10 = full.filter(pl.col("nt_buy_prob") > 0.10)
    if sig10.height > 0:
        yearly = (
            sig10
            .with_columns(pl.col("trade_date").dt.year().alias("year"))
            .group_by("year")
            .agg(
                pl.col("datetime").count().alias("signal_bars"),
                pl.col("trade_date").n_unique().alias("signal_days"),
                pl.col("nt_buy_prob").max().alias("max_prob"),
            )
            .sort("year")
        )
        report_lines.append("| 年份 | 信号 bar 数 | 信号交易日 | 最大概率 |")
        report_lines.append("|------|-----------|----------|---------|")
        for row in yearly.iter_rows(named=True):
            report_lines.append(
                f"| {row['year']} | {row['signal_bars']:,} | {row['signal_days']} | {row['max_prob']:.2%} |"
            )
    else:
        report_lines.append("*无信号*")

    # 高概率信号明细 (prob > 20%)
    report_lines.append("")
    report_lines.append("## 高概率信号明细 (prob > 20%)")
    report_lines.append("")

    sig20 = full.filter(pl.col("nt_buy_prob") > 0.20)
    if sig20.height > 0:
        # 按日聚合：每日取最大概率及其时刻
        daily_top = (
            sig20
            .sort("nt_buy_prob", descending=True)
            .unique(subset=["trade_date"], maintain_order=True)
            .sort("trade_date")
        )

        report_lines.append("| 日期 | 时刻 | 概率 | stress | vol_shock | absorption | resonance | lead_gap |")
        report_lines.append("|------|------|------|--------|-----------|-----------|-----------|----------|")
        for row in daily_top.iter_rows(named=True):
            dt = row["datetime"]
            time_str = dt.strftime("%H:%M") if dt else "-"
            report_lines.append(
                f"| {row['trade_date']} | {time_str} | {row['nt_buy_prob']:.2%} "
                f"| {row['stress_context']:.2f} "
                f"| {row['vol_shock']:.1f} "
                f"| {row['absorption']:.3f} "
                f"| {row['resonance_count']} "
                f"| {row['lead_gap']:.5f} |"
            )
    else:
        report_lines.append("*无高概率信号*")

    # 强信号日 (prob > 50%)
    report_lines.append("")
    report_lines.append("## 强信号日 (prob > 50%)")
    report_lines.append("")

    sig50 = full.filter(pl.col("nt_buy_prob") > 0.50)
    if sig50.height > 0:
        daily50 = (
            sig50
            .group_by("trade_date")
            .agg(
                pl.col("nt_buy_prob").max().alias("max_prob"),
                pl.col("nt_buy_prob").count().alias("signal_bars"),
                pl.col("resonance_count").max().alias("max_resonance"),
            )
            .sort("trade_date")
        )
        report_lines.append("| 日期 | 最大概率 | 信号 bar 数 | 最大共振数 |")
        report_lines.append("|------|---------|-----------|----------|")
        for row in daily50.iter_rows(named=True):
            report_lines.append(
                f"| {row['trade_date']} | {row['max_prob']:.2%} "
                f"| {row['signal_bars']} "
                f"| {row['max_resonance']} |"
            )
    else:
        report_lines.append("*无强信号*")

    # 写入文件
    report_text = "\n".join(report_lines) + "\n"
    out_path = Path(__file__).resolve().parent.parent / "doc" / "nt_buy_signals_backtest_v2.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report_text, encoding="utf-8")
    print(f"\nReport written to {out_path}")

    # 控制台也打印
    print("\n" + report_text)


if __name__ == "__main__":
    main()
