"""
弹性回测：国家队买入信号触发后，哪些个股反弹最强？

逻辑:
  1. 重跑买入因子，提取 prob > threshold 的信号日
  2. 对每个信号日，取沪深300成分股当天的日线（从 klines_1m_stock 聚合）
  3. 计算信号日后 T+1/T+3/T+5/T+10 的前复权收益率
  4. 按弹性排序，输出报告

输出:
  - doc/nt_buy_elasticity_report.md
"""

from __future__ import annotations

import os
import sys
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import clickhouse_connect

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.national_team.ch_client import get_etf_1m, get_index_1m
from src.national_team.nt_buy_prob import NTBuyProb

# ── 配置 ──────────────────────────────────────────────────────────────────
TARGET_ETF = "510300.SH"
INDEX_SYM = "000300"
INDEX_WEIGHT_CODE = "399300.SZ"  # dim_index_weights 中沪深300的代码
START_YEAR = 2013  # ETF 数据从 2012.5 开始，取完整年
END_DATE = "2026-04-16"

FLEET_ETFS = ["510300.SH", "510500.SH", "512100.SH", "510050.SH"]

# 信号阈值
SIGNAL_THRESHOLD = 0.10  # prob > 10% 算触发日

# 前向收益计算窗口（交易日）
FORWARD_WINDOWS = [1, 3, 5, 10, 20]

# 每日取最强 N 只展示
TOP_N = 20

# ClickHouse 连接
CH_KWARGS = dict(host="localhost", port=8123, username="default", password=os.environ["CH_PASSWORD"], database="astock")


def get_ch() -> clickhouse_connect.driver.Client:
    return clickhouse_connect.get_client(**CH_KWARGS)


def sym_tushare_to_local(con_code: str) -> str:
    """600519.SH -> sh600519, 000001.SZ -> sz000001"""
    code, exchange = con_code.split(".")
    return f"{exchange.lower()}{code}"


def sym_local_to_tushare(sym: str) -> str:
    """sh600519 -> 600519.SH"""
    return f"{sym[2:]}.{sym[:2].upper()}"


def get_hs300_members(ch, signal_date: date) -> list[str]:
    """获取最近一期沪深300成分股名单（返回 klines_1m_stock 格式的 symbol 列表）。"""
    r = ch.query(
        """
        SELECT con_code FROM dim_index_weights
        WHERE index_code = %(idx)s
          AND trade_date = (
              SELECT max(trade_date) FROM dim_index_weights
              WHERE index_code = %(idx)s AND trade_date <= %(d)s
          )
        """,
        parameters={"idx": INDEX_WEIGHT_CODE, "d": signal_date},
    )
    return [sym_tushare_to_local(row[0]) for row in r.result_rows]


def get_stock_daily_agg(ch, symbols: list[str], start_date: date, end_date: date) -> pl.DataFrame:
    """
    从 klines_1m_stock 聚合个股日线数据。

    Returns: DataFrame[symbol, trade_date, daily_open, daily_high, daily_low, daily_close, daily_amount]
    """
    if not symbols:
        return pl.DataFrame()

    r = ch.query_arrow(
        """
        SELECT symbol, trade_date,
               argMin(open, datetime) as daily_open,
               max(high) as daily_high,
               min(low) as daily_low,
               argMax(close, datetime) as daily_close,
               sum(amount) as daily_amount
        FROM klines_1m_stock
        WHERE trade_date BETWEEN %(sd)s AND %(ed)s
          AND symbol IN %(syms)s
        GROUP BY symbol, trade_date
        ORDER BY symbol, trade_date
        """,
        parameters={"sd": start_date, "ed": end_date, "syms": symbols},
    )
    if r.num_rows == 0:
        return pl.DataFrame()
    df = pl.from_arrow(r)
    return df.with_columns(
        pl.col("trade_date").cast(pl.Date),
        pl.col("daily_open").cast(pl.Float64),
        pl.col("daily_high").cast(pl.Float64),
        pl.col("daily_low").cast(pl.Float64),
        pl.col("daily_close").cast(pl.Float64),
        pl.col("daily_amount").cast(pl.Float64),
    )


def get_adj_factors(ch, symbols_tushare: list[str], start_date: date, end_date: date) -> pl.DataFrame:
    """获取前复权因子。"""
    if not symbols_tushare:
        return pl.DataFrame()

    r = ch.query_arrow(
        """
        SELECT symbol, trade_date, factor
        FROM adj_factor
        WHERE adj_type = 'qfq'
          AND fund_type = 'stock'
          AND trade_date BETWEEN %(sd)s AND %(ed)s
          AND symbol IN %(syms)s
        ORDER BY symbol, trade_date
        """,
        parameters={"sd": start_date, "ed": end_date, "syms": symbols_tushare},
    )
    if r.num_rows == 0:
        return pl.DataFrame()
    df = pl.from_arrow(r)
    return df.with_columns(
        pl.col("trade_date").cast(pl.Date),
        pl.col("factor").cast(pl.Float64),
    )


def compute_forward_returns(
    daily: pl.DataFrame,
    adj: pl.DataFrame,
    signal_date: date,
    windows: list[int],
) -> pl.DataFrame:
    """
    计算信号日后的前复权前向收益率。

    Returns: DataFrame[symbol, signal_date, ret_T1, ret_T3, ret_T5, ret_T10, ret_T20, ...]
    """
    # Map adj_factor symbol to local format for join
    adj = adj.with_columns(
        pl.col("symbol").map_elements(
            lambda s: sym_tushare_to_local(s), return_dtype=pl.Utf8,
        ).alias("local_sym"),
    )

    # Join adj factor to daily data
    daily_adj = daily.join(
        adj.select(
            pl.col("local_sym").alias("symbol"),
            "trade_date",
            pl.col("factor").alias("adj_factor"),
        ),
        on=["symbol", "trade_date"],
        how="left",
    )

    # Adjusted close = close * adj_factor
    daily_adj = daily_adj.with_columns(
        (pl.col("daily_close") * pl.col("adj_factor").fill_null(1.0)).alias("adj_close"),
    )

    # Get unique sorted trade dates per symbol
    # Signal date close (T0) and forward dates
    results = []
    for sym in daily_adj["symbol"].unique().to_list():
        sym_df = daily_adj.filter(pl.col("symbol") == sym).sort("trade_date")
        dates = sym_df["trade_date"].to_list()
        closes = sym_df["adj_close"].to_list()

        if signal_date not in dates:
            continue

        t0_idx = dates.index(signal_date)
        t0_close = closes[t0_idx]
        if t0_close is None or t0_close <= 0:
            continue

        row = {"symbol": sym, "signal_date": signal_date, "close_t0": t0_close}
        for w in windows:
            fw_idx = t0_idx + w
            if fw_idx < len(closes) and closes[fw_idx] is not None:
                row[f"ret_T{w}"] = closes[fw_idx] / t0_close - 1.0
                row[f"close_T{w}"] = closes[fw_idx]
            else:
                row[f"ret_T{w}"] = None
                row[f"close_T{w}"] = None

        results.append(row)

    if not results:
        return pl.DataFrame()

    return pl.DataFrame(results)


# ── 信号检测 ──────────────────────────────────────────────────────────────

def date_chunks(start: str, end: str, months: int = 6):
    from dateutil.relativedelta import relativedelta
    from datetime import datetime

    sd = datetime.strptime(start, "%Y-%m-%d").date() if isinstance(start, str) else start
    ed = datetime.strptime(end, "%Y-%m-%d").date() if isinstance(end, str) else end
    cur = sd
    while cur < ed:
        chunk_end = min(cur + relativedelta(months=months) - relativedelta(days=1), ed)
        yield cur, chunk_end
        cur = chunk_end + relativedelta(days=1)


def detect_signal_dates(threshold: float = SIGNAL_THRESHOLD) -> list[tuple[date, float]]:
    """运行买入因子，返回 [(signal_date, max_prob), ...]。"""
    from dateutil.relativedelta import relativedelta

    buy_model = NTBuyProb()
    signal_dates = {}  # date -> max_prob

    start_str = f"{START_YEAR}-01-01"
    chunks = list(date_chunks(start_str, END_DATE))

    etf_resonance_start = date(2013, 4, 1)

    for i, (sd, ed) in enumerate(chunks, 1):
        print(f"  [{i}/{len(chunks)}] Scanning {sd} ~ {ed} ...", flush=True)

        warmup_sd = sd - relativedelta(days=40)
        etf_1m = get_etf_1m(TARGET_ETF, warmup_sd, ed)
        index_1m = get_index_1m(INDEX_SYM, warmup_sd, ed)
        if etf_1m.height == 0 or index_1m.height == 0:
            continue

        fleet_data: dict[str, pl.DataFrame] = {}
        if ed >= etf_resonance_start:
            eff_sd = max(warmup_sd, date(2012, 6, 1))
            for sym in FLEET_ETFS:
                try:
                    fleet_df = get_etf_1m(sym, eff_sd, ed)
                    if fleet_df.height > 0:
                        fleet_data[sym] = fleet_df
                except Exception:
                    pass

        result = buy_model.compute(etf_1m, index_1m, fleet_data or None)
        result = result.filter(pl.col("trade_date") >= sd)

        # Extract signal dates
        sig = result.filter(pl.col("nt_buy_prob") > threshold)
        if sig.height > 0:
            daily_max = (
                sig.group_by("trade_date")
                .agg(pl.col("nt_buy_prob").max().alias("max_prob"))
            )
            for row in daily_max.iter_rows(named=True):
                d = row["trade_date"]
                p = row["max_prob"]
                if d not in signal_dates or p > signal_dates[d]:
                    signal_dates[d] = p

    return sorted(signal_dates.items())


# ── 主流程 ────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("国家队买入信号 → 个股弹性回测")
    print("=" * 60)

    # Step 1: 检测信号日
    print(f"\n[Step 1] 检测信号日 (prob > {SIGNAL_THRESHOLD:.0%}) ...")
    signals = detect_signal_dates()
    print(f"  → 共 {len(signals)} 个信号日\n")

    if not signals:
        print("No signal dates found!")
        return

    for d, p in signals:
        print(f"  {d}: max_prob = {p:.2%}")

    # Step 2: 对每个信号日计算个股弹性
    ch = get_ch()
    all_elasticity: list[pl.DataFrame] = []

    print(f"\n[Step 2] 计算个股前向收益 ...")
    for sig_date, max_prob in signals:
        print(f"\n  --- {sig_date} (prob={max_prob:.2%}) ---")

        # 获取成分股
        members = get_hs300_members(ch, sig_date)
        if not members:
            print("    无成分股数据，跳过")
            continue
        print(f"    成分股: {len(members)} 只")

        # 获取日线（信号日前 5 天 ~ 后 30 天 ← 覆盖最大窗口 T+20）
        data_sd = sig_date - timedelta(days=10)
        data_ed = sig_date + timedelta(days=45)  # 上浮以覆盖假日

        daily = get_stock_daily_agg(ch, members, data_sd, data_ed)
        if daily.height == 0:
            print("    无日线数据，跳过")
            continue

        # 获取前复权因子
        members_ts = [sym_local_to_tushare(s) for s in members]
        adj = get_adj_factors(ch, members_ts, data_sd, data_ed)
        print(f"    日线: {daily.height} rows, 复权因子: {adj.height} rows")

        # 计算前向收益
        fwd = compute_forward_returns(daily, adj, sig_date, FORWARD_WINDOWS)
        if fwd.height == 0:
            print("    无法计算前向收益，跳过")
            continue

        fwd = fwd.with_columns(
            pl.lit(max_prob).alias("signal_prob"),
        )

        # 添加 tushare symbol 列以便查看
        fwd = fwd.with_columns(
            pl.col("symbol").map_elements(
                lambda s: sym_local_to_tushare(s), return_dtype=pl.Utf8,
            ).alias("ts_symbol"),
        )

        print(f"    前向收益计算完成: {fwd.height} 只")

        # 展示 Top N
        for w in FORWARD_WINDOWS:
            col = f"ret_T{w}"
            if col in fwd.columns:
                valid = fwd.filter(pl.col(col).is_not_null())
                if valid.height > 0:
                    top = valid.sort(col, descending=True).head(3)
                    best = top.row(0, named=True)
                    print(f"    T+{w:2d} best: {best['ts_symbol']} {best[col]:+.2%}")

        all_elasticity.append(fwd)

    if not all_elasticity:
        print("\nNo elasticity data!")
        return

    full = pl.concat(all_elasticity, how="diagonal")
    print(f"\n[Step 3] 汇总 {full.height} 条记录")

    # ── 生成报告 ────────────────────────────────────────────────────────
    report = []
    report.append("# 国家队买入信号 → 个股弹性回测报告")
    report.append("")
    report.append(f"- **标的池**: 沪深300 成分股")
    report.append(f"- **信号因子**: NT_Buy_Prob v2 (threshold > {SIGNAL_THRESHOLD:.0%})")
    report.append(f"- **区间**: {START_YEAR}-01-01 ~ {END_DATE}")
    report.append(f"- **信号日数**: {len(signals)}")
    report.append(f"- **前向窗口**: T+{', T+'.join(str(w) for w in FORWARD_WINDOWS)}")
    report.append("")

    # 3a. 每个信号日的整体弹性统计
    report.append("## 信号日整体弹性统计")
    report.append("")
    header = "| 信号日 | 概率 | 股票数"
    for w in FORWARD_WINDOWS:
        header += f" | T+{w} 均值 | T+{w} 中位 | T+{w}>0% |"
    report.append(header)
    report.append("|" + "---|" * (3 + len(FORWARD_WINDOWS) * 3))

    for sig_date, max_prob in signals:
        chunk = full.filter(pl.col("signal_date") == sig_date)
        if chunk.height == 0:
            continue
        row_str = f"| {sig_date} | {max_prob:.1%} | {chunk.height}"
        for w in FORWARD_WINDOWS:
            col = f"ret_T{w}"
            if col in chunk.columns:
                valid = chunk.filter(pl.col(col).is_not_null())
                if valid.height > 0:
                    mean_r = valid[col].mean()
                    median_r = valid[col].median()
                    pct_pos = valid.filter(pl.col(col) > 0).height / valid.height
                    row_str += f" | {mean_r:+.2%} | {median_r:+.2%} | {pct_pos:.0%}"
                else:
                    row_str += " | - | - | -"
            else:
                row_str += " | - | - | -"
        row_str += " |"
        report.append(row_str)

    # 3b. 每个信号日的 Top 20 弹性最强个股
    for w in [5, 10, 20]:
        col = f"ret_T{w}"
        report.append("")
        report.append(f"## T+{w} 弹性 Top {TOP_N}（按信号日）")
        report.append("")

        for sig_date, max_prob in signals:
            chunk = full.filter(
                (pl.col("signal_date") == sig_date) & pl.col(col).is_not_null()
            )
            if chunk.height == 0:
                continue

            report.append(f"### {sig_date} (prob={max_prob:.1%})")
            report.append("")
            report.append(f"| # | 股票 | T+{w} 收益 | T+1 | T+3 | T+5 | 信号日收盘 |")
            report.append("|---|------|----------|-----|-----|-----|----------|")

            top = chunk.sort(col, descending=True).head(TOP_N)
            for rank, row in enumerate(top.iter_rows(named=True), 1):
                t1 = f"{row.get('ret_T1', 0) or 0:+.2%}"
                t3 = f"{row.get('ret_T3', 0) or 0:+.2%}"
                t5 = f"{row.get('ret_T5', 0) or 0:+.2%}"
                tw = f"{row[col]:+.2%}"
                report.append(
                    f"| {rank} | {row['ts_symbol']} | {tw} | {t1} | {t3} | {t5} | {row['close_t0']:.2f} |"
                )
            report.append("")

    # 3c. 跨信号日的弹性王者 — 多次出现在 Top 20 的个股
    report.append("## 弹性王者 — 多次信号中反弹最强的个股")
    report.append("")
    for w in [5, 10]:
        col = f"ret_T{w}"
        report.append(f"### T+{w} 视角")
        report.append("")

        # 对每个信号日取 Top 20
        top_records = []
        for sig_date, _ in signals:
            chunk = full.filter(
                (pl.col("signal_date") == sig_date) & pl.col(col).is_not_null()
            )
            if chunk.height > 0:
                top = chunk.sort(col, descending=True).head(TOP_N)
                top_records.append(top)

        if top_records:
            combined = pl.concat(top_records, how="diagonal")
            freq = (
                combined
                .group_by("ts_symbol")
                .agg(
                    pl.col(col).count().alias("出现次数"),
                    pl.col(col).mean().alias("平均收益"),
                    pl.col(col).max().alias("最大收益"),
                    pl.col("signal_date").min().alias("首次信号"),
                    pl.col("signal_date").max().alias("末次信号"),
                )
                .sort("出现次数", descending=True)
                .head(30)
            )

            report.append(f"| 股票 | 出现次数/{len(signals)} | 平均 T+{w} | 最大 T+{w} | 首次 | 末次 |")
            report.append("|------|---------|---------|---------|------|------|")
            for row in freq.iter_rows(named=True):
                report.append(
                    f"| {row['ts_symbol']} | {row['出现次数']} | {row['平均收益']:+.2%} | {row['最大收益']:+.2%} | {row['首次信号']} | {row['末次信号']} |"
                )
            report.append("")

    # 写入
    report_text = "\n".join(report) + "\n"
    out_path = Path(__file__).resolve().parent.parent / "doc" / "nt_buy_elasticity_report.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report_text, encoding="utf-8")
    print(f"\n报告已写入: {out_path}")

    # 也保存原始数据为 parquet
    parquet_path = out_path.with_suffix(".parquet")
    full.write_parquet(parquet_path)
    print(f"原始数据: {parquet_path}")


if __name__ == "__main__":
    main()
