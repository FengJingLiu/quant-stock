"""
五大实验: 诊断 alpha 来源与执行错位

实验 1: 收益拆解 — 事件日尾盘买 vs 次日开盘买 vs 次日 VWAP 买
实验 2: 候选池扩展 — HS300 vs CSI500 vs CSI500+1000
实验 3: 打分器改写 — 追强度 vs 未透支扩散度
实验 4: 退出改写 — 固定 2-3 天 + 事件 VWAP 止损
实验 5: 随机基准 + 去极值检验 — 打分 vs 随机 vs 单因子 + leave-one-year-out

所有实验复用 detect_signal_dates() 信号日列表。

输出:
  - 逐实验 DataFrame 打印
  - doc/five_experiments_report.md
"""

from __future__ import annotations

import os
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

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

# ── CH Client ────────────────────────────────────────────────────────────


def get_ch() -> clickhouse_connect.driver.Client:
    return clickhouse_connect.get_client(**CH_HTTP_KWARGS)


# ── Helper: 获取指数成分股 ────────────────────────────────────────────────


def get_index_members(ch, index_code: str, signal_date: date) -> list[str]:
    """获取指定指数最近一期成分股 (返回 klines_1m_stock 格式)。"""
    r = ch.query(
        """
        SELECT con_code FROM dim_index_weights
        WHERE index_code = %(idx)s
          AND trade_date = (
              SELECT max(trade_date) FROM dim_index_weights
              WHERE index_code = %(idx)s AND trade_date <= %(d)s
          )
        """,
        parameters={"idx": index_code, "d": signal_date},
    )
    return [sym_tushare_to_local(row[0]) for row in r.result_rows]


# ── Helper: 获取带前复权的日线 ────────────────────────────────────────────


def get_adj_daily(
    ch, symbols: list[str], start_date: date, end_date: date,
) -> pl.DataFrame:
    """获取个股日线聚合 + 前复权 adj_close。"""
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


# ── Helper: 获取 ETF 日线 (作为基准) ─────────────────────────────────────


def get_etf_daily_bench(ch, start_date: date, end_date: date) -> pl.DataFrame:
    """ETF 510300 日线。"""
    r = ch.query_arrow(
        """
        SELECT trade_date,
               argMin(open, datetime) AS etf_open,
               argMax(close, datetime) AS etf_close,
               sum(amount) AS etf_amount
        FROM klines_1m_etf
        WHERE symbol = '510300.SH'
          AND trade_date BETWEEN %(sd)s AND %(ed)s
        GROUP BY trade_date
        ORDER BY trade_date
        """,
        parameters={"sd": start_date, "ed": end_date},
    )
    if r.num_rows == 0:
        return pl.DataFrame()
    return pl.from_arrow(r).with_columns(
        pl.col("trade_date").cast(pl.Date),
        pl.col("etf_open").cast(pl.Float64),
        pl.col("etf_close").cast(pl.Float64),
    )


# ── Helper: 交易日历 ─────────────────────────────────────────────────────


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


# ── Helper: 尾盘价格 (最后 30 分钟 VWAP) ────────────────────────────────


def get_stock_close_session(
    ch, symbols: list[str], trade_date: date,
) -> pl.DataFrame:
    """获取个股事件日最后 30 分钟的 VWAP 和收盘价。"""
    if not symbols:
        return pl.DataFrame()
    r = ch.query_arrow(
        """
        SELECT symbol,
               argMax(close, datetime) AS eod_close,
               sum(amount) AS eod_amount,
               sum(volume) AS eod_volume
        FROM klines_1m_stock
        WHERE trade_date = %(d)s
          AND symbol IN %(syms)s
          AND toHour(datetime) >= 14 AND toMinute(datetime) >= 30
        GROUP BY symbol
        """,
        parameters={"d": trade_date, "syms": symbols},
    )
    if r.num_rows == 0:
        return pl.DataFrame(schema={"symbol": pl.Utf8, "eod_close": pl.Float64, "eod_vwap": pl.Float64})
    df = pl.from_arrow(r)
    return df.with_columns(
        pl.col("eod_close").cast(pl.Float64),
        pl.col("eod_amount").cast(pl.Float64),
        pl.col("eod_volume").cast(pl.Float64),
    ).with_columns(
        pl.when(pl.col("eod_volume") > 0)
        .then(pl.col("eod_amount") / (pl.col("eod_volume") * 100))  # 手→股
        .otherwise(pl.col("eod_close"))
        .alias("eod_vwap"),
    ).select("symbol", "eod_close", "eod_vwap")


# ── Helper: 日开盘价 ─────────────────────────────────────────────────────


def get_stock_open_prices(
    ch, symbols: list[str], trade_date: date,
) -> pl.DataFrame:
    """获取个股指定日的开盘价。"""
    if not symbols:
        return pl.DataFrame()
    r = ch.query_arrow(
        """
        SELECT symbol,
               argMin(open, datetime) AS day_open
        FROM klines_1m_stock
        WHERE trade_date = %(d)s
          AND symbol IN %(syms)s
        GROUP BY symbol
        """,
        parameters={"d": trade_date, "syms": symbols},
    )
    if r.num_rows == 0:
        return pl.DataFrame(schema={"symbol": pl.Utf8, "day_open": pl.Float64})
    df = pl.from_arrow(r)
    return df.with_columns(pl.col("day_open").cast(pl.Float64))


# ── Helper: 日 VWAP ──────────────────────────────────────────────────────


def get_stock_vwap(
    ch, symbols: list[str], trade_date: date,
) -> pl.DataFrame:
    """获取个股指定日 VWAP。"""
    if not symbols:
        return pl.DataFrame()
    r = ch.query_arrow(
        """
        SELECT symbol,
               sum(amount) AS d_amount,
               sum(volume) AS d_volume
        FROM klines_1m_stock
        WHERE trade_date = %(d)s
          AND symbol IN %(syms)s
        GROUP BY symbol
        """,
        parameters={"d": trade_date, "syms": symbols},
    )
    if r.num_rows == 0:
        return pl.DataFrame(schema={"symbol": pl.Utf8, "day_vwap": pl.Float64})
    df = pl.from_arrow(r)
    return df.with_columns(
        pl.col("d_amount").cast(pl.Float64),
        pl.col("d_volume").cast(pl.Float64),
    ).with_columns(
        pl.when(pl.col("d_volume") > 0)
        .then(pl.col("d_amount") / (pl.col("d_volume") * 100))
        .otherwise(None)
        .alias("day_vwap"),
    ).select("symbol", "day_vwap")


# ══════════════════════════════════════════════════════════════════════════
# 实验 1: 收益拆解 — 入场时机诊断
# ══════════════════════════════════════════════════════════════════════════


def experiment_1_entry_timing(
    ch, signals: list[tuple[date, float]],
) -> pl.DataFrame:
    """
    对每个信号日, 计算 4 种入场方式的 T+3 收益:
      A. 事件日收盘买 → 次日开盘卖 (隔夜 gap)
      B. 事件日收盘 → T+3 收盘 (事件日尾盘买)
      C. 次日开盘 → T+3 收盘 (当前策略)
      D. 次日 VWAP → T+3 收盘

    额外: 事件日尾盘 → T+1 收盘 (最短 alpha)
    """
    print("\n" + "=" * 70)
    print("实验 1: 入场时机收益拆解")
    print("=" * 70)

    rows = []
    for sig_date, prob in signals:
        members = get_hs300_members(ch, sig_date)
        if not members:
            continue

        # 获取日历
        cal = get_trading_dates(ch, sig_date - timedelta(days=5),
                                sig_date + timedelta(days=30))
        if sig_date not in cal:
            continue
        t0_idx = cal.index(sig_date)
        if t0_idx + 4 >= len(cal):
            continue

        t0 = cal[t0_idx]
        t1 = cal[t0_idx + 1]
        t3 = cal[t0_idx + 3]

        # 获取带复权的日线
        daily = get_adj_daily(ch, members, sig_date - timedelta(days=5),
                              t3 + timedelta(days=5))
        if daily.height == 0:
            continue

        # T0 收盘, T1 开盘/收盘/VWAP, T3 收盘
        t0_close_df = (
            daily.filter(pl.col("trade_date") == t0)
            .select("symbol", pl.col("adj_close").alias("t0_close"))
        )
        t1_data = daily.filter(pl.col("trade_date") == t1)
        t1_open_raw = get_stock_open_prices(ch, members, t1)
        # 需要用 adj_factor 调整 open
        t1_adj = daily.filter(pl.col("trade_date") == t1).select(
            "symbol", pl.col("adj_factor").fill_null(1.0).alias("t1_adj"),
        )
        t1_open_df = t1_open_raw.join(t1_adj, on="symbol", how="left").with_columns(
            (pl.col("day_open") * pl.col("t1_adj").fill_null(1.0)).alias("t1_open"),
        ).select("symbol", "t1_open")

        t1_close_df = (
            daily.filter(pl.col("trade_date") == t1)
            .select("symbol", pl.col("adj_close").alias("t1_close"))
        )
        t1_vwap_raw = get_stock_vwap(ch, members, t1)
        t1_vwap_df = t1_vwap_raw.join(t1_adj, on="symbol", how="left").with_columns(
            (pl.col("day_vwap").fill_null(0.0) * pl.col("t1_adj").fill_null(1.0)).alias("t1_vwap"),
        ).select("symbol", "t1_vwap")

        t3_close_df = (
            daily.filter(pl.col("trade_date") == t3)
            .select("symbol", pl.col("adj_close").alias("t3_close"))
        )

        # Merge
        merged = (
            t0_close_df
            .join(t1_open_df, on="symbol", how="inner")
            .join(t1_close_df, on="symbol", how="inner")
            .join(t1_vwap_df, on="symbol", how="inner")
            .join(t3_close_df, on="symbol", how="inner")
        )

        # 过滤零价格 (停牌/数据缺失)
        merged = merged.filter(
            (pl.col("t0_close") > 0)
            & (pl.col("t1_open") > 0)
            & (pl.col("t1_close") > 0)
            & (pl.col("t3_close") > 0)
        )

        if merged.height == 0:
            continue

        # 计算各段收益 (等权平均跨所有成分股 — 全篮子基准)
        merged = merged.with_columns(
            # A: 隔夜 gap (T0 close → T1 open)
            (pl.col("t1_open") / pl.col("t0_close") - 1.0).alias("ret_overnight"),
            # B: T0 close → T3 close (尾盘买, 持有3天)
            (pl.col("t3_close") / pl.col("t0_close") - 1.0).alias("ret_eod_to_t3"),
            # C: T1 open → T3 close (当前策略)
            (pl.col("t3_close") / pl.col("t1_open") - 1.0).alias("ret_t1open_to_t3"),
            # D: T1 VWAP → T3 close
            pl.when(pl.col("t1_vwap") > 0)
            .then(pl.col("t3_close") / pl.col("t1_vwap") - 1.0)
            .otherwise(None)
            .alias("ret_t1vwap_to_t3"),
            # E: T0 close → T1 close (最短)
            (pl.col("t1_close") / pl.col("t0_close") - 1.0).alias("ret_eod_to_t1"),
        )

        n = merged.height
        row = {
            "signal_date": sig_date,
            "signal_prob": prob,
            "n_stocks": n,
            "ret_overnight": merged["ret_overnight"].mean(),
            "ret_eod_to_t3": merged["ret_eod_to_t3"].mean(),
            "ret_t1open_to_t3": merged["ret_t1open_to_t3"].mean(),
            "ret_t1vwap_to_t3": merged["ret_t1vwap_to_t3"].mean(),
            "ret_eod_to_t1": merged["ret_eod_to_t1"].mean(),
        }
        rows.append(row)
        print(f"  {sig_date}: overnight={row['ret_overnight']:+.3%}, "
              f"eod→T3={row['ret_eod_to_t3']:+.3%}, "
              f"T1open→T3={row['ret_t1open_to_t3']:+.3%}")

    if not rows:
        return pl.DataFrame()

    df = pl.DataFrame(rows)

    # 输出汇总
    print("\n── 收益拆解汇总 ──")
    for col in ["ret_overnight", "ret_eod_to_t3", "ret_t1open_to_t3",
                "ret_t1vwap_to_t3", "ret_eod_to_t1"]:
        vals = df[col].drop_nulls()
        if vals.len() > 0:
            print(f"  {col:25s}: mean={vals.mean():+.3%}, "
                  f"median={vals.median():+.3%}, "
                  f"win_rate={vals.filter(vals > 0).len() / vals.len():.0%}, "
                  f"sum={vals.sum():+.2%}")

    return df


# ══════════════════════════════════════════════════════════════════════════
# 实验 2: 候选池扩展
# ══════════════════════════════════════════════════════════════════════════


def experiment_2_pool_expansion(
    ch, signals: list[tuple[date, float]],
) -> pl.DataFrame:
    """
    对比 3 种候选池的等权 T+3 收益:
      A. HS300
      B. CSI500
      C. HS300 + CSI500
    """
    print("\n" + "=" * 70)
    print("实验 2: 候选池扩展 (HS300 vs CSI500 vs HS300+500)")
    print("=" * 70)

    pools = {
        "HS300": "399300.SZ",
        "CSI500": "000905.SH",
    }
    rows = []

    for sig_date, prob in signals:
        cal = get_trading_dates(ch, sig_date - timedelta(days=5),
                                sig_date + timedelta(days=20))
        if sig_date not in cal:
            continue
        t0_idx = cal.index(sig_date)
        if t0_idx + 3 >= len(cal):
            continue
        t0 = cal[t0_idx]
        t3 = cal[t0_idx + 3]

        pool_members: dict[str, list[str]] = {}
        for name, idx_code in pools.items():
            members = get_index_members(ch, idx_code, sig_date)
            pool_members[name] = members

        pool_members["HS300+500"] = list(
            set(pool_members.get("HS300", []))
            | set(pool_members.get("CSI500", []))
        )

        row: dict = {"signal_date": sig_date, "signal_prob": prob}

        for pool_name, members in pool_members.items():
            if not members:
                row[f"ret_{pool_name}"] = None
                row[f"n_{pool_name}"] = 0
                continue

            daily = get_adj_daily(ch, members, t0 - timedelta(days=5),
                                  t3 + timedelta(days=5))
            if daily.height == 0:
                row[f"ret_{pool_name}"] = None
                row[f"n_{pool_name}"] = 0
                continue

            t0_close = daily.filter(pl.col("trade_date") == t0).select("symbol", "adj_close")
            t3_close = daily.filter(pl.col("trade_date") == t3).select(
                "symbol", pl.col("adj_close").alias("t3_close"),
            )
            merged = (
                t0_close.join(t3_close, on="symbol", how="inner")
                .filter((pl.col("adj_close") > 0) & (pl.col("t3_close") > 0))
                .with_columns(
                    (pl.col("t3_close") / pl.col("adj_close") - 1.0).alias("ret"),
                )
            )
            row[f"ret_{pool_name}"] = merged["ret"].mean() if merged.height > 0 else None
            row[f"n_{pool_name}"] = merged.height

        rows.append(row)
        print(f"  {sig_date}: "
              + " | ".join(f"{k}={row.get(f'ret_{k}', 0):+.3%}"
                           for k in pool_members if row.get(f"ret_{k}") is not None))

    if not rows:
        return pl.DataFrame()

    df = pl.DataFrame(rows)

    print("\n── 候选池对比汇总 (T0 close → T3 close, 等权全篮子) ──")
    for pool_name in ["HS300", "CSI500", "HS300+500"]:
        col = f"ret_{pool_name}"
        if col in df.columns:
            vals = df[col].drop_nulls()
            if vals.len() > 0:
                print(f"  {pool_name:15s}: mean={vals.mean():+.3%}, "
                      f"win_rate={vals.filter(vals > 0).len() / vals.len():.0%}, "
                      f"sum={vals.sum():+.2%}")

    return df


# ══════════════════════════════════════════════════════════════════════════
# 实验 3: 打分器改写 — 未透支扩散度
# ══════════════════════════════════════════════════════════════════════════


def _score_v2_response(ch, members: list[str], signal_date: date) -> pl.DataFrame:
    """原始打分器逻辑: event_response (午后超额) 排序。"""
    from src.national_team.elastic_scorer import ElasticScorer
    scorer = ElasticScorer()
    return scorer.score(ch, signal_date, members)


def _score_diffusion(ch, members: list[str], signal_date: date) -> pl.DataFrame:
    """
    新打分器: "未透支扩散度"

    逻辑: 事件日先动了 (午后有超额) 但没有透支 (不是涨停/换手爆炸)
    diffusion_score = event_response * (1 - exhaustion)

    exhaustion = 涨幅 rank * turnover rank 的乘积
    → 涨太多/换手太高 = 透支
    """
    lookback_sd = signal_date - timedelta(days=50)

    # 日线
    daily = get_stock_daily_agg(ch, members, lookback_sd, signal_date)
    if daily.height == 0:
        return pl.DataFrame(schema={
            "symbol": pl.Utf8, "elastic_score": pl.Float64,
            "event_response": pl.Float64, "exhaustion": pl.Float64,
        })

    # 事件日数据
    event_day = daily.filter(pl.col("trade_date") == signal_date)
    if event_day.height == 0:
        return pl.DataFrame(schema={
            "symbol": pl.Utf8, "elastic_score": pl.Float64,
            "event_response": pl.Float64, "exhaustion": pl.Float64,
        })

    # 事件日涨幅
    event_day = event_day.with_columns(
        (pl.col("daily_close") / pl.col("daily_open") - 1.0).alias("day_ret"),
    )

    # ADV20
    adv = (
        daily.filter(pl.col("trade_date") < signal_date)
        .group_by("symbol")
        .agg(pl.col("daily_amount").tail(20).mean().alias("adv20"))
    )

    # 事件日换手比率 = amount / adv20
    scored = event_day.select("symbol", "day_ret", "daily_amount").join(
        adv, on="symbol", how="left",
    ).with_columns(
        (pl.col("daily_amount") / pl.col("adv20").clip(lower_bound=1.0))
        .alias("turnover_ratio"),
    )

    # 午后超额 (用 elastic_scorer 查询复用)
    from src.national_team.elastic_scorer import ElasticScorer
    scorer = ElasticScorer()
    afternoon = scorer._query_afternoon_alpha(ch, members, signal_date, "000300")
    idx_ret = scorer._query_index_daily_ret(ch, "000300", signal_date)
    if afternoon.height > 0:
        afternoon = afternoon.with_columns(
            (pl.col("stock_pm_ret") - idx_ret).alias("event_response"),
        )
    else:
        afternoon = pl.DataFrame(schema={"symbol": pl.Utf8, "event_response": pl.Float64})

    scored = scored.join(
        afternoon.select("symbol", "event_response"),
        on="symbol", how="left",
    ).with_columns(pl.col("event_response").fill_null(0.0))

    n = scored.height
    if n < 2:
        scored = scored.with_columns(
            pl.col("event_response").alias("elastic_score"),
            pl.lit(0.0).alias("exhaustion"),
        )
        return scored.select("symbol", "elastic_score", "event_response", "exhaustion")

    # Rank percentile
    scored = scored.with_columns(
        (pl.col("day_ret").rank("ordinal").cast(pl.Float64) / n).alias("ret_pct"),
        (pl.col("turnover_ratio").fill_null(0.0).rank("ordinal").cast(pl.Float64) / n)
        .alias("turnover_pct"),
    )

    # exhaustion = ret_pct * turnover_pct (两个都高 = 透支)
    scored = scored.with_columns(
        (pl.col("ret_pct") * pl.col("turnover_pct")).alias("exhaustion"),
    )

    # diffusion_score = response × (1 - exhaustion)
    # 只取 response > 0 的 (有被点火)
    scored = scored.with_columns(
        (
            pl.col("event_response").clip(lower_bound=0.0)
            * (1.0 - pl.col("exhaustion"))
        ).alias("elastic_score"),
    )

    return (
        scored.select("symbol", "elastic_score", "event_response", "exhaustion")
        .sort("elastic_score", descending=True)
    )


def experiment_3_scorer_comparison(
    ch, signals: list[tuple[date, float]], top_n: int = 5,
) -> pl.DataFrame:
    """
    对比原始打分器 vs 未透支扩散打分器, 各选 TopN, 计算 T0→T3 收益。
    """
    print("\n" + "=" * 70)
    print("实验 3: 打分器对比 — 追强度 vs 未透支扩散")
    print("=" * 70)

    rows = []
    for sig_date, prob in signals:
        members = get_hs300_members(ch, sig_date)
        if not members:
            continue

        cal = get_trading_dates(ch, sig_date - timedelta(days=5),
                                sig_date + timedelta(days=20))
        if sig_date not in cal:
            continue
        t0_idx = cal.index(sig_date)
        if t0_idx + 3 >= len(cal):
            continue
        t0 = cal[t0_idx]
        t3 = cal[t0_idx + 3]

        # 获取日线 (for forward returns)
        daily = get_adj_daily(ch, members, t0 - timedelta(days=5),
                              t3 + timedelta(days=5))
        if daily.height == 0:
            continue

        t0_close = daily.filter(pl.col("trade_date") == t0).select(
            "symbol", pl.col("adj_close").alias("t0_close"),
        )
        t3_close = daily.filter(pl.col("trade_date") == t3).select(
            "symbol", pl.col("adj_close").alias("t3_close"),
        )
        fwd = (
            t0_close.join(t3_close, on="symbol", how="inner")
            .filter((pl.col("t0_close") > 0) & (pl.col("t3_close") > 0))
            .with_columns(
                (pl.col("t3_close") / pl.col("t0_close") - 1.0).alias("fwd_ret"),
            )
        )

        # 打分器 A: 原始
        scored_v2 = _score_v2_response(ch, members, sig_date)
        top_v2 = scored_v2.head(top_n)["symbol"].to_list() if scored_v2.height > 0 else []

        # 打分器 B: 未透支
        scored_diff = _score_diffusion(ch, members, sig_date)
        top_diff = scored_diff.head(top_n)["symbol"].to_list() if scored_diff.height > 0 else []

        def avg_ret(syms):
            if not syms:
                return None
            sel = fwd.filter(pl.col("symbol").is_in(syms))
            return sel["fwd_ret"].mean() if sel.height > 0 else None

        row = {
            "signal_date": sig_date,
            "ret_scorer_v2": avg_ret(top_v2),
            "ret_diffusion": avg_ret(top_diff),
            "ret_basket": fwd["fwd_ret"].mean() if fwd.height > 0 else None,
            "overlap": len(set(top_v2) & set(top_diff)),
        }
        rows.append(row)
        print(f"  {sig_date}: v2_top{top_n}={row['ret_scorer_v2']:+.3%}, "
              f"diff_top{top_n}={row['ret_diffusion']:+.3%}, "
              f"basket={row['ret_basket']:+.3%}"
              if row['ret_scorer_v2'] is not None else f"  {sig_date}: skip")

    if not rows:
        return pl.DataFrame()

    df = pl.DataFrame(rows)

    print("\n── 打分器对比汇总 (T0→T3, TopN 等权) ──")
    for col, label in [
        ("ret_scorer_v2", "原始打分器"),
        ("ret_diffusion", "未透支扩散"),
        ("ret_basket", "全篮子等权"),
    ]:
        vals = df[col].drop_nulls()
        if vals.len() > 0:
            print(f"  {label:15s}: mean={vals.mean():+.3%}, "
                  f"win_rate={vals.filter(vals > 0).len() / vals.len():.0%}, "
                  f"sum={vals.sum():+.2%}")

    return df


# ══════════════════════════════════════════════════════════════════════════
# 实验 4: 退出改写 — 固定天数 + 事件 VWAP 止损
# ══════════════════════════════════════════════════════════════════════════


def experiment_4_exit_rules(
    ch, signals: list[tuple[date, float]], top_n: int = 5,
) -> pl.DataFrame:
    """
    对比退出规则 (用"未透支扩散"打分器选股):
      A. 固定 2 天
      B. 固定 3 天
      C. 固定 3 天 + 跌破事件日 VWAP 止损 (T+1 可执行)
    入场: T0 收盘买入
    """
    print("\n" + "=" * 70)
    print("实验 4: 退出规则对比 (T0 收盘入场)")
    print("=" * 70)

    rows = []
    for sig_date, prob in signals:
        members = get_hs300_members(ch, sig_date)
        if not members:
            continue

        cal = get_trading_dates(ch, sig_date - timedelta(days=5),
                                sig_date + timedelta(days=30))
        if sig_date not in cal:
            continue
        t0_idx = cal.index(sig_date)
        if t0_idx + 4 >= len(cal):
            continue

        t0 = cal[t0_idx]

        # 选股: 未透支
        scored = _score_diffusion(ch, members, sig_date)
        top_syms = scored.head(top_n)["symbol"].to_list() if scored.height > 0 else []
        if not top_syms:
            continue

        # 日线
        daily = get_adj_daily(ch, top_syms, t0 - timedelta(days=5),
                              cal[min(t0_idx + 5, len(cal) - 1)] + timedelta(days=5))
        if daily.height == 0:
            continue

        # 事件日 VWAP (止损基准)
        event_vwap = get_stock_vwap(ch, top_syms, t0)
        # 需复权
        t0_adj = daily.filter(pl.col("trade_date") == t0).select(
            "symbol", pl.col("adj_factor").fill_null(1.0).alias("t0_adj"),
        )
        if event_vwap.height > 0 and t0_adj.height > 0:
            event_vwap = event_vwap.join(t0_adj, on="symbol", how="left").with_columns(
                (pl.col("day_vwap").fill_null(0.0) * pl.col("t0_adj").fill_null(1.0)).alias("event_vwap"),
            ).select("symbol", "event_vwap")

        row: dict = {"signal_date": sig_date}

        for rule_name, hold_days, use_vwap_stop in [
            ("fixed_2", 2, False),
            ("fixed_3", 3, False),
            ("fixed_3_vwap_stop", 3, True),
        ]:
            rets = []
            for sym in top_syms:
                sym_daily = daily.filter(pl.col("symbol") == sym).sort("trade_date")
                sym_dates = sym_daily["trade_date"].to_list()
                sym_close = sym_daily["adj_close"].to_list()

                if t0 not in sym_dates:
                    continue
                entry_idx = sym_dates.index(t0)
                entry_price = sym_close[entry_idx]
                if entry_price is None or entry_price <= 0:
                    continue

                # VWAP 止损线
                vwap_line = None
                if use_vwap_stop and event_vwap.height > 0:
                    v = event_vwap.filter(pl.col("symbol") == sym)
                    if v.height > 0:
                        vwap_line = v["event_vwap"][0]

                exit_price = None
                for d_off in range(1, hold_days + 1):
                    ci = entry_idx + d_off
                    if ci >= len(sym_dates):
                        break

                    # VWAP 止损: 如果当日收盘跌破事件VWAP, 提前退出
                    if vwap_line and vwap_line > 0 and sym_close[ci] is not None:
                        if sym_close[ci] < vwap_line * 0.99:  # 1% 容忍
                            exit_price = sym_close[ci]
                            break

                    if d_off == hold_days:
                        exit_price = sym_close[ci]

                if exit_price is not None and entry_price > 0:
                    rets.append(exit_price / entry_price - 1.0)

            row[f"ret_{rule_name}"] = sum(rets) / len(rets) if rets else None
            row[f"n_{rule_name}"] = len(rets)

        rows.append(row)
        print(f"  {sig_date}: " +
              " | ".join(f"{k.replace('ret_','')}={row[k]:+.3%}"
                         for k in row if k.startswith("ret_") and row[k] is not None))

    if not rows:
        return pl.DataFrame()

    df = pl.DataFrame(rows)

    print("\n── 退出规则对比汇总 (T0 收盘入场, TopN 等权) ──")
    for rule in ["fixed_2", "fixed_3", "fixed_3_vwap_stop"]:
        col = f"ret_{rule}"
        if col in df.columns:
            vals = df[col].drop_nulls()
            if vals.len() > 0:
                print(f"  {rule:25s}: mean={vals.mean():+.3%}, "
                      f"win_rate={vals.filter(vals > 0).len() / vals.len():.0%}, "
                      f"sum={vals.sum():+.2%}")

    return df


# ══════════════════════════════════════════════════════════════════════════
# 实验 5: 随机基准 + 去极值检验
# ══════════════════════════════════════════════════════════════════════════


def experiment_5_baselines(
    ch, signals: list[tuple[date, float]],
    top_n: int = 5, n_random_trials: int = 200,
) -> pl.DataFrame:
    """
    4 个基准对比 + leave-one-year-out:
      基准 1: 随机 TopN (蒙特卡洛 200 次)
      基准 2: 全篮子等权
      基准 3: 纯事件响应排序 (单因子)
      基准 4: 未透支扩散打分器

    额外: 去掉 2015 的累计收益
    """
    print("\n" + "=" * 70)
    print("实验 5: 随机基准 + 去极值检验")
    print("=" * 70)

    all_rows = []

    for sig_date, prob in signals:
        members = get_hs300_members(ch, sig_date)
        if not members:
            continue

        cal = get_trading_dates(ch, sig_date - timedelta(days=5),
                                sig_date + timedelta(days=20))
        if sig_date not in cal:
            continue
        t0_idx = cal.index(sig_date)
        if t0_idx + 3 >= len(cal):
            continue
        t0 = cal[t0_idx]
        t3 = cal[t0_idx + 3]

        # 日线
        daily = get_adj_daily(ch, members, t0 - timedelta(days=60),
                              t3 + timedelta(days=5))
        if daily.height == 0:
            continue

        # 前向收益
        t0_close = daily.filter(pl.col("trade_date") == t0).select(
            "symbol", pl.col("adj_close").alias("t0_close"),
        )
        t3_close = daily.filter(pl.col("trade_date") == t3).select(
            "symbol", pl.col("adj_close").alias("t3_close"),
        )
        fwd = (
            t0_close.join(t3_close, on="symbol", how="inner")
            .filter((pl.col("t0_close") > 0) & (pl.col("t3_close") > 0))
            .with_columns(
                (pl.col("t3_close") / pl.col("t0_close") - 1.0).alias("fwd_ret"),
            )
        )
        if fwd.height < top_n:
            continue

        all_syms = fwd["symbol"].to_list()
        all_rets = {
            s: r for s, r in zip(fwd["symbol"].to_list(), fwd["fwd_ret"].to_list())
            if r is not None and not (isinstance(r, float) and (r != r or abs(r) == float("inf")))
        }
        if len(all_rets) < top_n:
            continue

        # 基准 1: 随机 TopN (蒙特卡洛)
        valid_syms = list(all_rets.keys())
        random_rets = []
        for _ in range(n_random_trials):
            sample = random.sample(valid_syms, min(top_n, len(valid_syms)))
            avg = sum(all_rets[s] for s in sample) / len(sample)
            random_rets.append(avg)
        random_mean = sum(random_rets) / len(random_rets) if random_rets else 0.0

        # 基准 2: 全篮子等权
        basket_ret = fwd["fwd_ret"].mean()

        # 基准 3: 纯事件响应排序
        scored_v2 = _score_v2_response(ch, members, sig_date)
        if scored_v2.height > 0:
            top_resp = scored_v2.head(top_n)["symbol"].to_list()
            resp_ret = fwd.filter(pl.col("symbol").is_in(top_resp))["fwd_ret"].mean()
        else:
            resp_ret = None

        # 基准 4: 未透支扩散
        scored_diff = _score_diffusion(ch, members, sig_date)
        if scored_diff.height > 0:
            top_diff = scored_diff.head(top_n)["symbol"].to_list()
            diff_ret = fwd.filter(pl.col("symbol").is_in(top_diff))["fwd_ret"].mean()
        else:
            diff_ret = None

        row = {
            "signal_date": sig_date,
            "year": sig_date.year,
            "ret_random_mean": random_mean,
            "ret_basket": basket_ret,
            "ret_response_top": resp_ret,
            "ret_diffusion_top": diff_ret,
        }
        all_rows.append(row)
        print(f"  {sig_date}: random={random_mean:+.3%}, basket={basket_ret:+.3%}, "
              f"resp={resp_ret:+.3%}, diff={diff_ret:+.3%}"
              if resp_ret is not None else f"  {sig_date}: partial")

    if not all_rows:
        return pl.DataFrame()

    df = pl.DataFrame(all_rows)

    print("\n── 基准对比汇总 (T0→T3, TopN 等权) ──")
    for col, label in [
        ("ret_random_mean", "随机TopN均值"),
        ("ret_basket", "全篮子等权"),
        ("ret_response_top", "纯响应排序"),
        ("ret_diffusion_top", "未透支扩散"),
    ]:
        vals = df[col].drop_nulls()
        if vals.len() > 0:
            print(f"  {label:15s}: mean={vals.mean():+.3%}, "
                  f"sum={vals.sum():+.2%}, "
                  f"win_rate={vals.filter(vals > 0).len() / vals.len():.0%}")

    # Leave-one-year-out
    print("\n── 去极值检验 (去除特定年份后的累计收益) ──")
    years = sorted(df["year"].unique().to_list())
    for col in ["ret_diffusion_top", "ret_response_top", "ret_basket"]:
        label = col.replace("ret_", "")
        vals_all = df[col].drop_nulls().sum()
        print(f"  {label}: 全量={vals_all:+.2%}")
        for exclude_year in [2015, 2016, 2024, 2025]:
            if exclude_year in years:
                vals = df.filter(pl.col("year") != exclude_year)[col].drop_nulls().sum()
                print(f"    去{exclude_year}: {vals:+.2%}")

    # 分年统计
    print("\n── 分年统计 ──")
    print(f"  {'Year':>6s} | {'N':>4s} | {'全篮子':>8s} | {'纯响应':>8s} | {'未透支':>8s} | {'随机':>8s}")
    for year in years:
        ydf = df.filter(pl.col("year") == year)
        n = ydf.height
        b = ydf["ret_basket"].drop_nulls().mean()
        r = ydf["ret_response_top"].drop_nulls().mean()
        d = ydf["ret_diffusion_top"].drop_nulls().mean()
        rr = ydf["ret_random_mean"].drop_nulls().mean()
        print(f"  {year:>6d} | {n:>4d} | "
              f"{b:+.3%}" if b is not None else "--", end="")
        print(f" | {r:+.3%}" if r is not None else " | --", end="")
        print(f" | {d:+.3%}" if d is not None else " | --", end="")
        print(f" | {rr:+.3%}" if rr is not None else " | --")

    return df


# ══════════════════════════════════════════════════════════════════════════
# 报告生成
# ══════════════════════════════════════════════════════════════════════════


def _clean(vals: pl.Series) -> pl.Series:
    """过滤 inf/nan/null。"""
    return vals.filter(~vals.is_infinite() & ~vals.is_nan()).drop_nulls()


def _compound(vals: pl.Series) -> float:
    """复合收益 ∏(1+r_i) - 1。"""
    clean = _clean(vals)
    if clean.len() == 0:
        return 0.0
    return float((1 + clean).product() - 1)


def generate_report(
    results: dict[str, pl.DataFrame],
    output_path: Path,
) -> None:
    """生成汇总报告。"""
    lines = [
        "# 五大实验诊断报告",
        "",
        "诊断目标: alpha 来源是事件识别、执行方式、选股方式、还是样本结构？",
        "",
    ]

    # 实验 1
    if "exp1" in results and results["exp1"].height > 0:
        df = results["exp1"]
        lines.append("## 实验 1: 入场时机收益拆解")
        lines.append("")
        lines.append("| 入场方式 | 含义 | 平均收益 | 中位数 | 胜率 | 累计 |")
        lines.append("|---------|------|---------|--------|------|------|")
        for col, desc in [
            ("ret_overnight", "T0 收盘→T1 开盘 (隔夜 gap)"),
            ("ret_eod_to_t1", "T0 收盘→T1 收盘 (持有1天)"),
            ("ret_eod_to_t3", "T0 收盘→T3 收盘 (尾盘买持3天)"),
            ("ret_t1open_to_t3", "T1 开盘→T3 收盘 (当前策略)"),
            ("ret_t1vwap_to_t3", "T1 VWAP→T3 收盘"),
        ]:
            vals = _clean(df[col])
            if vals.len() > 0:
                lines.append(
                    f"| {col} | {desc} "
                    f"| {vals.mean():+.3%} | {vals.median():+.3%} "
                    f"| {vals.filter(vals > 0).len() / vals.len():.0%} "
                    f"| {_compound(df[col]):+.2%} |"
                )
        lines.append("")

    # 实验 2
    if "exp2" in results and results["exp2"].height > 0:
        df = results["exp2"]
        lines.append("## 实验 2: 候选池扩展")
        lines.append("")
        lines.append("| 候选池 | 平均收益 | 胜率 | 累计 |")
        lines.append("|--------|---------|------|------|")
        for pool in ["HS300", "CSI500", "HS300+500"]:
            col = f"ret_{pool}"
            if col in df.columns:
                vals = _clean(df[col])
                if vals.len() > 0:
                    lines.append(
                        f"| {pool} | {vals.mean():+.3%} "
                        f"| {vals.filter(vals > 0).len() / vals.len():.0%} "
                        f"| {_compound(df[col]):+.2%} |"
                    )
        lines.append("")

    # 实验 3
    if "exp3" in results and results["exp3"].height > 0:
        df = results["exp3"]
        lines.append("## 实验 3: 打分器对比")
        lines.append("")
        lines.append("| 打分器 | 平均收益 | 胜率 | 累计 |")
        lines.append("|--------|---------|------|------|")
        for col, label in [
            ("ret_scorer_v2", "原始(追强度)"),
            ("ret_diffusion", "未透支扩散"),
            ("ret_basket", "全篮子等权"),
        ]:
            vals = _clean(df[col])
            if vals.len() > 0:
                lines.append(
                    f"| {label} | {vals.mean():+.3%} "
                    f"| {vals.filter(vals > 0).len() / vals.len():.0%} "
                    f"| {_compound(df[col]):+.2%} |"
                )
        lines.append("")

    # 实验 4
    if "exp4" in results and results["exp4"].height > 0:
        df = results["exp4"]
        lines.append("## 实验 4: 退出规则对比 (T0 收盘入场)")
        lines.append("")
        lines.append("| 退出规则 | 平均收益 | 胜率 | 累计 |")
        lines.append("|---------|---------|------|------|")
        for rule in ["fixed_2", "fixed_3", "fixed_3_vwap_stop"]:
            col = f"ret_{rule}"
            if col in df.columns:
                vals = _clean(df[col])
                if vals.len() > 0:
                    lines.append(
                        f"| {rule} | {vals.mean():+.3%} "
                        f"| {vals.filter(vals > 0).len() / vals.len():.0%} "
                        f"| {_compound(df[col]):+.2%} |"
                    )
        lines.append("")

    # 实验 5
    if "exp5" in results and results["exp5"].height > 0:
        df = results["exp5"]
        lines.append("## 实验 5: 基准对比 + 去极值")
        lines.append("")
        lines.append("| 基准 | 平均收益 | 胜率 | 累计 |")
        lines.append("|------|---------|------|------|")
        for col, label in [
            ("ret_random_mean", "随机TopN"),
            ("ret_basket", "全篮子等权"),
            ("ret_response_top", "纯响应排序"),
            ("ret_diffusion_top", "未透支扩散"),
        ]:
            vals = _clean(df[col])
            if vals.len() > 0:
                lines.append(
                    f"| {label} | {vals.mean():+.3%} "
                    f"| {vals.filter(vals > 0).len() / vals.len():.0%} "
                    f"| {_compound(df[col]):+.2%} |"
                )
        lines.append("")

        # 分年
        lines.append("### 分年统计")
        lines.append("")
        lines.append("| Year | N | 全篮子 | 纯响应 | 未透支 | 随机 |")
        lines.append("|------|---|--------|--------|--------|------|")
        years = sorted(df["year"].unique().to_list())
        for year in years:
            ydf = df.filter(pl.col("year") == year)
            n = ydf.height
            cols_vals = []
            for c in ["ret_basket", "ret_response_top", "ret_diffusion_top", "ret_random_mean"]:
                v = _clean(ydf[c])
                cols_vals.append(f"{v.mean():+.3%}" if v.len() > 0 else "--")
            lines.append(f"| {year} | {n} | " + " | ".join(cols_vals) + " |")
        lines.append("")

        # 去极值
        lines.append("### 去极值累计收益")
        lines.append("")
        for exclude_year in [2015, 2016]:
            sub = df.filter(pl.col("year") != exclude_year)
            lines.append(f"**去{exclude_year}后:**")
            for col, label in [
                ("ret_diffusion_top", "未透支扩散"),
                ("ret_response_top", "纯响应排序"),
                ("ret_basket", "全篮子等权"),
            ]:
                vals = _clean(sub[col])
                if vals.len() > 0:
                    lines.append(f"- {label}: {_compound(sub[col]):+.2%}")
            lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n报告已写入: {output_path}")


# ══════════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════════


def main() -> None:
    print("=" * 70)
    print("  五大实验: 诊断 alpha 来源与执行错位")
    print("=" * 70)

    random.seed(42)

    # Step 0: 检测信号日 (复用现有逻辑)
    print(f"\n[Step 0] 检测信号日 (prob > {SIGNAL_THRESHOLD:.0%}) ...")
    signals = detect_signal_dates()
    print(f"  → 共 {len(signals)} 个信号日")
    if not signals:
        print("No signals!")
        return

    ch = get_ch()

    results: dict[str, pl.DataFrame] = {}

    # ── 实验 1 ────────────────────────────────────────────────────────────
    results["exp1"] = experiment_1_entry_timing(ch, signals)

    # ── 实验 2 ────────────────────────────────────────────────────────────
    results["exp2"] = experiment_2_pool_expansion(ch, signals)

    # ── 实验 3 ────────────────────────────────────────────────────────────
    results["exp3"] = experiment_3_scorer_comparison(ch, signals, top_n=5)

    # ── 实验 4 ────────────────────────────────────────────────────────────
    results["exp4"] = experiment_4_exit_rules(ch, signals, top_n=5)

    # ── 实验 5 ────────────────────────────────────────────────────────────
    results["exp5"] = experiment_5_baselines(ch, signals, top_n=5, n_random_trials=200)

    # ── 保存结果 ──────────────────────────────────────────────────────────
    out_dir = ROOT / "data" / "five_experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in results.items():
        if df.height > 0:
            df.write_parquet(out_dir / f"{name}.parquet")
            print(f"  {name}: saved ({df.height} rows)")

    # ── 生成报告 ──────────────────────────────────────────────────────────
    report_path = ROOT / "doc" / "five_experiments_report.md"
    generate_report(results, report_path)

    print("\n" + "=" * 70)
    print("  全部实验完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()
