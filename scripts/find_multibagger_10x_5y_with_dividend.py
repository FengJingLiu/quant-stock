#!/usr/bin/env python3
"""筛选近 N 年内“买入后(计入分红送转)涨了 M 倍以上”的股票。

核心思路
- 使用 akshare 的 A 股日线 + 前复权(qfq)价格来近似“含分红送转”的总回报。
- “如果买入后”通常意味着：在过去 N 年区间内任选一个买入点。
  本脚本默认用区间内的“最低前复权收盘价”作为买入点（事后最优 entry_mode=min）。
  你也可以用 entry_mode=start 表示“恰好在 N 年前附近买入并持有到今天”。

输出
1) data/multibagger_10x_5y_list.csv
2) data/multibagger_10x_5y_summary_by_board_industry.csv

注意
- 全市场扫描请求量很大，可能耗时较久；建议先用 --limit 50 自检。
- 网络波动/限流常见：已内置重试，并支持 --resume 断点续跑（依赖 cache 文件）。
"""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

import akshare as ak
import akshare_proxy_patch
import pandas as pd

DEFAULT_PROXY_HOST = "***AKSHARE_HOST***"
DEFAULT_TOKEN = os.getenv("AKSHARE_PROXY_TOKEN", "***AKSHARE_TOKEN***")

DEFAULT_CACHE = Path("data/multibagger_10x_5y_cache.csv")
DEFAULT_OUT_LIST = Path("data/multibagger_10x_5y_list.csv")
DEFAULT_OUT_SUMMARY = Path("data/multibagger_10x_5y_summary_by_board_industry.csv")


def retry(func: Callable[[], Any], tries: int, base_sleep: float) -> Any:
    last_error: Exception | None = None
    for i in range(tries):
        try:
            return func()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            wait_s = base_sleep * (i + 1)
            print(f"[WARN] retry {i + 1}/{tries}: {type(exc).__name__}: {exc}")
            time.sleep(wait_s)
    if last_error:
        raise last_error
    raise RuntimeError("未知重试错误")


def install_proxy_patch(proxy_host: str, token: str, proxy_retry: int) -> None:
    if not token:
        raise ValueError("未提供 token，请通过 --token 或环境变量 AKSHARE_PROXY_TOKEN 设置")
    akshare_proxy_patch.install_patch(proxy_host, token, retry=proxy_retry)
    print(f"[INFO] 已启用 akshare-proxy-patch: host={proxy_host}, retry={proxy_retry}")


def classify_board(code: str) -> str:
    c = str(code).strip()
    # A股常见板块/交易所粗分（按代码前缀）
    if c.startswith(("68",)):
        return "科创板"
    if c.startswith(("60", "601", "603", "605")):
        return "沪主板"
    if c.startswith(("30",)):
        return "创业板"
    if c.startswith(("00",)):
        return "深主板"
    if c.startswith(("83", "87", "88", "43", "92")):
        return "北交/新三板"
    return "其他"


def fetch_universe(tries: int, sleep: float) -> pd.DataFrame:
    # 使用实时行情接口拿“当前存续”的全市场列表（含北交）
    df = retry(lambda: ak.stock_zh_a_spot_em(), tries=tries, base_sleep=sleep)
    if df is None or df.empty:
        raise RuntimeError("未获取到 A 股列表(stock_zh_a_spot_em)")

    need = [c for c in ["代码", "名称"] if c in df.columns]
    out = df[need].copy()
    out["代码"] = out["代码"].astype(str).str.zfill(6)
    out["名称"] = out["名称"].astype(str)
    return out.drop_duplicates(subset=["代码"]).reset_index(drop=True)


def _safe_float(x: Any) -> float | None:
    try:
        v = float(x)
        if pd.isna(v):
            return None
        return v
    except Exception:  # noqa: BLE001
        return None


def calc_multiple_for_code(
    code: str,
    name: str,
    start_date: str,
    end_date: str,
    entry_mode: str,
    tries: int,
    sleep: float,
) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "代码": str(code).zfill(6),
        "名称": str(name),
        "板块": classify_board(code),
        "entry_mode": entry_mode,
        "起始交易日": None,
        "起始收盘": None,
        "结束交易日": None,
        "结束收盘": None,
        "收益倍数": None,
        "错误": "",
    }

    try:
        df = retry(
            lambda: ak.stock_zh_a_hist(
                symbol=str(code).zfill(6),
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust="qfq",
            ),
            tries=tries,
            base_sleep=sleep,
        )

        if df is None or df.empty:
            rec["错误"] = "hist_empty"
            return rec

        # 统一字段
        if "日期" not in df.columns or "收盘" not in df.columns:
            rec["错误"] = f"hist_missing_cols:{sorted({'日期','收盘'} - set(df.columns))}"
            return rec

        work = df.copy()
        work["日期"] = pd.to_datetime(work["日期"], errors="coerce")
        work["收盘"] = pd.to_numeric(work["收盘"], errors="coerce")
        work = work.dropna(subset=["日期", "收盘"]).sort_values("日期")
        if work.empty:
            rec["错误"] = "hist_empty_after_clean"
            return rec

        end_row = work.iloc[-1]
        end_close = _safe_float(end_row["收盘"])
        end_dt = pd.Timestamp(end_row["日期"]).strftime("%Y-%m-%d")

        if entry_mode == "start":
            entry_row = work.iloc[0]
        elif entry_mode == "min":
            entry_row = work.loc[work["收盘"].idxmin()]
        else:
            raise ValueError(f"未知 entry_mode: {entry_mode}")

        entry_close = _safe_float(entry_row["收盘"])
        entry_dt = pd.Timestamp(entry_row["日期"]).strftime("%Y-%m-%d")

        rec["起始交易日"] = entry_dt
        rec["起始收盘"] = entry_close
        rec["结束交易日"] = end_dt
        rec["结束收盘"] = end_close

        if entry_close is None or entry_close <= 0 or end_close is None or end_close <= 0:
            rec["错误"] = "bad_price"
            return rec

        rec["收益倍数"] = float(end_close / entry_close)

    except Exception as exc:  # noqa: BLE001
        rec["错误"] = f"{type(exc).__name__}: {exc}"

    finally:
        if sleep and sleep > 0:
            time.sleep(sleep)

    return rec


def fetch_industry(code: str, tries: int, sleep: float) -> str:
    try:
        df = retry(lambda: ak.stock_individual_info_em(symbol=str(code).zfill(6)), tries=tries, base_sleep=sleep)
        if df is None or df.empty or not {"item", "value"}.issubset(df.columns):
            return ""
        m = df.loc[df["item"].astype(str).str.strip() == "行业", "value"]
        if m.empty:
            return ""
        v = str(m.iloc[0]).strip()
        return v
    except Exception:
        return ""


def append_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    write_header = not path.exists()
    df.to_csv(path, mode="a", header=write_header, index=False, encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="近N年(含分红送转)涨M倍股筛选（akshare + qfq）")

    p.add_argument("--token", type=str, default=DEFAULT_TOKEN, help="akshare-proxy-patch token")
    p.add_argument("--proxy-host", type=str, default=DEFAULT_PROXY_HOST, help="akshare-proxy-patch host")
    p.add_argument("--proxy-retry", type=int, default=3, help="proxy patch retry")

    p.add_argument("--years", type=int, default=5, help="回看年数")
    p.add_argument("--multiple", type=float, default=10.0, help="涨幅倍数阈值")
    p.add_argument(
        "--entry-mode",
        type=str,
        choices=["min", "start"],
        default="min",
        help="买入点选择：min=区间最低点(事后最优)；start=区间起点",
    )

    p.add_argument("--workers", type=int, default=8, help="并发线程数")
    p.add_argument("--sleep", type=float, default=0.2, help="每次请求后额外 sleep 秒")
    p.add_argument("--tries", type=int, default=3, help="单请求重试次数")

    p.add_argument("--limit", type=int, default=None, help="只跑前 N 只股票用于自检")
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True, help="断点续跑")

    p.add_argument("--cache", type=Path, default=DEFAULT_CACHE, help="扫描 cache 文件")
    p.add_argument("--out-list", type=Path, default=DEFAULT_OUT_LIST, help="命中明细输出")
    p.add_argument("--out-summary", type=Path, default=DEFAULT_OUT_SUMMARY, help="汇总输出")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    install_proxy_patch(args.proxy_host, args.token, args.proxy_retry)

    as_of = pd.Timestamp.today().normalize()
    start = as_of - pd.DateOffset(years=int(args.years))
    start_date = start.strftime("%Y%m%d")
    end_date = as_of.strftime("%Y%m%d")

    print(f"[INFO] window: {start_date} -> {end_date}, entry_mode={args.entry_mode}, multiple>={args.multiple}")

    universe = fetch_universe(tries=args.tries, sleep=args.sleep)
    if args.limit is not None:
        universe = universe.head(int(args.limit)).reset_index(drop=True)

    processed: set[str] = set()
    if args.resume and args.cache.exists():
        try:
            old = pd.read_csv(args.cache, dtype={"代码": str})
            if "代码" in old.columns:
                processed = set(old["代码"].astype(str).str.zfill(6).tolist())
            print(f"[INFO] resume enabled: cache_rows={len(old)}, processed={len(processed)}")
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] failed to read cache for resume: {exc}")

    todo = universe.loc[~universe["代码"].isin(processed)].reset_index(drop=True)
    print(f"[INFO] universe={len(universe)}, todo={len(todo)}, workers={args.workers}")

    batch: list[dict[str, Any]] = []
    flush_every = 50

    with ThreadPoolExecutor(max_workers=int(args.workers)) as ex:
        futs = []
        for _, row in todo.iterrows():
            code = str(row["代码"]).zfill(6)
            name = str(row["名称"])
            futs.append(
                ex.submit(
                    calc_multiple_for_code,
                    code,
                    name,
                    start_date,
                    end_date,
                    args.entry_mode,
                    args.tries,
                    args.sleep,
                )
            )

        done_n = 0
        for fut in as_completed(futs):
            rec = fut.result()
            batch.append(rec)
            done_n += 1

            if len(batch) >= flush_every:
                append_csv(args.cache, batch)
                batch.clear()

            if done_n % 100 == 0 or done_n == len(futs):
                print(f"[INFO] progress {done_n}/{len(futs)}")

    if batch:
        append_csv(args.cache, batch)
        batch.clear()

    # 读取 cache 做筛选
    df = pd.read_csv(args.cache, dtype={"代码": str})
    if df.empty:
        print("[WARN] cache empty, nothing to do")
        return

    df["代码"] = df["代码"].astype(str).str.zfill(6)
    df["收益倍数"] = pd.to_numeric(df.get("收益倍数"), errors="coerce")
    df["错误"] = df.get("错误", "").astype(str)

    hits = df.loc[(df["错误"] == "") & (df["收益倍数"].notna()) & (df["收益倍数"] >= float(args.multiple))].copy()
    print(f"[INFO] hits: {len(hits)}")

    # 只对命中股票补行业信息
    if not hits.empty:
        industries = []
        for i, r in hits.iterrows():
            code = str(r["代码"]).zfill(6)
            ind = fetch_industry(code=code, tries=args.tries, sleep=args.sleep)
            industries.append(ind)
            if (len(industries)) % 20 == 0 or (len(industries)) == len(hits):
                print(f"[INFO] industry progress {len(industries)}/{len(hits)}")
        hits["行业"] = industries
    else:
        hits["行业"] = ""

    # 排序输出
    hits = hits.sort_values(by=["收益倍数"], ascending=False, na_position="last").reset_index(drop=True)

    out_cols = [
        "代码",
        "名称",
        "板块",
        "行业",
        "起始交易日",
        "起始收盘",
        "结束交易日",
        "结束收盘",
        "收益倍数",
        "entry_mode",
        "错误",
    ]
    for c in out_cols:
        if c not in hits.columns:
            hits[c] = pd.NA

    args.out_list.parent.mkdir(parents=True, exist_ok=True)
    hits[out_cols].to_csv(args.out_list, index=False, encoding="utf-8-sig")
    print(f"[INFO] wrote: {args.out_list}")

    # 汇总：板块 x 行业
    if hits.empty:
        summary = pd.DataFrame(columns=["板块", "行业", "数量", "样本列表"])
    else:
        def _sample_list(g: pd.DataFrame, k: int = 20) -> str:
            show = g.sort_values("收益倍数", ascending=False).head(k)
            return ";".join(
                f"{row['代码']}{row['名称']}({float(row['收益倍数']):.1f}x)" for _, row in show.iterrows()
            )

        grp = hits.groupby(["板块", "行业"], dropna=False)
        summary = grp.apply(lambda g: pd.Series({"数量": len(g), "样本列表": _sample_list(g)})).reset_index()
        summary = summary.sort_values(by=["数量"], ascending=False).reset_index(drop=True)

    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out_summary, index=False, encoding="utf-8-sig")
    print(f"[INFO] wrote: {args.out_summary}")

    # 控制台展示 Top 30
    if not hits.empty:
        show = hits.head(30)
        print("\n[TOP] multibaggers")
        for _, r in show.iterrows():
            print(
                f"{r['代码']} {r['名称']} | {r['板块']} | {r.get('行业','')} | {float(r['收益倍数']):.2f}x | entry={r['起始交易日']}->{r['结束交易日']}"
            )


if __name__ == "__main__":
    main()
