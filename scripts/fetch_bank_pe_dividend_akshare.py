#!/usr/bin/env python3
"""使用 akshare + akshare-proxy-patch 获取 A 股银行股股息与 PE 列表。"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any, Callable

import akshare as ak
import akshare_proxy_patch
import pandas as pd

DEFAULT_OUTPUT = Path("data/bank_pe_dividend_list_akshare_proxy.csv")


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


def fetch_bank_constituents(tries: int, sleep: float) -> pd.DataFrame:
    df = retry(lambda: ak.stock_board_industry_cons_em(symbol="银行"), tries=tries, base_sleep=sleep)
    if df is None or df.empty:
        raise RuntimeError("未获取到银行板块成分")

    use_cols = [c for c in ["代码", "名称", "最新价", "市盈率-动态", "市净率"] if c in df.columns]
    out = df[use_cols].copy()
    out["代码"] = out["代码"].astype(str).str.zfill(6)

    for c in ["最新价", "市盈率-动态", "市净率"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    return out


def fetch_dividend_info(code: str, tries: int, sleep: float, as_of_date: pd.Timestamp) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "公告日": None,
        "分红类型": None,
        "分红说明": None,
        "报告时间": None,
        "每股分红_最近": None,
        "每股分红_TTM": None,
        "错误": "",
    }

    try:
        df = retry(lambda: ak.stock_dividend_cninfo(symbol=code), tries=tries, base_sleep=sleep)
        if df is None or df.empty:
            rec["错误"] = "dividend_empty"
            return rec

        required_cols = {"实施方案公告日期", "派息比例"}
        if not required_cols.issubset(df.columns):
            rec["错误"] = f"dividend_missing_cols: {sorted(required_cols - set(df.columns))}"
            return rec

        work = df.copy()
        work["实施方案公告日期"] = pd.to_datetime(work["实施方案公告日期"], errors="coerce")
        work["派息比例"] = pd.to_numeric(work["派息比例"], errors="coerce")
        work = work.dropna(subset=["实施方案公告日期", "派息比例"]).sort_values("实施方案公告日期")
        if work.empty:
            rec["错误"] = "dividend_empty_after_clean"
            return rec

        last = work.iloc[-1]
        rec["公告日"] = (
            pd.Timestamp(last["实施方案公告日期"]).strftime("%Y-%m-%d")
            if pd.notna(last["实施方案公告日期"])
            else None
        )
        rec["分红类型"] = str(last.get("分红类型", ""))
        rec["分红说明"] = str(last.get("实施方案分红说明", ""))
        rec["报告时间"] = str(last.get("报告时间", ""))

        # 接口字段“派息比例”是每 10 股派现金额，折算为每股现金分红
        rec["每股分红_最近"] = float(last["派息比例"]) / 10.0

        start_date = as_of_date - pd.Timedelta(days=365)
        ttm_cash_per_10 = work.loc[work["实施方案公告日期"] >= start_date, "派息比例"].sum(min_count=1)
        if pd.notna(ttm_cash_per_10):
            rec["每股分红_TTM"] = float(ttm_cash_per_10) / 10.0

    except Exception as exc:  # noqa: BLE001
        rec["错误"] = f"{type(exc).__name__}: {exc}"

    return rec


def build_bank_list(tries: int, sleep: float, limit: int | None) -> pd.DataFrame:
    banks = fetch_bank_constituents(tries=tries, sleep=sleep)
    if limit is not None:
        banks = banks.head(limit)

    print(f"[INFO] 银行成分数: {len(banks)}")
    as_of_date = pd.Timestamp.today().normalize()

    rows: list[dict[str, Any]] = []
    for i, row in banks.iterrows():
        code = str(row["代码"]).zfill(6)
        name = str(row.get("名称", ""))

        div = fetch_dividend_info(code=code, tries=tries, sleep=sleep, as_of_date=as_of_date)

        price = pd.to_numeric(row.get("最新价"), errors="coerce")
        pe_ttm = pd.to_numeric(row.get("市盈率-动态"), errors="coerce")
        pb = pd.to_numeric(row.get("市净率"), errors="coerce")

        rec: dict[str, Any] = {
            "代码": code,
            "名称": name,
            "最新价": float(price) if pd.notna(price) else None,
            "pe_ttm": float(pe_ttm) if pd.notna(pe_ttm) else None,
            "pb": float(pb) if pd.notna(pb) else None,
            "公告日": div["公告日"],
            "分红类型": div["分红类型"],
            "报告时间": div["报告时间"],
            "分红说明": div["分红说明"],
            "每股分红_最近": div["每股分红_最近"],
            "每股分红_TTM": div["每股分红_TTM"],
            "股息率_最近": None,
            "股息率TTM": None,
            "错误": div["错误"],
        }

        if pd.notna(price) and price > 0:
            cash_recent = pd.to_numeric(div["每股分红_最近"], errors="coerce")
            cash_ttm = pd.to_numeric(div["每股分红_TTM"], errors="coerce")
            if pd.notna(cash_recent) and cash_recent >= 0:
                rec["股息率_最近"] = float(cash_recent / price * 100.0)
            if pd.notna(cash_ttm) and cash_ttm >= 0:
                rec["股息率TTM"] = float(cash_ttm / price * 100.0)

        rows.append(rec)
        if (i + 1) % 5 == 0 or (i + 1) == len(banks):
            print(f"[INFO] 进度 {i + 1}/{len(banks)}")

    out = pd.DataFrame(rows)

    for c in [
        "最新价",
        "pe_ttm",
        "pb",
        "每股分红_最近",
        "每股分红_TTM",
        "股息率_最近",
        "股息率TTM",
    ]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    for c in ["股息率TTM", "pe_ttm"]:
        if c not in out.columns:
            out[c] = pd.NA

    out = out.sort_values(by=["股息率TTM", "pe_ttm"], ascending=[False, True], na_position="last").reset_index(
        drop=True
    )
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用 akshare-proxy-patch 获取 A 股银行股股息和 PE 列表")
    parser.add_argument(
        "--token",
        type=str,
        default=os.getenv("AKSHARE_PROXY_TOKEN", "***AKSHARE_TOKEN***"),
        help="akshare-proxy-patch token（默认: ***AKSHARE_TOKEN***；环境变量 AKSHARE_PROXY_TOKEN 可覆盖）",
    )
    parser.add_argument(
        "--proxy-host",
        type=str,
        default="***AKSHARE_HOST***",
        help="akshare-proxy-patch 服务地址",
    )
    parser.add_argument(
        "--proxy-retry",
        type=int,
        default=30,
        help="akshare-proxy-patch 内部重试次数",
    )
    parser.add_argument(
        "--tries",
        type=int,
        default=4,
        help="脚本层每个接口调用重试次数",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.8,
        help="脚本层重试基础等待秒数",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="仅处理前 N 只（调试用）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="输出 CSV 路径",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    install_proxy_patch(proxy_host=args.proxy_host, token=args.token, proxy_retry=args.proxy_retry)

    df = build_bank_list(tries=args.tries, sleep=args.sleep, limit=args.limit)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False, encoding="utf-8-sig")

    print(f"\n[OK] 已保存: {args.output}")

    show_cols = [
        c
        for c in [
            "代码",
            "名称",
            "最新价",
            "pe_ttm",
            "pb",
            "公告日",
            "分红说明",
            "每股分红_TTM",
            "股息率TTM",
            "错误",
        ]
        if c in df.columns
    ]
    print(df[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()
