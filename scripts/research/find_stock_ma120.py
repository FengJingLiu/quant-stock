#!/usr/bin/env python3
"""
筛选 A 股 PE < 20、股息率 > 5% 且股价处于 MA120 下方的公司。

筛选条件:
- PE(TTM) < 20
- 股息率(TTM) > 5%
- 股价 < MA120 (股价在120日均线下方)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

import akshare as ak
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data_clients import ensure_akshare_proxy_patch

DEFAULT_OUTPUT = Path("data/stock_filter_ma120.csv")
CACHE_DIR = Path("data/cache")
TOKEN = os.environ["AKSHARE_PROXY_TOKEN"]
PROXY_HOST = os.environ["AKSHARE_PROXY_HOST"]


def retry(func: Callable[[], Any], tries: int, base_sleep: float) -> Any:
    last_error: Exception | None = None
    for i in range(tries):
        try:
            return func()
        except Exception as exc:
            last_error = exc
            wait_s = base_sleep * (i + 1)
            print(f"[WARN] retry {i + 1}/{tries}: {type(exc).__name__}: {exc}")
            time.sleep(wait_s)
    if last_error:
        raise last_error
    raise RuntimeError("未知重试错误")


def install_proxy_patch() -> None:
    ensure_akshare_proxy_patch(proxy_host=PROXY_HOST, token=TOKEN, retry=30)
    print(f"[INFO] 已启用 akshare-proxy-patch: host={PROXY_HOST}")


# ============ 缓存机制 ============

def get_cache_path(cache_type: str, code: str = None) -> Path:
    """获取缓存文件路径"""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if code:
        return CACHE_DIR / f"{cache_type}_{code}.json"
    return CACHE_DIR / f"{cache_type}.json"


def load_from_cache(cache_type: str, code: str = None, max_age_hours: int = 24) -> dict | None:
    """从缓存加载数据"""
    cache_file = get_cache_path(cache_type, code)
    if not cache_file.exists():
        return None
    
    try:
        mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
        age = datetime.now() - mtime
        if age.total_seconds() > max_age_hours * 3600:
            return None
        
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def save_to_cache(cache_type: str, data: dict, code: str = None) -> None:
    """保存数据到缓存"""
    cache_file = get_cache_path(cache_type, code)
    try:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    except Exception as e:
        print(f"[WARN] 缓存写入失败: {e}")


# ============ 数据获取 ============

def fetch_all_stocks_realtime(tries: int, sleep: float) -> pd.DataFrame:
    """获取A股所有股票实时数据（包含PE和PB）"""
    # 先尝试从缓存加载
    cache_data = load_from_cache("all_stocks_realtime", max_age_hours=1)
    if cache_data is not None:
        print("[INFO] 从缓存加载股票实时数据")
        df = pd.DataFrame(cache_data)
        df["代码"] = df["代码"].astype(str).str.zfill(6)
        for col in ["最新价", "pe_ttm", "pb"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        print(f"[INFO] 缓存中共有 {len(df)} 只股票")
        return df[["代码", "名称", "最新价", "pe_ttm", "pb"]]
    
    print("[INFO] 获取A股实时行情...")
    df = retry(lambda: ak.stock_zh_a_spot_em(), tries=tries, base_sleep=sleep)
    if df is None or df.empty:
        raise RuntimeError("未获取到股票实时数据")
    
    df = df.copy()
    df["代码"] = df["代码"].astype(str).str.zfill(6)
    
    for col in ["最新价", "市盈率-动态", "市净率"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    df = df.rename(columns={
        "最新价": "最新价",
        "市盈率-动态": "pe_ttm",
        "市净率": "pb"
    })
    
    # 保存到缓存
    cache_data = df[["代码", "名称", "最新价", "pe_ttm", "pb"]].to_dict(orient="records")
    save_to_cache("all_stocks_realtime", cache_data)
    
    print(f"[INFO] 获取到 {len(df)} 只股票，已缓存")
    return df[["代码", "名称", "最新价", "pe_ttm", "pb"]]


def fetch_dividend_ttm(code: str, tries: int, sleep: float) -> dict[str, Any]:
    """获取个股TTM股息率（带缓存）"""
    # 先尝试从缓存加载
    cache_data = load_from_cache("dividend", code, max_age_hours=24*7)  # 分红数据缓存7天
    if cache_data is not None:
        return cache_data
    
    rec = {"每股分红_TTM": None, "error": ""}
    
    try:
        df = retry(
            lambda: ak.stock_dividend_cninfo(symbol=code),
            tries=tries,
            base_sleep=sleep
        )
        if df is None or df.empty:
            rec["error"] = "no_dividend"
            save_to_cache("dividend", rec, code)
            return rec
        
        if "实施方案公告日期" not in df.columns or "派息比例" not in df.columns:
            rec["error"] = "missing_cols"
            save_to_cache("dividend", rec, code)
            return rec
        
        work = df.copy()
        work["实施方案公告日期"] = pd.to_datetime(work["实施方案公告日期"], errors="coerce")
        work["派息比例"] = pd.to_numeric(work["派息比例"], errors="coerce")
        work = work.dropna(subset=["实施方案公告日期", "派息比例"])
        
        if work.empty:
            rec["error"] = "no_valid_dividend"
            save_to_cache("dividend", rec, code)
            return rec
        
        # 最近一年
        one_year_ago = datetime.now() - timedelta(days=365)
        work = work[work["实施方案公告日期"] >= one_year_ago]
        
        if work.empty:
            rec["error"] = "no_dividend_1y"
            save_to_cache("dividend", rec, code)
            return rec
        
        # 每股分红 = 派息比例 / 10
        total_cash_per_10 = work["派息比例"].sum()
        if pd.notna(total_cash_per_10):
            rec["每股分红_TTM"] = float(total_cash_per_10) / 10.0
        
    except Exception as exc:
        rec["error"] = f"{type(exc).__name__}: {exc}"
    
    save_to_cache("dividend", rec, code)
    return rec


def fetch_price_and_ma(code: str, tries: int, sleep: float) -> dict[str, Any]:
    """获取个股价格和MA120（带缓存）"""
    # 先尝试从缓存加载
    cache_data = load_from_cache("price_ma", code, max_age_hours=24)  # 价格数据缓存1天
    if cache_data is not None:
        return cache_data
    
    rec = {"最新价": None, "ma120": None, "ma120_status": None, "error": ""}
    
    try:
        # 获取最近180天数据，确保有足够计算MA120
        df = retry(
            lambda: ak.stock_zh_a_hist(symbol=code, period="daily", adjust="qfq", start_date="20250101"),
            tries=tries,
            base_sleep=sleep
        )
        
        if df is None or df.empty:
            rec["error"] = "no_price_data"
            save_to_cache("price_ma", rec, code)
            return rec
        
        df["日期"] = pd.to_datetime(df["日期"])
        df = df.sort_values("日期")
        
        if len(df) < 120:
            rec["error"] = "insufficient_data"
            save_to_cache("price_ma", rec, code)
            return rec
        
        # 计算MA120
        df["ma120"] = df["收盘"].rolling(window=120).mean()
        
        latest = df.iloc[-1]
        price = float(latest["收盘"])
        ma120 = float(latest["ma120"])
        
        rec["最新价"] = price
        rec["ma120"] = ma120
        
        # 修改：当前价格在MA120下方
        if price < ma120:
            rec["ma120_status"] = "below"
        else:
            rec["ma120_status"] = "above"
        
    except Exception as exc:
        rec["error"] = f"{type(exc).__name__}: {exc}"
    
    save_to_cache("price_ma", rec, code)
    return rec


def calculate_dividend_yield(price: float, cash_per_share: float | None) -> float | None:
    """计算股息率"""
    if price and price > 0 and cash_per_share and cash_per_share > 0:
        return (cash_per_share / price) * 100.0
    return None


def build_filtered_list(tries: int, sleep: float, limit: int | None) -> pd.DataFrame:
    """构建筛选后的股票列表"""
    
    # 1. 获取所有股票实时数据（PE、PB）
    stocks = fetch_all_stocks_realtime(tries=tries, sleep=sleep)
    
    # 2. 预先筛选 PE < 20，减少后续请求
    stocks = stocks[stocks["pe_ttm"].notna() & (stocks["pe_ttm"] < 20) & (stocks["pe_ttm"] > 0)]
    print(f"[INFO] 0 < PE < 20 的股票: {len(stocks)} 只")
    
    if limit is not None:
        stocks = stocks.head(limit)
    
    rows: list[dict[str, Any]] = []
    
    for idx, (i, row) in enumerate(stocks.iterrows()):
        code = str(row["代码"])
        name = str(row.get("名称", ""))
        price = row["最新价"]
        pe = row["pe_ttm"]
        
        print(f"[{idx+1}/{len(stocks)}] {code} {name} (PE={pe:.2f})...", end=" ")
        
        # 获取分红数据（带缓存）
        dividend = fetch_dividend_ttm(code, tries=tries, sleep=sleep)
        
        # 计算股息率
        div_yield = calculate_dividend_yield(price, dividend.get("每股分红_TTM"))
        
        # 如果股息率 <= 5%，直接跳过（不需要获取MA120）
        if div_yield is None or div_yield <= 5:
            print(f"股息率={div_yield}%, 不符合")
            continue
        
        # 获取MA120（带缓存）
        price_ma = fetch_price_and_ma(code, tries=tries, sleep=sleep)
        
        rec: dict[str, Any] = {
            "代码": code,
            "名称": name,
            "最新价": price,
            "pe_ttm": pe,
            "pb": row.get("pb"),
            "ma120": price_ma.get("ma120"),
            "ma120_status": price_ma.get("ma120_status"),
            "每股分红_TTM": dividend.get("每股分红_TTM"),
            "股息率TTM": div_yield,
            "error": dividend.get("error", "") + ";" + price_ma.get("error", ""),
        }
        
        # 修改：筛选条件 - 当前价格在MA120下方
        if rec["ma120_status"] == "below":
            rows.append(rec)
            print(f"✓ 符合! 股息率={div_yield:.2f}%, MA120={rec['ma120']:.2f}")
        else:
            print(f"MA120={rec['ma120_status']}")
        
        time.sleep(0.3)
    
    out = pd.DataFrame(rows)
    
    if not out.empty:
        out = out.sort_values(by=["股息率TTM", "pe_ttm"], ascending=[False, True], na_position="last")
        out = out.reset_index(drop=True)
    
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="筛选 A 股 PE<20、股息率>5% 且 MA120 下方的公司")
    parser.add_argument("--limit", type=int, default=None, help="仅处理前 N 只（调试用）")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="输出 CSV 路径")
    parser.add_argument("--tries", type=int, default=3, help="每个接口调用重试次数")
    parser.add_argument("--sleep", type=float, default=1.0, help="重试基础等待秒数")
    parser.add_argument("--clear-cache", action="store_true", help="清除缓存后重新获取")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    if args.clear_cache:
        import shutil
        if CACHE_DIR.exists():
            shutil.rmtree(CACHE_DIR)
            print(f"[INFO] 已清除缓存: {CACHE_DIR}")
    
    install_proxy_patch()
    
    df = build_filtered_list(
        tries=args.tries,
        sleep=args.sleep,
        limit=args.limit
    )
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False, encoding="utf-8-sig")
    
    print(f"\n[OK] 已保存: {args.output}")
    print(f"[INFO] 符合条件股票数: {len(df)}")
    
    if not df.empty:
        show_cols = ["代码", "名称", "最新价", "pe_ttm", "股息率TTM", "ma120"]
        print(df[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()
