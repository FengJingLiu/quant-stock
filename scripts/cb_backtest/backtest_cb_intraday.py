#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.cb_backtest.engine import run_backtest


def _parse_date(v: str) -> date:
    if v.lower() == "today":
        return date.today()
    return datetime.strptime(v, "%Y-%m-%d").date()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Convertible bond intraday backtest (bondTrader-style)")
    ap.add_argument("--config", default="data/cb_trigger.json", help="trigger config json path")
    ap.add_argument("--cb-info", default="data/bondtick/可转债基础信息列表.csv", help="cb-stock map csv")
    ap.add_argument("--shares", default="/home/autumn/quant/bondTrader/data/shares.json", help="shares json path")
    ap.add_argument("--start", default="2014-01-01", help="start date")
    ap.add_argument("--end", default="today", help="end date")
    ap.add_argument("--initial-cash", type=float, default=200_000.0)
    ap.add_argument("--workers", type=int, default=0, help="day-parallel workers; 0=CPU cores")
    ap.add_argument("--out", default="", help="optional output json file")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    result = run_backtest(
        config_path=args.config,
        cb_info_csv=args.cb_info,
        shares_json=args.shares,
        start_date=_parse_date(args.start),
        end_date=_parse_date(args.end),
        initial_cash=args.initial_cash,
        workers=args.workers,
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
