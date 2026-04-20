from __future__ import annotations

import os
import sys
from pathlib import Path

import akshare as ak
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data_clients import ensure_akshare_proxy_patch


def normalize_symbol_code(raw: object) -> str:
    digits = "".join(ch for ch in str(raw) if ch.isdigit())
    if len(digits) != 6:
        return ""
    if digits.startswith("6"):
        return f"{digits}.SH"
    if digits.startswith(("4", "8")):
        return f"{digits}.BJ"
    return f"{digits}.SZ"


def main() -> None:
    ensure_akshare_proxy_patch(
        proxy_host=os.environ["AKSHARE_PROXY_HOST"],
        token=os.environ["AKSHARE_PROXY_TOKEN"],
        retry=30,
    )

    picks = pd.read_csv(
        Path("data/backtest_bank_metal_balance_502525/monthly_picks.csv")
    )
    picks["symbol"] = picks["symbol"].astype(str)

    metal_picks = set(picks.loc[picks["sleeve"] == "metal", "symbol"])
    other_picks = set(picks.loc[picks["sleeve"] == "other", "symbol"])

    metal_df = ak.stock_board_industry_cons_em(symbol="BK0478")
    metal_board_symbols = {
        s for s in (normalize_symbol_code(x) for x in metal_df["代码"].tolist()) if s
    }

    boards = ak.stock_board_industry_name_em()
    board_names = [str(x) for x in boards["板块名称"].tolist()]
    selected_other_boards = [
        name
        for name in board_names
        if any(keyword in name for keyword in ("科技", "消费", "化工"))
    ]

    other_board_symbols: set[str] = set()
    for board_name in selected_other_boards:
        cons = ak.stock_board_industry_cons_em(symbol=board_name)
        other_board_symbols.update(
            {s for s in (normalize_symbol_code(x) for x in cons["代码"].tolist()) if s}
        )
    other_board_symbols.difference_update(metal_board_symbols)

    print("metal_not_in_BK0478", len(metal_picks - metal_board_symbols))
    print("other_not_in_board_set", len(other_picks - other_board_symbols))


if __name__ == "__main__":
    main()
