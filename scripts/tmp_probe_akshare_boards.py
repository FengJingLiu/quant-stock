#!/usr/bin/env python3
from __future__ import annotations

import os

import akshare as ak
import akshare_proxy_patch

import _load_env  # noqa: F401


def main() -> None:
    akshare_proxy_patch.install_patch(os.environ["AKSHARE_PROXY_HOST"], os.environ["AKSHARE_PROXY_TOKEN"], retry=30)

    names = ak.stock_board_industry_name_em()
    print("[board_names] rows=", 0 if names is None else len(names))
    if names is not None and not names.empty:
        cols = [
            c
            for c in ["板块代码", "板块名称", "最新价", "涨跌幅"]
            if c in names.columns
        ]
        print(names[cols].head(10).to_string(index=False))
        hit = names[names["板块代码"].astype(str) == "BK0478"]
        print("[BK0478 in board list]", not hit.empty)
        if not hit.empty:
            print(
                hit[
                    [c for c in ["板块代码", "板块名称"] if c in hit.columns]
                ].to_string(index=False)
            )

    cons_by_code = ak.stock_board_industry_cons_em(symbol="BK0478")
    print(
        "[BK0478 cons by code] rows=", 0 if cons_by_code is None else len(cons_by_code)
    )
    if cons_by_code is not None and not cons_by_code.empty:
        print(cons_by_code.head(5).to_string(index=False))

    cons_by_name = ak.stock_board_industry_cons_em(symbol="有色金属")
    print(
        "[有色金属 cons by name] rows=",
        0 if cons_by_name is None else len(cons_by_name),
    )
    if cons_by_name is not None and not cons_by_name.empty:
        print(cons_by_name.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
