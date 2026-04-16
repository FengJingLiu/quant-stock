from __future__ import annotations

import unittest

import pandas as pd

from src.bank_metal_balance_502525.backtest import filter_value_dividend_ma


class TestNonBankMarketCapFilter(unittest.TestCase):
    def test_filter_requires_circ_mv_above_200yi(self) -> None:
        df = pd.DataFrame(
            {
                "symbol": ["LOW", "HIGH"],
                "pe_ttm": [10.0, 10.0],
                "dividend_yield": [5.0, 5.0],
                "ma120_adj": [100.0, 100.0],
                "close_adj": [80.0, 80.0],
                "circ_mv_10k": [1_500_000.0, 2_500_000.0],
            }
        )

        out = filter_value_dividend_ma(df)

        self.assertEqual(out["symbol"].tolist(), ["HIGH"])


if __name__ == "__main__":
    unittest.main()
