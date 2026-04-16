from __future__ import annotations

import unittest
from typing import cast

import pandas as pd

from src.bank_metal_balance_502525.backtest import (
    RebalanceSelection,
    build_daily_targets_with_take_profit,
)


class TestBankDividendRebalanceRule(unittest.TestCase):
    def test_bank_not_monthly_rebalanced_when_dividend_not_below_5(self) -> None:
        d0 = cast(pd.Timestamp, pd.Timestamp("2020-01-02"))
        d1 = cast(pd.Timestamp, pd.Timestamp("2020-02-03"))
        d2 = cast(pd.Timestamp, pd.Timestamp("2020-03-02"))

        trading_dates = [d0, d1, d2]
        rebalance_dates = [d0, d1, d2]

        selections = {
            d0: RebalanceSelection(
                rebalance_date=d0,
                signal_date=d0,
                bank_symbols=["A", "B"],
                metal_symbols=[],
                other_symbols=[],
            ),
            d1: RebalanceSelection(
                rebalance_date=d1,
                signal_date=d1,
                bank_symbols=["C", "D"],
                metal_symbols=[],
                other_symbols=[],
            ),
            d2: RebalanceSelection(
                rebalance_date=d2,
                signal_date=d2,
                bank_symbols=["C", "D"],
                metal_symbols=[],
                other_symbols=[],
            ),
        }

        _, daily_bank_symbols, bank_rebalance_days, _, _, _ = (
            build_daily_targets_with_take_profit(
                trading_dates=trading_dates,
                rebalance_dates=rebalance_dates,
                selections=selections,
                close_lookup={},
                bank_dividend_lookup={
                    (d1, "A"): 6.2,
                    (d1, "B"): 5.8,
                    (d2, "A"): 4.6,
                    (d2, "B"): 5.6,
                    (d2, "C"): 6.1,
                },
            )
        )

        self.assertEqual(daily_bank_symbols[d0], ["A", "B"])
        self.assertEqual(daily_bank_symbols[d1], ["A", "B"])
        self.assertEqual(daily_bank_symbols[d2], ["B", "C"])
        self.assertIn(d0, bank_rebalance_days)
        self.assertNotIn(d1, bank_rebalance_days)
        self.assertIn(d2, bank_rebalance_days)

    def test_bank_not_rotated_when_all_replacements_below_5(self) -> None:
        d0 = cast(pd.Timestamp, pd.Timestamp("2020-01-02"))
        d1 = cast(pd.Timestamp, pd.Timestamp("2020-02-03"))
        d2 = cast(pd.Timestamp, pd.Timestamp("2020-03-02"))

        trading_dates = [d0, d1, d2]
        rebalance_dates = [d0, d1, d2]

        selections = {
            d0: RebalanceSelection(
                rebalance_date=d0,
                signal_date=d0,
                bank_symbols=["A", "B"],
                metal_symbols=[],
                other_symbols=[],
            ),
            d1: RebalanceSelection(
                rebalance_date=d1,
                signal_date=d1,
                bank_symbols=["C", "D"],
                metal_symbols=[],
                other_symbols=[],
            ),
            d2: RebalanceSelection(
                rebalance_date=d2,
                signal_date=d2,
                bank_symbols=["C", "D"],
                metal_symbols=[],
                other_symbols=[],
            ),
        }

        _, daily_bank_symbols, bank_rebalance_days, _, _, _ = (
            build_daily_targets_with_take_profit(
                trading_dates=trading_dates,
                rebalance_dates=rebalance_dates,
                selections=selections,
                close_lookup={},
                bank_dividend_lookup={
                    (d1, "A"): 4.3,
                    (d1, "B"): 4.7,
                    (d1, "C"): 4.8,
                    (d1, "D"): 4.9,
                    (d2, "A"): 4.1,
                    (d2, "B"): 4.6,
                    (d2, "C"): 5.4,
                    (d2, "D"): 5.7,
                },
            )
        )

        self.assertEqual(daily_bank_symbols[d0], ["A", "B"])
        self.assertEqual(daily_bank_symbols[d1], ["A", "B"])
        self.assertEqual(daily_bank_symbols[d2], ["C", "D"])
        self.assertIn(d0, bank_rebalance_days)
        self.assertNotIn(d1, bank_rebalance_days)
        self.assertIn(d2, bank_rebalance_days)


if __name__ == "__main__":
    unittest.main()
