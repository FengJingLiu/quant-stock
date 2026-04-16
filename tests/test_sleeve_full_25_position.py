from __future__ import annotations

import unittest
from typing import cast

import pandas as pd

from src.bank_metal_balance_502525.backtest import (
    RebalanceSelection,
    TRIGGER_STOP_LOSS,
    TRIGGER_TAKE_PROFIT,
    build_daily_targets_with_take_profit,
)


class TestSleeveFull25Position(unittest.TestCase):
    def test_metal_and_other_each_use_full_25pct(self) -> None:
        d0 = cast(pd.Timestamp, pd.Timestamp("2020-01-02"))

        selections = {
            d0: RebalanceSelection(
                rebalance_date=d0,
                signal_date=d0,
                bank_symbols=[],
                metal_symbols=["M1", "M2"],
                other_symbols=["O1", "O2"],
            )
        }

        daily_targets, _, _, _, _, _ = build_daily_targets_with_take_profit(
            trading_dates=[d0],
            rebalance_dates=[d0],
            selections=selections,
            close_lookup={
                (d0, "M1"): 10.0,
                (d0, "M2"): 11.0,
                (d0, "O1"): 8.0,
                (d0, "O2"): 9.0,
            },
        )

        target = daily_targets[d0]
        self.assertEqual(target.get("M1", 0.0), 0.25)
        self.assertEqual(target.get("O1", 0.0), 0.25)
        self.assertNotIn("M2", target)
        self.assertNotIn("O2", target)

    def test_nonbank_no_monthly_rebalance_only_stoploss_takeprofit(self) -> None:
        d0 = cast(pd.Timestamp, pd.Timestamp("2020-01-02"))
        d1 = cast(pd.Timestamp, pd.Timestamp("2020-02-03"))
        d2 = cast(pd.Timestamp, pd.Timestamp("2020-02-04"))

        selections = {
            d0: RebalanceSelection(
                rebalance_date=d0,
                signal_date=d0,
                bank_symbols=[],
                metal_symbols=["M1"],
                other_symbols=["O1"],
            ),
            d1: RebalanceSelection(
                rebalance_date=d1,
                signal_date=d1,
                bank_symbols=[],
                metal_symbols=["M2"],
                other_symbols=["O2"],
            ),
        }

        (
            daily_targets,
            _,
            _,
            _,
            nonbank_exit_events,
            _,
        ) = build_daily_targets_with_take_profit(
            trading_dates=[d0, d1, d2],
            rebalance_dates=[d0, d1],
            selections=selections,
            close_lookup={
                (d0, "M1"): 100.0,
                (d0, "O1"): 100.0,
                (d1, "M1"): 100.0,
                (d1, "O1"): 100.0,
                (d2, "M1"): 130.0,
                (d2, "O1"): 80.0,
                (d1, "M2"): 50.0,
                (d1, "O2"): 60.0,
                (d2, "M2"): 50.0,
                (d2, "O2"): 60.0,
            },
        )

        self.assertEqual(daily_targets[d0].get("M1", 0.0), 0.25)
        self.assertEqual(daily_targets[d0].get("O1", 0.0), 0.25)
        self.assertEqual(daily_targets[d1].get("M1", 0.0), 0.25)
        self.assertEqual(daily_targets[d1].get("O1", 0.0), 0.25)
        self.assertNotIn("M2", daily_targets[d1])
        self.assertNotIn("O2", daily_targets[d1])

        self.assertNotIn("M1", daily_targets[d2])
        self.assertNotIn("O1", daily_targets[d2])
        self.assertEqual(nonbank_exit_events[(d2, "M1")], TRIGGER_TAKE_PROFIT)
        self.assertEqual(nonbank_exit_events[(d2, "O1")], TRIGGER_STOP_LOSS)


if __name__ == "__main__":
    unittest.main()
