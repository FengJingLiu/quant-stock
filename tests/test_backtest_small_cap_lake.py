from __future__ import annotations

import unittest
from typing import cast

import pandas as pd

from scripts.backtest_small_cap_lake import (
    build_daily_plan,
    build_rebalance_plan,
    build_weekly_schedule,
    normalize_date_close,
    select_small_cap_candidates,
)


class TestBacktestSmallCapLake(unittest.TestCase):
    def test_select_small_cap_candidates_uses_industry_relative_value_and_market_cap(
        self,
    ) -> None:
        signal_date = cast(pd.Timestamp, pd.Timestamp("2020-01-10"))
        snapshot = pd.DataFrame(
            {
                "symbol": [
                    "T1",
                    "T2",
                    "T3",
                    "T4",
                    "M1",
                    "M2",
                    "M3",
                    "M4",
                    "NEW",
                ],
                "industry": [
                    "Tech",
                    "Tech",
                    "Tech",
                    "Tech",
                    "Manu",
                    "Manu",
                    "Manu",
                    "Manu",
                    "Tech",
                ],
                "pe_ttm": [5.0, 10.0, 20.0, 30.0, 4.0, 9.0, 12.0, 15.0, 3.0],
                "pb": [1.0, 1.5, 3.0, 4.0, 0.8, 1.1, 2.0, 2.5, 0.9],
                "total_mv_10k": [
                    300.0,
                    100.0,
                    50.0,
                    30.0,
                    80.0,
                    90.0,
                    40.0,
                    20.0,
                    10.0,
                ],
                "list_date": pd.to_datetime(
                    [
                        "2010-01-01",
                        "2010-01-01",
                        "2010-01-01",
                        "2010-01-01",
                        "2010-01-01",
                        "2010-01-01",
                        "2010-01-01",
                        "2010-01-01",
                        "2019-10-01",
                    ]
                ),
            }
        )

        out = select_small_cap_candidates(
            snapshot,
            signal_date=signal_date,
            stock_num=3,
            value_quantile=0.5,
            min_list_days=250,
        )

        self.assertEqual(out["symbol"].tolist(), ["M1", "M2", "T2"])
        self.assertNotIn("NEW", out["symbol"].tolist())

    def test_build_weekly_schedule_uses_week_last_day_signal_and_next_day_rebalance(
        self,
    ) -> None:
        trading_dates = pd.to_datetime(
            [
                "2020-01-02",
                "2020-01-03",
                "2020-01-06",
                "2020-01-07",
                "2020-01-10",
                "2020-01-13",
            ]
        )

        out = build_weekly_schedule(trading_dates)

        self.assertEqual(out["signal_date"].dt.strftime("%Y-%m-%d").tolist(), ["2020-01-03", "2020-01-10"])
        self.assertEqual(
            out["rebalance_date"].dt.strftime("%Y-%m-%d").tolist(),
            ["2020-01-06", "2020-01-13"],
        )

    def test_build_rebalance_plan_clears_targets_when_index_below_ma20(self) -> None:
        schedule = pd.DataFrame(
            {
                "signal_date": pd.to_datetime(["2020-01-03", "2020-01-10"]),
                "rebalance_date": pd.to_datetime(["2020-01-06", "2020-01-13"]),
            }
        )
        weekly_picks = pd.DataFrame(
            {
                "signal_date": pd.to_datetime(
                    ["2020-01-03", "2020-01-03", "2020-01-10", "2020-01-10"]
                ),
                "symbol": ["A", "B", "C", "D"],
            }
        )
        timing = pd.DataFrame(
            {
                "date": pd.to_datetime(["2020-01-03", "2020-01-10"]),
                "safe_market": [True, False],
            }
        )
        trading_dates = pd.to_datetime(
            ["2020-01-06", "2020-01-07", "2020-01-13", "2020-01-14"]
        )

        rebalance_plan = build_rebalance_plan(schedule, weekly_picks, timing)
        daily_plan = build_daily_plan(trading_dates, rebalance_plan)

        self.assertEqual(rebalance_plan[pd.Timestamp("2020-01-06")], ["A", "B"])
        self.assertEqual(rebalance_plan[pd.Timestamp("2020-01-13")], [])
        self.assertEqual(daily_plan[pd.Timestamp("2020-01-07")], ["A", "B"])
        self.assertEqual(daily_plan[pd.Timestamp("2020-01-13")], [])
        self.assertEqual(daily_plan[pd.Timestamp("2020-01-14")], [])

    def test_normalize_date_close_accepts_chinese_columns(self) -> None:
        raw = pd.DataFrame(
            {
                "日期": ["2020-01-02", "2020-01-03"],
                "收盘": [100.0, 101.5],
            }
        )

        out = normalize_date_close(raw)

        self.assertEqual(out.index.strftime("%Y-%m-%d").tolist(), ["2020-01-02", "2020-01-03"])
        self.assertEqual(out.tolist(), [100.0, 101.5])


if __name__ == "__main__":
    unittest.main()
