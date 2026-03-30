from __future__ import annotations

import unittest
from typing import cast

import pandas as pd

from scripts.backtest_small_cap_lake import (
    build_daily_plan,
    build_rebalance_plan,
    build_risk_flags_from_sz_name_changes,
    build_timing_flags,
    build_timing_state,
    build_weekly_schedule,
    estimate_announcement_date,
    merge_visible_financial_quality,
    normalize_date_close,
    normalize_financial_quality_rows,
    resolve_mode_configs,
    select_small_cap_candidates,
    update_buffered_holdings,
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

    def test_select_small_cap_candidates_requires_momentum_roe_and_non_st(self) -> None:
        signal_date = cast(pd.Timestamp, pd.Timestamp("2020-01-10"))
        snapshot = pd.DataFrame(
            {
                "symbol": ["A", "B", "C", "D"],
                "industry": ["Tech", "Tech", "Tech", "Tech"],
                "pe_ttm": [5.0, 6.0, 7.0, 8.0],
                "pb": [1.0, 1.1, 1.2, 1.3],
                "total_mv_10k": [100.0, 110.0, 120.0, 130.0],
                "list_date": pd.to_datetime(
                    ["2010-01-01", "2010-01-01", "2010-01-01", "2010-01-01"]
                ),
                "close": [10.0, 8.0, 12.0, 13.0],
                "ma20_stock": [9.0, 9.0, 11.0, 12.0],
                "roe": [5.0, 5.0, -1.0, 6.0],
                "is_st_like": [False, False, False, True],
            }
        )

        out = select_small_cap_candidates(
            snapshot,
            signal_date=signal_date,
            stock_num=3,
            value_quantile=1.0,
            min_list_days=250,
            require_momentum=True,
            require_positive_roe=True,
            exclude_st=True,
        )

        self.assertEqual(out["symbol"].tolist(), ["A"])

    def test_build_timing_state_uses_hysteresis_band(self) -> None:
        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(["2020-01-03", "2020-01-10", "2020-01-17"]),
                "close": [102.0, 100.0, 98.0],
                "ma": [100.0, 100.0, 100.0],
            }
        )

        out = build_timing_state(frame, entry_buffer=0.015, exit_buffer=0.015)

        self.assertEqual(out["safe_market"].tolist(), [True, True, False])

    def test_build_timing_flags_uses_state_machine_not_single_threshold(self) -> None:
        index_series = pd.Series(
            [100.0, 100.0, 104.0, 101.0, 98.0],
            index=pd.to_datetime(
                ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-10", "2020-01-17"]
            ),
            name="close",
        )
        signal_dates = pd.to_datetime(["2020-01-03", "2020-01-10", "2020-01-17"])

        out = build_timing_flags(
            index_series=index_series,
            signal_dates=signal_dates,
            ma_window=3,
            entry_buffer=0.015,
            exit_buffer=0.015,
        )

        self.assertEqual(out["safe_market"].tolist(), [True, True, False])

    def test_update_buffered_holdings_keeps_names_inside_buffer_zone(self) -> None:
        ranked = pd.DataFrame(
            {
                "symbol": ["A", "B", "C", "D", "E", "F"],
                "market_cap_rank": [12, 1, 2, 3, 4, 5],
                "pe_rank": [0.55, 0.2, 0.2, 0.2, 0.2, 0.2],
                "pb_rank": [0.55, 0.2, 0.2, 0.2, 0.2, 0.2],
                "buy_candidate": [False, True, True, True, True, True],
                "hold_candidate": [True, True, True, True, True, True],
            }
        )

        out = update_buffered_holdings(
            previous_holdings=["A"],
            ranked_snapshot=ranked,
            stock_num=3,
            buy_rank_cutoff=3,
            hold_rank_cutoff=30,
            hold_value_quantile=0.6,
        )

        self.assertEqual(out, ["A", "B", "C"])

    def test_update_buffered_holdings_drops_names_outside_sell_zone(self) -> None:
        ranked = pd.DataFrame(
            {
                "symbol": ["A", "B", "C", "D"],
                "market_cap_rank": [31, 1, 2, 3],
                "pe_rank": [0.7, 0.2, 0.2, 0.2],
                "pb_rank": [0.7, 0.2, 0.2, 0.2],
                "buy_candidate": [False, True, True, True],
                "hold_candidate": [False, True, True, True],
            }
        )

        out = update_buffered_holdings(
            previous_holdings=["A"],
            ranked_snapshot=ranked,
            stock_num=3,
            buy_rank_cutoff=3,
            hold_rank_cutoff=30,
            hold_value_quantile=0.6,
        )

        self.assertEqual(out, ["B", "C", "D"])

    def test_estimate_announcement_date_uses_conservative_lag(self) -> None:
        self.assertEqual(
            estimate_announcement_date("2020-03-31"),
            pd.Timestamp("2020-05-15"),
        )
        self.assertEqual(
            estimate_announcement_date("2020-06-30"),
            pd.Timestamp("2020-08-29"),
        )
        self.assertEqual(
            estimate_announcement_date("2020-12-31"),
            pd.Timestamp("2021-04-30"),
        )

    def test_normalize_financial_quality_rows_parses_roe_and_dates(self) -> None:
        raw = pd.DataFrame(
            {
                "报告期": ["2020-03-31", "2020-06-30"],
                "净资产收益率": ["5.2%", "-1.0%"],
            }
        )

        out = normalize_financial_quality_rows("000001.SZ", raw)

        self.assertEqual(out["symbol"].tolist(), ["000001.SZ", "000001.SZ"])
        self.assertEqual(out["roe"].tolist(), [5.2, -1.0])
        self.assertEqual(
            out["announcement_date"].dt.strftime("%Y-%m-%d").tolist(),
            ["2020-05-15", "2020-08-29"],
        )

    def test_merge_visible_financial_quality_uses_latest_visible_report(self) -> None:
        snapshot = pd.DataFrame(
            {
                "signal_date": pd.to_datetime(["2020-05-20", "2020-09-01"]),
                "symbol": ["000001.SZ", "000001.SZ"],
            }
        )
        quality = pd.DataFrame(
            {
                "symbol": ["000001.SZ", "000001.SZ"],
                "report_period": pd.to_datetime(["2020-03-31", "2020-06-30"]),
                "announcement_date": pd.to_datetime(["2020-05-15", "2020-08-29"]),
                "roe": [5.2, 7.8],
            }
        )

        out = merge_visible_financial_quality(snapshot, quality)

        self.assertEqual(out["roe"].tolist(), [5.2, 7.8])

    def test_build_risk_flags_from_sz_name_changes_detects_st_period(self) -> None:
        name_changes = pd.DataFrame(
            {
                "变更日期": pd.to_datetime(["2020-01-10", "2020-03-10"]),
                "证券代码": ["000001", "000001"],
                "变更前简称": ["平安银行", "*ST平安"],
                "变更后简称": ["*ST平安", "平安银行"],
            }
        )
        signal_dates = pd.to_datetime(["2020-01-03", "2020-02-14", "2020-03-20"])

        out = build_risk_flags_from_sz_name_changes(
            change_df=name_changes,
            signal_dates=signal_dates,
            symbols=["000001.SZ"],
        )

        self.assertEqual(
            out.sort_values(["symbol", "date"])["is_st_like"].tolist(),
            [False, True, False],
        )

    def test_resolve_mode_configs_expands_all(self) -> None:
        out = resolve_mode_configs("all")

        self.assertEqual(
            list(out.keys()),
            ["baseline", "buffer_only", "buffer_risk", "full_combo"],
        )


if __name__ == "__main__":
    unittest.main()
