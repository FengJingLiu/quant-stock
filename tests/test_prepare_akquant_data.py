import unittest

import pandas as pd

from scripts.prepare_akquant_data import (
    clean_indicator_frame,
    clean_price_frame,
    merge_symbol_frames,
    normalize_stock_code,
)


class PrepareAkquantDataTests(unittest.TestCase):
    def test_normalize_stock_code(self):
        self.assertEqual(normalize_stock_code(1), "000001")
        self.assertEqual(normalize_stock_code("600519"), "600519")
        self.assertEqual(normalize_stock_code("000001.SZ"), "000001")

    def test_clean_price_frame_maps_columns(self):
        price = pd.DataFrame(
            {
                "日期": ["2026-01-03", "2026-01-02"],
                "股票代码": [1, 1],
                "开盘": [10.0, 9.9],
                "收盘": [10.2, 10.0],
                "最高": [10.3, 10.1],
                "最低": [9.8, 9.7],
                "成交量": [1000, 1200],
                "成交额": [100000, 120000],
                "涨跌幅": [2.0, 1.0],
                "换手率": [0.5, 0.6],
            }
        )

        out = clean_price_frame(price, ts_code="000001.SZ")

        self.assertListEqual(
            list(out.columns[:8]),
            [
                "timestamp",
                "symbol",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "amount",
            ],
        )
        self.assertEqual(out.iloc[0]["timestamp"].strftime("%Y-%m-%d"), "2026-01-02")
        self.assertEqual(out.iloc[0]["symbol"], "000001.SZ")

    def test_clean_indicator_frame_maps_columns(self):
        indicators = pd.DataFrame(
            {
                "股票代码": ["000001.SZ", "000001.SZ"],
                "交易日期": [20260102, 20260103],
                "市盈率TTM": [10.5, 10.8],
                "市净率": [1.2, 1.25],
                "股息率TTM": [2.1, 2.2],
                "量比": [1.3, 1.1],
            }
        )

        out = clean_indicator_frame(indicators)

        self.assertIn("pe_ttm", out.columns)
        self.assertIn("pb", out.columns)
        self.assertIn("dividend_yield_ttm", out.columns)
        self.assertEqual(out.iloc[0]["timestamp"].strftime("%Y-%m-%d"), "2026-01-02")

    def test_clean_indicator_frame_dedup_keeps_per_symbol_per_day(self):
        indicators = pd.DataFrame(
            {
                "股票代码": ["000001.SZ", "000002.SZ", "000001.SZ"],
                "交易日期": [20260102, 20260102, 20260102],
                "市盈率TTM": [10.5, 20.0, 11.2],
                "市净率": [1.2, 2.0, 1.3],
                "股息率TTM": [2.1, 3.0, 2.2],
                "量比": [1.3, 0.8, 1.1],
            }
        )

        out = clean_indicator_frame(indicators)
        self.assertEqual(len(out), 2)

        pe_map = {r["symbol"]: float(r["pe_ttm"]) for _, r in out.iterrows()}
        self.assertEqual(pe_map["000001.SZ"], 11.2)
        self.assertEqual(pe_map["000002.SZ"], 20.0)

    def test_merge_symbol_frames_keeps_akquant_schema(self):
        price = clean_price_frame(
            pd.DataFrame(
                {
                    "日期": ["2026-01-03", "2026-01-02"],
                    "股票代码": [1, 1],
                    "开盘": [10.0, 9.9],
                    "收盘": [10.2, 10.0],
                    "最高": [10.3, 10.1],
                    "最低": [9.8, 9.7],
                    "成交量": [1000, 1200],
                    "成交额": [100000, 120000],
                }
            ),
            ts_code="000001.SZ",
        )
        ind = clean_indicator_frame(
            pd.DataFrame(
                {
                    "股票代码": ["000001.SZ", "000001.SZ"],
                    "交易日期": [20260102, 20260103],
                    "市盈率TTM": [10.5, 10.8],
                }
            )
        )

        merged = merge_symbol_frames(price, ind)

        self.assertListEqual(list(merged["symbol"].unique()), ["000001.SZ"])
        self.assertEqual(merged.iloc[-1]["pe_ttm"], 10.8)
        self.assertTrue((merged["high"] >= merged["open"]).all())


if __name__ == "__main__":
    unittest.main()
