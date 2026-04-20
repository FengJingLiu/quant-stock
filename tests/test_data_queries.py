from __future__ import annotations

import unittest

from src.data_queries import (
    etf_daily_ohlcv_sql,
    etf_daily_open_close_sql,
    latest_index_members_sql,
    latest_index_weights_sql,
    stock_daily_agg_sql,
    stock_qfq_adj_factor_sql,
    trading_dates_sql,
)


class TestDataQueries(unittest.TestCase):
    def test_latest_index_queries_include_asof_subquery(self) -> None:
        members_sql = latest_index_members_sql()
        weights_sql = latest_index_weights_sql()

        self.assertIn("SELECT con_code FROM dim_index_weights", members_sql)
        self.assertIn("SELECT con_code, weight FROM dim_index_weights", weights_sql)
        self.assertIn("SELECT max(trade_date) FROM dim_index_weights", members_sql)
        self.assertIn("trade_date <= %(d)s", weights_sql)

    def test_trading_dates_sql_validates_table(self) -> None:
        sql = trading_dates_sql("klines_1m_etf")
        self.assertIn("FROM klines_1m_etf", sql)
        self.assertIn("symbol = %(sym)s", sql)
        with self.assertRaises(ValueError):
            trading_dates_sql("dim_index_weights")

    def test_stock_daily_agg_and_adj_factor_sql_keep_expected_clauses(self) -> None:
        agg_sql = stock_daily_agg_sql()
        adj_sql = stock_qfq_adj_factor_sql("ts_sym")

        self.assertIn("argMin(open, datetime) as daily_open", agg_sql)
        self.assertIn("argMax(close, datetime) as daily_close", agg_sql)
        self.assertIn("sum(amount) as daily_amount", agg_sql)
        self.assertIn("SELECT symbol AS ts_sym, trade_date, factor", adj_sql)
        self.assertIn("adj_type = 'qfq'", adj_sql)

    def test_etf_daily_sql_builders_cover_both_variants(self) -> None:
        ohlcv_sql = etf_daily_ohlcv_sql()
        open_close_sql = etf_daily_open_close_sql(include_amount=True)

        self.assertIn("max(high) AS high", ohlcv_sql)
        self.assertIn("sum(volume) AS volume", ohlcv_sql)
        self.assertIn("argMin(open, datetime) AS etf_open", open_close_sql)
        self.assertIn("sum(amount) AS etf_amount", open_close_sql)


if __name__ == "__main__":
    unittest.main()
