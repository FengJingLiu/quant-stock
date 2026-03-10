import tempfile
import unittest
import zipfile
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd

from scripts.sync_lake_daily import sync_lake_daily


class SyncLakeDailyTests(unittest.TestCase):
    def _write_increment_csv(self, path: Path) -> None:
        df = pd.DataFrame(
            {
                "股票代码": ["000001.SZ", "000002.SZ", "000001.SZ"],
                "交易日期": [20260303, 20260303, 20260303],
                "开盘价": [10.0, 20.0, 10.1],
                "最高价": [10.5, 20.5, 10.6],
                "最低价": [9.8, 19.8, 9.9],
                "收盘价": [10.2, 20.2, 10.3],
                "昨收价": [10.0, 20.0, 10.2],
                "涨跌额": [0.2, 0.2, 0.1],
                "涨跌幅": [2.0, 1.0, 1.0],
                "成交量(手)": [1000, 2000, 1100],
                "成交额(千元)": [100, 300, 120],
                "换手率": [1.0, 1.2, 1.1],
                "换手率(自由流通股)": [1.1, 1.3, 1.2],
                "量比": [0.8, 0.9, 0.85],
                "市盈率": [10.0, 20.0, 10.5],
                "市盈率TTM": [11.0, 21.0, 11.5],
                "市净率": [1.2, 2.2, 1.3],
                "市销率": [2.0, 3.0, 2.1],
                "市销率TTM": [2.1, 3.1, 2.2],
                "股息率": [1.8, 2.8, 1.9],
                "股息率TTM": [1.7, 2.7, 1.8],
                "总股本(万股)": [1000, 2000, 1000],
                "流通股本(万股)": [800, 1500, 800],
                "自由流通股本(万股)": [700, 1200, 700],
                "总市值(万元)": [10000, 20000, 10050],
                "流通市值(万元)": [8000, 15000, 8050],
            }
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)

    def _write_zip_close(self, zip_path: Path, price_map: dict[str, float]) -> None:
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for code, close in price_map.items():
                df = pd.DataFrame({"日期": ["2026-03-03"], "收盘": [close]})
                zf.writestr(f"{code}_daily.csv", df.to_csv(index=False))

    def _write_symbol_lists(self, stock_list: Path, delisted_list: Path) -> None:
        cols = [
            "TS代码",
            "股票代码",
            "股票名称",
            "地域",
            "所属行业",
            "股票全称",
            "英文全称",
            "拼音缩写",
            "市场类型",
            "交易所代码",
            "上市日期",
            "实控人名称",
            "实控人企业性质",
        ]
        active = pd.DataFrame(
            [
                [
                    "000001.SZ",
                    1,
                    "平安银行",
                    "深圳",
                    "银行",
                    "平安银行股份有限公司",
                    "Ping An Bank Co., Ltd.",
                    "PAYH",
                    "主板",
                    "SZSE",
                    19910403,
                    "无实际控制人",
                    "其他",
                ],
                [
                    "000002.SZ",
                    2,
                    "万科A",
                    "深圳",
                    "全国地产",
                    "万科企业股份有限公司",
                    "China Vanke Co.,Ltd.",
                    "WKA",
                    "主板",
                    "SZSE",
                    19910129,
                    "无实际控制人",
                    "其他",
                ],
            ],
            columns=cols,
        )
        delisted = pd.DataFrame(columns=cols)

        stock_list.parent.mkdir(parents=True, exist_ok=True)
        active.to_csv(stock_list, index=False)
        delisted.to_csv(delisted_list, index=False)

    def test_sync_lake_daily_dry_run_core_logic(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inc = root / "增量数据" / "每日指标" / "2026-03" / "20260303.csv"
            self._write_increment_csv(inc)

            raw_zip = root / "A股数据_zip" / "daily.zip"
            hfq_zip = root / "A股数据_zip" / "daily_hfq.zip"
            qfq_zip = root / "A股数据_zip" / "daily_qfq.zip"
            self._write_zip_close(raw_zip, {"000001": 10.3, "000002": 20.2})
            self._write_zip_close(hfq_zip, {"000001": 20.6, "000002": 30.3})
            self._write_zip_close(qfq_zip, {"000001": 9.27, "000002": 18.18})

            stock_list = root / "A股数据_每日指标" / "股票列表.csv"
            delisted_list = root / "A股数据_每日指标" / "退市股票列表.csv"
            self._write_symbol_lists(stock_list, delisted_list)

            summary = sync_lake_daily(
                incremental_root=root / "增量数据" / "每日指标",
                reprocess_days=3,
                dates=None,
                raw_base_dir=root / "data/lake/fact_bar_daily/adjust=none",
                indicator_base_dir=root / "data/lake/fact_indicator_daily",
                factor_base_dir=root / "data/lake/adj_factor_daily",
                update_factors=True,
                raw_zip=raw_zip,
                hfq_zip=hfq_zip,
                qfq_zip=qfq_zip,
                stock_list=stock_list,
                delisted_stock_list=delisted_list,
                dim_out_file=root / "data/lake/dim_symbol/symbols.parquet",
                db_path=root / "data/duckdb/stock.duckdb",
                raw_glob=(root / "data/lake/fact_bar_daily/adjust=none/**/*.parquet").as_posix(),
                factor_glob=(root / "data/lake/adj_factor_daily/**/*.parquet").as_posix(),
                indicator_glob=(root / "data/lake/fact_indicator_daily/**/*.parquet").as_posix(),
                dim_glob=(root / "data/lake/dim_symbol/**/*.parquet").as_posix(),
                dry_run=True,
            )

            self.assertEqual(summary["bar_delta_rows"], 2)
            self.assertEqual(summary["indicator_delta_rows"], 2)
            self.assertEqual(summary["factor_stats"]["delta_rows"], 2)
            self.assertEqual(summary["bar_stats"]["months"], 1)

    def test_sync_lake_daily_non_dry_run_refreshes_views_with_dim(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inc = root / "增量数据" / "每日指标" / "2026-03" / "20260303.csv"
            self._write_increment_csv(inc)

            raw_zip = root / "A股数据_zip" / "daily.zip"
            hfq_zip = root / "A股数据_zip" / "daily_hfq.zip"
            qfq_zip = root / "A股数据_zip" / "daily_qfq.zip"
            self._write_zip_close(raw_zip, {"000001": 10.3, "000002": 20.2})
            self._write_zip_close(hfq_zip, {"000001": 20.6, "000002": 30.3})
            self._write_zip_close(qfq_zip, {"000001": 9.27, "000002": 18.18})

            stock_list = root / "A股数据_每日指标" / "股票列表.csv"
            delisted_list = root / "A股数据_每日指标" / "退市股票列表.csv"
            self._write_symbol_lists(stock_list, delisted_list)

            db_path = root / "data/duckdb/stock.duckdb"
            sync_lake_daily(
                incremental_root=root / "增量数据" / "每日指标",
                reprocess_days=3,
                dates=[date(2026, 3, 3)],
                raw_base_dir=root / "data/lake/fact_bar_daily/adjust=none",
                indicator_base_dir=root / "data/lake/fact_indicator_daily",
                factor_base_dir=root / "data/lake/adj_factor_daily",
                update_factors=True,
                raw_zip=raw_zip,
                hfq_zip=hfq_zip,
                qfq_zip=qfq_zip,
                stock_list=stock_list,
                delisted_stock_list=delisted_list,
                dim_out_file=root / "data/lake/dim_symbol/symbols.parquet",
                db_path=db_path,
                raw_glob=(root / "data/lake/fact_bar_daily/adjust=none/**/*.parquet").as_posix(),
                factor_glob=(root / "data/lake/adj_factor_daily/**/*.parquet").as_posix(),
                indicator_glob=(root / "data/lake/fact_indicator_daily/**/*.parquet").as_posix(),
                dim_glob=(root / "data/lake/dim_symbol/**/*.parquet").as_posix(),
                dry_run=False,
            )

            con = duckdb.connect(db_path.as_posix())
            row = con.execute(
                """
                SELECT symbol, close, pe_ttm, industry, name
                FROM v_daily_hfq_w_ind_dim
                WHERE date = DATE '2026-03-03' AND symbol='000001.SZ'
                """
            ).fetchone()
            self.assertIsNotNone(row)
            self.assertEqual(row[0], "000001.SZ")
            self.assertAlmostEqual(float(row[1]), 20.6, places=6)
            self.assertAlmostEqual(float(row[2]), 11.5, places=6)
            self.assertEqual(row[3], "银行")
            self.assertEqual(row[4], "平安银行")


if __name__ == "__main__":
    unittest.main()
