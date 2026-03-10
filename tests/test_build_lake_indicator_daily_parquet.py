import tempfile
import unittest
import zipfile
from pathlib import Path

import duckdb
import pandas as pd

from scripts.build_lake_indicator_daily_parquet import build_indicator_daily_parquet


class BuildLakeIndicatorDailyParquetTests(unittest.TestCase):
    def _valid_indicator_df(self, symbol: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "股票代码": [symbol, symbol],
                "交易日期": [20260102, 20260103],
                "换手率": [1.1, 1.2],
                "换手率(自由流通股)": [1.0, 1.1],
                "量比": [0.9, 1.0],
                "市盈率": [12.0, 12.5],
                "市盈率TTM": [11.0, 11.4],
                "市净率": [1.2, 1.25],
                "市销率": [2.1, 2.2],
                "市销率TTM": [2.0, 2.1],
                "股息率": [1.8, 1.9],
                "股息率TTM": [1.7, 1.8],
                "总股本(万股)": [100.0, 100.0],
                "流通股本(万股)": [80.0, 80.0],
                "自由流通股本(万股)": [70.0, 70.0],
                "总市值(万元)": [1000.0, 1010.0],
                "流通市值(万元)": [800.0, 808.0],
            }
        )

    def _write_zip(self, path: Path, members: dict[str, pd.DataFrame]) -> None:
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for name, df in members.items():
                zf.writestr(name, df.to_csv(index=False))

    def test_build_with_delisted_enabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            indicator_zip = tmp_path / "indicator.zip"
            delisted_zip = tmp_path / "indicator_delisted.zip"
            out_dir = tmp_path / "lake"
            tmp_db = tmp_path / "ingest.duckdb"

            bad_df = pd.DataFrame({"股票代码": ["000003.SZ"], "市盈率TTM": [9.8]})
            self._write_zip(
                indicator_zip,
                {
                    "000001.SZ.csv": self._valid_indicator_df("000001.SZ"),
                    "000003.SZ.csv": bad_df,
                },
            )
            self._write_zip(delisted_zip, {"000002.SZ.csv": self._valid_indicator_df("000002.SZ")})

            stats = build_indicator_daily_parquet(
                indicator_zip=indicator_zip,
                delisted_zip=delisted_zip,
                out_dir=out_dir,
                tmp_db=tmp_db,
                include_delisted=True,
                limit=None,
                overwrite=True,
                progress_every=1,
            )

            con = duckdb.connect()
            lake_glob = out_dir.as_posix() + "/**/*.parquet"
            symbols = {
                row[0]
                for row in con.execute(
                    f"SELECT DISTINCT symbol FROM read_parquet('{lake_glob}') ORDER BY 1"
                ).fetchall()
            }
            self.assertSetEqual(symbols, {"000001.SZ", "000002.SZ"})

            cols = [
                row[0]
                for row in con.execute(
                    f"DESCRIBE SELECT * FROM read_parquet('{lake_glob}') LIMIT 1"
                ).fetchall()
            ]
            self.assertNotIn("pct_chg", cols)
            self.assertGreaterEqual(stats["skipped_parse_fail"], 1)

    def test_build_with_delisted_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            indicator_zip = tmp_path / "indicator.zip"
            delisted_zip = tmp_path / "indicator_delisted.zip"
            out_dir = tmp_path / "lake"
            tmp_db = tmp_path / "ingest.duckdb"

            self._write_zip(indicator_zip, {"000001.SZ.csv": self._valid_indicator_df("000001.SZ")})
            self._write_zip(delisted_zip, {"000002.SZ.csv": self._valid_indicator_df("000002.SZ")})

            build_indicator_daily_parquet(
                indicator_zip=indicator_zip,
                delisted_zip=delisted_zip,
                out_dir=out_dir,
                tmp_db=tmp_db,
                include_delisted=False,
                limit=None,
                overwrite=True,
                progress_every=10,
            )

            con = duckdb.connect()
            lake_glob = out_dir.as_posix() + "/**/*.parquet"
            symbols = {
                row[0]
                for row in con.execute(
                    f"SELECT DISTINCT symbol FROM read_parquet('{lake_glob}') ORDER BY 1"
                ).fetchall()
            }
            self.assertSetEqual(symbols, {"000001.SZ"})


if __name__ == "__main__":
    unittest.main()
