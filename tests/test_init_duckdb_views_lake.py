import tempfile
import unittest
from pathlib import Path

import duckdb

from scripts.init_duckdb_views_lake import init_views


class InitDuckdbViewsLakeTests(unittest.TestCase):
    def test_init_views_creates_indicator_join_view(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            raw_dir = base / "fact_bar_daily" / "adjust=none"
            factor_dir = base / "adj_factor_daily"
            ind_dir = base / "fact_indicator_daily"
            dim_dir = base / "dim_symbol"
            db = base / "stock.duckdb"
            raw_dir.mkdir(parents=True, exist_ok=True)
            factor_dir.mkdir(parents=True, exist_ok=True)
            ind_dir.mkdir(parents=True, exist_ok=True)
            dim_dir.mkdir(parents=True, exist_ok=True)

            con = duckdb.connect()
            con.execute(
                f"""
                COPY (
                  SELECT
                    DATE '2026-01-02' AS date,
                    '000001.SZ' AS symbol,
                    10.0 AS open,
                    11.0 AS high,
                    9.5 AS low,
                    10.5 AS close,
                    1000.0 AS volume,
                    10000.0 AS amount,
                    2026 AS year,
                    1 AS month
                ) TO '{raw_dir.as_posix()}'
                (FORMAT PARQUET, PARTITION_BY (year, month), COMPRESSION ZSTD)
                """
            )
            con.execute(
                f"""
                COPY (
                  SELECT
                    DATE '2026-01-02' AS date,
                    '000001.SZ' AS symbol,
                    2.0 AS hfq_factor,
                    1.5 AS qfq_factor,
                    2026 AS year,
                    1 AS month
                ) TO '{factor_dir.as_posix()}'
                (FORMAT PARQUET, PARTITION_BY (year, month), COMPRESSION ZSTD)
                """
            )
            con.execute(
                f"""
                COPY (
                  SELECT
                    DATE '2026-01-02' AS date,
                    '000001.SZ' AS symbol,
                    1.23 AS turnover_rate,
                    1.11 AS turnover_rate_free,
                    0.95 AS volume_ratio,
                    12.0 AS pe,
                    11.4 AS pe_ttm,
                    1.25 AS pb,
                    2.3 AS ps,
                    2.2 AS ps_ttm,
                    1.8 AS dividend_yield,
                    1.7 AS dividend_yield_ttm,
                    100.0 AS total_share_10k,
                    80.0 AS float_share_10k,
                    70.0 AS free_float_share_10k,
                    1000.0 AS total_mv_10k,
                    800.0 AS circ_mv_10k,
                    2026 AS year,
                    1 AS month
                ) TO '{ind_dir.as_posix()}'
                (FORMAT PARQUET, PARTITION_BY (year, month), COMPRESSION ZSTD)
                """
            )
            con.execute(
                f"""
                COPY (
                  SELECT
                    '000001.SZ' AS symbol,
                    '000001' AS code,
                    '平安银行' AS name,
                    '深圳' AS region,
                    '深圳' AS area,
                    '银行' AS industry,
                    '主板' AS market_type,
                    '主板' AS market,
                    'SZSE' AS exchange,
                    DATE '1991-04-03' AS list_date,
                    0 AS is_delisted,
                    NOW()::TIMESTAMP AS updated_at
                ) TO '{(dim_dir / 'symbols.parquet').as_posix()}'
                (FORMAT PARQUET, COMPRESSION ZSTD)
                """
            )

            init_views(
                db_path=db,
                raw_glob=raw_dir.as_posix() + '/**/*.parquet',
                factor_glob=factor_dir.as_posix() + '/**/*.parquet',
                indicator_glob=ind_dir.as_posix() + '/**/*.parquet',
                dim_glob=dim_dir.as_posix() + '/**/*.parquet',
            )

            db_con = duckdb.connect(db.as_posix())
            row = db_con.execute(
                """
                SELECT close, pe_ttm, pb, industry, name, year, month
                FROM v_daily_hfq_w_ind_dim
                WHERE symbol='000001.SZ' AND date=DATE '2026-01-02'
                """
            ).fetchone()

            self.assertIsNotNone(row)
            self.assertAlmostEqual(row[0], 21.0)
            self.assertAlmostEqual(float(row[1]), 11.4)
            self.assertAlmostEqual(float(row[2]), 1.25)
            self.assertEqual(row[3], "银行")
            self.assertEqual(row[4], "平安银行")
            self.assertEqual(row[5], 2026)
            self.assertEqual(row[6], 1)


if __name__ == "__main__":
    unittest.main()
