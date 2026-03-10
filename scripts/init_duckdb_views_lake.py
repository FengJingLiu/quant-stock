#!/usr/bin/env python3
"""Initialize DuckDB catalog (views) for the Parquet lake.

Creates/updates:
- v_bar_daily_raw: unadjusted bars (adjust=none)
- v_adj_factor_daily: hfq/qfq factors
- v_bar_daily_hfq: derived hfq prices (权威口径)
- v_bar_daily_qfq: derived qfq prices
- v_indicator_daily: daily indicators
- v_dim_symbol: symbol dimension
- v_daily_hfq_w_ind: hfq bars + indicator columns
- v_daily_hfq_w_ind_dim: v_daily_hfq_w_ind + dim_symbol

DuckDB file:
- data/duckdb/stock.duckdb

This DB is intended as a lightweight catalog; the heavy data stays in Parquet.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb

try:
    from prepare_akquant_data import INDICATOR_COLUMN_MAP
except ModuleNotFoundError:  # pragma: no cover
    from scripts.prepare_akquant_data import INDICATOR_COLUMN_MAP  # type: ignore

INDICATOR_VALUE_COLUMNS = [
    col
    for col in INDICATOR_COLUMN_MAP.values()
    if col not in {"symbol", "trade_date", "pct_chg"}
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Init DuckDB views for stock Parquet lake")
    p.add_argument("--db", type=Path, default=Path("data/duckdb/stock.duckdb"))
    p.add_argument(
        "--raw-glob",
        type=str,
        default="data/lake/fact_bar_daily/adjust=none/**/*.parquet",
        help="glob for raw parquet",
    )
    p.add_argument(
        "--factor-glob",
        type=str,
        default="data/lake/adj_factor_daily/**/*.parquet",
        help="glob for factor parquet",
    )
    p.add_argument(
        "--indicator-glob",
        type=str,
        default="data/lake/fact_indicator_daily/**/*.parquet",
        help="glob for indicator parquet",
    )
    p.add_argument(
        "--dim-glob",
        type=str,
        default="data/lake/dim_symbol/**/*.parquet",
        help="glob for dim_symbol parquet",
    )
    return p.parse_args()


def init_views(
    db_path: Path,
    raw_glob: str,
    factor_glob: str,
    indicator_glob: str,
    dim_glob: str = "data/lake/dim_symbol/**/*.parquet",
) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(db_path.as_posix())
    con.execute("PRAGMA threads=8")

    # Views: raw + factors + indicators + dimension
    con.execute(
        f"""
        CREATE OR REPLACE VIEW v_bar_daily_raw AS
        SELECT * FROM read_parquet('{raw_glob}');
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE VIEW v_adj_factor_daily AS
        SELECT * FROM read_parquet('{factor_glob}');
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE VIEW v_indicator_daily AS
        SELECT * FROM read_parquet('{indicator_glob}');
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE VIEW v_dim_symbol AS
        SELECT * FROM read_parquet('{dim_glob}');
        """
    )

    # Derived (权威口径 = hfq)
    con.execute(
        """
        CREATE OR REPLACE VIEW v_bar_daily_hfq AS
        SELECT
          b.date,
          b.symbol,
          b.open  * f.hfq_factor AS open,
          b.high  * f.hfq_factor AS high,
          b.low   * f.hfq_factor AS low,
          b.close * f.hfq_factor AS close,
          b.volume,
          b.amount,
          f.hfq_factor,
          f.qfq_factor,
          b.year,
          b.month
        FROM v_bar_daily_raw b
        LEFT JOIN v_adj_factor_daily f
          ON b.symbol = f.symbol AND b.date = f.date;
        """
    )

    con.execute(
        """
        CREATE OR REPLACE VIEW v_bar_daily_qfq AS
        SELECT
          b.date,
          b.symbol,
          b.open  * f.qfq_factor AS open,
          b.high  * f.qfq_factor AS high,
          b.low   * f.qfq_factor AS low,
          b.close * f.qfq_factor AS close,
          b.volume,
          b.amount,
          f.hfq_factor,
          f.qfq_factor,
          b.year,
          b.month
        FROM v_bar_daily_raw b
        LEFT JOIN v_adj_factor_daily f
          ON b.symbol = f.symbol AND b.date = f.date;
        """
    )

    indicator_cols_join = ",\n          ".join(f"i.{c}" for c in INDICATOR_VALUE_COLUMNS)
    indicator_cols_select = ", ".join(INDICATOR_VALUE_COLUMNS)
    con.execute(
        f"""
        CREATE OR REPLACE VIEW v_daily_hfq_w_ind AS
        SELECT
          b.date,
          b.symbol,
          b.open,
          b.high,
          b.low,
          b.close,
          b.volume,
          b.amount,
          b.hfq_factor,
          b.qfq_factor,
          b.year,
          b.month,
          {indicator_cols_join}
        FROM v_bar_daily_hfq b
        LEFT JOIN (
          SELECT date, symbol, {indicator_cols_select}
          FROM v_indicator_daily
        ) i
          USING (symbol, date);
        """
    )

    con.execute(
        """
        CREATE OR REPLACE VIEW v_daily_hfq_w_ind_dim AS
        SELECT
          f.*,
          d.code,
          d.name,
          d.region,
          d.area,
          d.industry,
          d.market_type,
          d.market,
          d.exchange,
          d.list_date,
          d.is_delisted,
          d.updated_at
        FROM v_daily_hfq_w_ind f
        LEFT JOIN v_dim_symbol d
          USING (symbol);
        """
    )

    con.close()
    print(f"[DONE] initialized DuckDB views: {db_path}")


def main() -> None:
    args = parse_args()
    init_views(
        db_path=args.db,
        raw_glob=args.raw_glob,
        factor_glob=args.factor_glob,
        indicator_glob=args.indicator_glob,
        dim_glob=args.dim_glob,
    )


if __name__ == "__main__":
    main()
