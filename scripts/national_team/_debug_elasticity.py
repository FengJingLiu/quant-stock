"""Debug: test symbol mapping and daily aggregation from klines_1m_stock."""
import sys
from pathlib import Path

import time

import _load_env  # noqa: F401

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data_clients import create_clickhouse_http_client

ch = create_clickhouse_http_client()

# 1. Symbol mapping: con_code (600519.SH) -> klines_1m_stock symbol (sh600519)
r = ch.query("""
    SELECT con_code,
        if(con_code LIKE '%.SH', concat('sh', substring(con_code, 1, 6)), concat('sz', substring(con_code, 1, 6))) as mapped
    FROM dim_index_weights
    WHERE index_code = '399300.SZ'
      AND trade_date = '2020-02-03'
    LIMIT 10
""")
print("Symbol mapping:", r.result_rows)
print(f"Total members on 2020-02-03: {ch.query('SELECT count() FROM dim_index_weights WHERE index_code=%(i)s AND trade_date=%(d)s', parameters={'i': '399300.SZ', 'd': '2020-02-03'}).result_rows[0][0]}")

# 2. Performance test: aggregate daily OHLC for HS300 stocks over 20 trading days
t0 = time.time()
r2 = ch.query("""
    WITH members AS (
        SELECT
            if(con_code LIKE '%%.SH', concat('sh', substring(con_code, 1, 6)), concat('sz', substring(con_code, 1, 6))) AS sym
        FROM dim_index_weights
        WHERE index_code = '399300.SZ'
          AND trade_date = '2020-02-03'
    )
    SELECT symbol, trade_date,
           argMin(open, datetime) as daily_open,
           max(high) as daily_high,
           min(low) as daily_low,
           argMax(close, datetime) as daily_close,
           sum(amount) as daily_amount
    FROM klines_1m_stock
    WHERE trade_date BETWEEN '2020-02-03' AND '2020-02-28'
      AND symbol IN (SELECT sym FROM members)
    GROUP BY symbol, trade_date
    ORDER BY symbol, trade_date
""")
t1 = time.time()
print(f"\nDaily agg: {len(r2.result_rows)} rows, {t1-t0:.1f}s")
if r2.result_rows:
    print("Sample:", r2.result_rows[:3])
    syms = set(row[0] for row in r2.result_rows)
    print(f"Unique symbols: {len(syms)}")

# 3. Check adj_factor symbol format  
r3 = ch.query("SELECT DISTINCT symbol FROM adj_factor LIMIT 10")
print(f"\nadj_factor symbols: {[row[0] for row in r3.result_rows]}")
