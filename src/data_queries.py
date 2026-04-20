"""
重复 SQL 查询模板集中管理。

约定:
- 使用 clickhouse-connect 的 ``%(name)s`` 参数风格
- 仅负责生成 SQL，不直接执行查询
"""

from __future__ import annotations

_ALLOWED_MINUTE_TABLES = frozenset({
    "klines_1m_stock",
    "klines_1m_index",
    "klines_1m_etf",
})
_ALLOWED_TICK_TABLES = frozenset({
    "tick_cb",
    "tick_stock_cb_underlying",
})


def _validate_minute_table(table: str) -> str:
    if table not in _ALLOWED_MINUTE_TABLES:
        raise ValueError(f"table must be one of {_ALLOWED_MINUTE_TABLES}, got {table!r}")
    return table


def _validate_tick_table(table: str) -> str:
    if table not in _ALLOWED_TICK_TABLES:
        raise ValueError(f"table must be one of {_ALLOWED_TICK_TABLES}, got {table!r}")
    return table


def tick_trade_dates_sql(table: str) -> str:
    safe_table = _validate_tick_table(table)
    return f"""
        SELECT DISTINCT trade_date
        FROM {safe_table}
        WHERE trade_date BETWEEN %(sd)s AND %(ed)s
        ORDER BY trade_date
    """


def tick_day_rows_sql(table: str) -> str:
    safe_table = _validate_tick_table(table)
    return f"""
        SELECT ts, symbol, price, volume
        FROM {safe_table}
        WHERE trade_date = %(d)s
          AND symbol IN %(syms)s
        ORDER BY symbol, ts
    """


def latest_index_members_sql() -> str:
    return """
        SELECT con_code FROM dim_index_weights
        WHERE index_code = %(idx)s
          AND trade_date = (
              SELECT max(trade_date) FROM dim_index_weights
              WHERE index_code = %(idx)s AND trade_date <= %(d)s
          )
    """


def latest_index_weights_sql() -> str:
    return """
        SELECT con_code, weight FROM dim_index_weights
        WHERE index_code = %(idx)s
          AND trade_date = (
              SELECT max(trade_date) FROM dim_index_weights
              WHERE index_code = %(idx)s AND trade_date <= %(d)s
          )
    """


def trading_dates_sql(table: str = "klines_1m_index") -> str:
    safe_table = _validate_minute_table(table)
    return f"""
        SELECT DISTINCT trade_date FROM {safe_table}
        WHERE symbol = %(sym)s
          AND trade_date BETWEEN %(sd)s AND %(ed)s
        ORDER BY trade_date
    """


def stock_daily_agg_sql() -> str:
    return """
        SELECT symbol, trade_date,
               argMin(open, datetime) as daily_open,
               max(high) as daily_high,
               min(low) as daily_low,
               argMax(close, datetime) as daily_close,
               sum(amount) as daily_amount
        FROM klines_1m_stock
        WHERE trade_date BETWEEN %(sd)s AND %(ed)s
          AND symbol IN %(syms)s
        GROUP BY symbol, trade_date
        ORDER BY symbol, trade_date
    """


def stock_daily_close_sql() -> str:
    return """
        SELECT symbol, trade_date,
               argMax(close, datetime) as daily_close
        FROM klines_1m_stock
        WHERE trade_date BETWEEN %(sd)s AND %(ed)s
          AND symbol IN %(syms)s
        GROUP BY symbol, trade_date
        ORDER BY symbol, trade_date
    """


def stock_qfq_adj_factor_sql(symbol_alias: str = "symbol") -> str:
    return f"""
        SELECT symbol AS {symbol_alias}, trade_date, factor
        FROM adj_factor
        WHERE adj_type = 'qfq'
          AND fund_type = 'stock'
          AND trade_date BETWEEN %(sd)s AND %(ed)s
          AND symbol IN %(syms)s
        ORDER BY {symbol_alias}, trade_date
    """


def index_daily_open_close_sql() -> str:
    return """
        SELECT trade_date,
               argMin(open, datetime) as d_open,
               argMax(close, datetime) as d_close
        FROM klines_1m_index
        WHERE symbol = %(sym)s
          AND trade_date BETWEEN %(sd)s AND %(ed)s
        GROUP BY trade_date
        ORDER BY trade_date
    """


def etf_daily_ohlcv_sql() -> str:
    return """
        SELECT trade_date,
               argMin(open, datetime) AS open,
               max(high) AS high,
               min(low) AS low,
               argMax(close, datetime) AS close,
               sum(volume) AS volume
        FROM klines_1m_etf
        WHERE symbol = %(sym)s
          AND trade_date BETWEEN %(sd)s AND %(ed)s
        GROUP BY trade_date
        ORDER BY trade_date
    """


def etf_daily_open_close_sql(
    *,
    open_alias: str = "etf_open",
    close_alias: str = "etf_close",
    include_amount: bool = False,
) -> str:
    amount_clause = ",\n               sum(amount) AS etf_amount" if include_amount else ""
    return f"""
        SELECT trade_date,
               argMin(open, datetime) AS {open_alias},
               argMax(close, datetime) AS {close_alias}{amount_clause}
        FROM klines_1m_etf
        WHERE symbol = %(sym)s
          AND trade_date BETWEEN %(sd)s AND %(ed)s
        GROUP BY trade_date
        ORDER BY trade_date
    """


def etf_daily_close_sql(close_alias: str = "etf_close") -> str:
    return f"""
        SELECT trade_date,
               argMax(close, datetime) as {close_alias}
        FROM klines_1m_etf
        WHERE symbol = %(sym)s
          AND trade_date BETWEEN %(sd)s AND %(ed)s
        GROUP BY trade_date
        ORDER BY trade_date
    """


__all__ = [
    "tick_trade_dates_sql",
    "tick_day_rows_sql",
    "latest_index_members_sql",
    "latest_index_weights_sql",
    "trading_dates_sql",
    "stock_daily_agg_sql",
    "stock_daily_close_sql",
    "stock_qfq_adj_factor_sql",
    "index_daily_open_close_sql",
    "etf_daily_ohlcv_sql",
    "etf_daily_open_close_sql",
    "etf_daily_close_sql",
]
