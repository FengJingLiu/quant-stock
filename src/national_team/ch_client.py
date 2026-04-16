"""
ClickHouse 数据访问层 — astock 数据库 1分钟K线查询 (Polars + clickhouse-connect)

数据路径: clickhouse-connect → query_arrow() → PyArrow Table → pl.from_arrow() (zero-copy)

表:
  astock.klines_1m_stock   股票  symbol='sh600000'
  astock.klines_1m_index   指数  symbol='000300'
  astock.klines_1m_etf     ETF   symbol='510300.SH'
  astock.adj_factor        复权因子

注意:
  - volume 单位是「手」(1手=100股)，amount 单位是「元」
  - 真实 VWAP(元/股) = amount / (volume * 100)
  - 指数无成交量 (volume=0)
  - ClickHouse DateTime 通过 Arrow 输出为 UInt32 (epoch seconds)，需后处理转 Datetime
"""

from __future__ import annotations

from datetime import date

import clickhouse_connect
import polars as pl

_DEFAULT_CH = dict(
    host="localhost",
    port=8123,
    username="default",
    password="***CH_PASSWORD***",
    database="astock",
)

KLINES_SCHEMA = {
    "symbol": pl.Utf8,
    "trade_date": pl.Date,
    "datetime": pl.Datetime("us"),
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
    "amount": pl.Float64,
    "vwap": pl.Float64,
}


def _get_client(ch_kwargs: dict | None = None) -> clickhouse_connect.driver.Client:
    kw = {**_DEFAULT_CH, **(ch_kwargs or {})}
    return clickhouse_connect.get_client(**kw)


def _query_arrow_df(
    sql: str,
    parameters: dict | None = None,
    ch_kwargs: dict | None = None,
) -> pl.DataFrame:
    """
    执行 SQL → Arrow Table → Polars DataFrame (zero-copy)。

    使用 clickhouse-connect 的 query_arrow 接口，
    数据在 Rust/C++ 层面直接从 Arrow 列式格式映射到 Polars，无逐行拷贝。
    """
    client = _get_client(ch_kwargs)
    arrow_tbl = client.query_arrow(sql, parameters=parameters or {})
    if arrow_tbl.num_rows == 0:
        columns = [f.name for f in arrow_tbl.schema]
        return pl.DataFrame(schema={c: pl.Utf8 for c in columns})
    return pl.from_arrow(arrow_tbl)


def _cast_klines(df: pl.DataFrame) -> pl.DataFrame:
    """
    统一 K 线 DataFrame 的列类型。

    clickhouse-connect Arrow 输出中:
      - DateTime → UInt32 (epoch seconds) → 需转 Datetime
      - Float32 → 需转 Float64
    """
    return df.with_columns(
        pl.from_epoch(pl.col("datetime").cast(pl.Int64), time_unit="s")
          .alias("datetime"),
        pl.col("trade_date").cast(pl.Date),
        pl.col("open").cast(pl.Float64),
        pl.col("high").cast(pl.Float64),
        pl.col("low").cast(pl.Float64),
        pl.col("close").cast(pl.Float64),
        pl.col("volume").cast(pl.Float64),
        pl.col("amount").cast(pl.Float64),
        pl.col("vwap").cast(pl.Float64),
    )


# ── 允许的表名白名单 (防止 SQL 注入) ────────────────────────────────────────
_ALLOWED_TABLES = frozenset({
    "klines_1m_stock", "klines_1m_index", "klines_1m_etf",
})


# ── 公共查询接口 ──────────────────────────────────────────────────────────


def get_1m_bars(
    symbol: str,
    start_date: str | date,
    end_date: str | date,
    table: str = "klines_1m_stock",
    ch_kwargs: dict | None = None,
) -> pl.DataFrame:
    """
    查询 1 分钟 K 线数据。

    Returns Polars DataFrame with columns:
      symbol, trade_date, datetime, open, high, low, close, volume, amount, vwap
    按 datetime 升序排列，已去重。
    """
    if table not in _ALLOWED_TABLES:
        raise ValueError(f"table must be one of {_ALLOWED_TABLES}, got {table!r}")

    sql = (
        f"SELECT symbol, trade_date, datetime,"
        f"       open, high, low, close, volume, amount, vwap"
        f"  FROM {table}"
        f" WHERE symbol = {{sym:String}}"
        f"   AND trade_date BETWEEN {{sd:Date}} AND {{ed:Date}}"
        f" ORDER BY datetime"
    )
    df = _query_arrow_df(
        sql,
        parameters={"sym": symbol, "sd": str(start_date), "ed": str(end_date)},
        ch_kwargs=ch_kwargs,
    )
    if df.height == 0:
        return pl.DataFrame(schema=KLINES_SCHEMA)
    df = _cast_klines(df)
    df = df.unique(subset=["datetime"], maintain_order=True)
    return df


def get_stock_1m(
    symbol: str,
    start_date: str | date,
    end_date: str | date,
    ch_kwargs: dict | None = None,
) -> pl.DataFrame:
    """股票 1 分钟线。symbol 如 'sh600000'。"""
    return get_1m_bars(symbol, start_date, end_date, "klines_1m_stock", ch_kwargs)


def get_index_1m(
    symbol: str,
    start_date: str | date,
    end_date: str | date,
    ch_kwargs: dict | None = None,
) -> pl.DataFrame:
    """指数 1 分钟线。symbol 如 '000300' / '000001'。"""
    return get_1m_bars(symbol, start_date, end_date, "klines_1m_index", ch_kwargs)


def get_etf_1m(
    symbol: str,
    start_date: str | date,
    end_date: str | date,
    ch_kwargs: dict | None = None,
) -> pl.DataFrame:
    """ETF 1 分钟线。symbol 如 '510300.SH'。"""
    return get_1m_bars(symbol, start_date, end_date, "klines_1m_etf", ch_kwargs)


def get_trading_dates(
    start_date: str | date,
    end_date: str | date,
    ch_kwargs: dict | None = None,
) -> list[date]:
    """通过指数表获取交易日历。"""
    sql = (
        "SELECT DISTINCT trade_date"
        "  FROM klines_1m_index"
        " WHERE symbol = {sym:String}"
        "   AND trade_date BETWEEN {sd:Date} AND {ed:Date}"
        " ORDER BY trade_date"
    )
    df = _query_arrow_df(
        sql,
        parameters={"sym": "000300", "sd": str(start_date), "ed": str(end_date)},
        ch_kwargs=ch_kwargs,
    )
    if df.height == 0:
        return []
    return df.with_columns(
        pl.col("trade_date").cast(pl.Date)
    )["trade_date"].to_list()


def compute_vwap_price(df: pl.DataFrame) -> pl.Series:
    """
    从 amount 和 volume 计算真实 VWAP（元/股）。

    volume 单位是手 (1手=100股)，amount 单位是元。
    vwap_price = amount / (volume * 100)，volume=0 时用 close 填充。
    """
    return df.select(vwap_price_expr().alias("vwap_price")).to_series()


def vwap_price_expr() -> pl.Expr:
    """VWAP 计算表达式，可直接用于 with_columns。"""
    return (
        pl.when(pl.col("volume") > 0)
          .then(pl.col("amount") / (pl.col("volume") * 100.0))
          .otherwise(pl.col("close").cast(pl.Float64))
    )
