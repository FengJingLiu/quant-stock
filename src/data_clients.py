"""
统一数据客户端入口。

目标:
- ClickHouse 统一走这里创建 HTTP client
- Tushare Pro 统一走这里获取已配置实例
- AKShare 统一走这里安装 proxy patch / 获取模块
"""

from __future__ import annotations

import importlib
from typing import Any, cast

import clickhouse_connect
import polars as pl
import tushare as ts

from src.config import (
    AKSHARE_PROXY_HOST,
    AKSHARE_PROXY_RETRY,
    AKSHARE_PROXY_TOKEN,
    CH_HTTP_KWARGS,
    TUSHARE_API_URL,
    TUSHARE_TOKEN,
)

_TushareKey = tuple[str, str]
_AkshareKey = tuple[str, str, int]

_tushare_pro: Any | None = None
_tushare_key: _TushareKey | None = None
_akshare_module: Any | None = None
_akshare_key: _AkshareKey | None = None
_akshare_patch_key: _AkshareKey | None = None


def create_clickhouse_http_client(
    overrides: dict[str, Any] | None = None,
) -> Any:
    """返回 ClickHouse HTTP client。"""
    kwargs = cast(dict[str, Any], {**CH_HTTP_KWARGS, **(overrides or {})})
    return clickhouse_connect.get_client(**kwargs)


def query_clickhouse_arrow_df(
    sql: str,
    parameters: dict[str, Any] | None = None,
    *,
    client: Any | None = None,
    client_overrides: dict[str, Any] | None = None,
) -> pl.DataFrame:
    """执行 query_arrow 并转成 Polars DataFrame。"""
    query_client = client if client is not None else create_clickhouse_http_client(client_overrides)
    return cast(
        pl.DataFrame,
        pl.from_arrow(query_client.query_arrow(sql, parameters=parameters or {})),
    )


def query_clickhouse_rows(
    sql: str,
    parameters: dict[str, Any] | None = None,
    *,
    client: Any | None = None,
    client_overrides: dict[str, Any] | None = None,
) -> list[tuple[Any, ...]]:
    """执行 query 并返回 result_rows。"""
    query_client = client if client is not None else create_clickhouse_http_client(client_overrides)
    return cast(
        list[tuple[Any, ...]],
        query_client.query(sql, parameters=parameters or {}).result_rows,
    )


def query_clickhouse_scalar(
    sql: str,
    parameters: dict[str, Any] | None = None,
    *,
    client: Any | None = None,
    client_overrides: dict[str, Any] | None = None,
) -> Any:
    """执行 query 并返回第一行第一列；无结果时返回 None。"""
    rows = query_clickhouse_rows(
        sql,
        parameters=parameters,
        client=client,
        client_overrides=client_overrides,
    )
    if not rows:
        return None
    return rows[0][0]


def get_tushare_pro(
    *,
    token: str | None = None,
    api_url: str | None = None,
) -> Any:
    """返回已配置的 Tushare Pro 单例。"""
    global _tushare_pro, _tushare_key

    effective_token = (token or TUSHARE_TOKEN).strip()
    effective_api_url = (api_url or TUSHARE_API_URL).strip()
    key = (effective_token, effective_api_url)

    if _tushare_pro is None or _tushare_key != key:
        pro = ts.pro_api(effective_token)
        setattr(pro, "_DataApi__http_url", effective_api_url)
        _tushare_pro = pro
        _tushare_key = key
    return _tushare_pro


def ensure_akshare_proxy_patch(
    *,
    proxy_host: str | None = None,
    token: str | None = None,
    retry: int | None = None,
) -> None:
    """安装 AKShare proxy patch；相同参数只安装一次。"""
    global _akshare_patch_key

    effective_host = (proxy_host or AKSHARE_PROXY_HOST).strip()
    effective_token = (token or AKSHARE_PROXY_TOKEN).strip()
    effective_retry = max(1, int(retry or AKSHARE_PROXY_RETRY))

    if not effective_token:
        raise ValueError("AKSHARE_PROXY_TOKEN 未设置，无法安装 akshare_proxy_patch")

    key = (effective_host, effective_token, effective_retry)
    if _akshare_patch_key == key:
        return

    akshare_proxy_patch = importlib.import_module("akshare_proxy_patch")
    cast(Any, akshare_proxy_patch).install_patch(
        effective_host,
        effective_token,
        retry=effective_retry,
    )
    _akshare_patch_key = key


def get_akshare(
    *,
    proxy_host: str | None = None,
    token: str | None = None,
    retry: int | None = None,
) -> Any:
    """返回已安装 proxy patch 的 akshare 模块。"""
    global _akshare_module, _akshare_key

    effective_host = (proxy_host or AKSHARE_PROXY_HOST).strip()
    effective_token = (token or AKSHARE_PROXY_TOKEN).strip()
    effective_retry = max(1, int(retry or AKSHARE_PROXY_RETRY))
    key = (effective_host, effective_token, effective_retry)

    ensure_akshare_proxy_patch(
        proxy_host=effective_host,
        token=effective_token,
        retry=effective_retry,
    )
    if _akshare_module is None or _akshare_key != key:
        _akshare_module = importlib.import_module("akshare")
        _akshare_key = key
    return _akshare_module


__all__ = [
    "create_clickhouse_http_client",
    "query_clickhouse_arrow_df",
    "query_clickhouse_rows",
    "query_clickhouse_scalar",
    "get_tushare_pro",
    "ensure_akshare_proxy_patch",
    "get_akshare",
]
