#!/usr/bin/env python3
"""A-share data router: local DuckDB first, AkShare fallback.

Usage examples:
    .venv/bin/python skills/ashare-data-duckdb-first/scripts/ashare_data_router.py 600519 --mode history --period 1y --adjust hfq
    .venv/bin/python skills/ashare-data-duckdb-first/scripts/ashare_data_router.py 000001 --mode latest --pretty
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any


DEFAULT_DB_PATH = Path("/home/autumn/quant/stock/data/duckdb/stock.duckdb")
DEFAULT_TOKEN = os.environ.get("AKSHARE_PROXY_TOKEN", "")
DEFAULT_PROXY_HOST = os.environ.get("AKSHARE_PROXY_HOST", "")


@dataclass(frozen=True)
class RequestConfig:
    symbol: str
    symbol_ts: str
    symbol_ak: str
    mode: str
    adjust: str
    start_date: date
    end_date: date
    min_rows: int
    max_local_lag_days: int


def _period_to_days(period: str) -> int:
    mapping = {
        "1m": 30,
        "3m": 90,
        "6m": 180,
        "1y": 365,
        "2y": 730,
        "5y": 1825,
        "max": 7300,
    }
    return mapping.get(period, 365)


def _normalize_symbol_code(symbol: str) -> str:
    upper = symbol.strip().upper()

    match = re.search(r"(\d{6})", upper)
    if match:
        return match.group(1)

    cleaned = re.sub(r"[^0-9]", "", upper)
    if len(cleaned) == 6:
        return cleaned

    raise ValueError(f"Invalid A-share symbol: {symbol}")


def _to_ts_symbol(code6: str) -> str:
    if code6.startswith("6"):
        return f"{code6}.SH"
    if code6.startswith(("0", "3")):
        return f"{code6}.SZ"
    if code6.startswith(("4", "8")):
        return f"{code6}.BJ"
    raise ValueError(f"Unsupported A-share code prefix: {code6}")


def _to_iso_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _date_to_ak(value: date) -> str:
    return value.strftime("%Y%m%d")


def _to_jsonable(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


def _build_request(args: argparse.Namespace) -> RequestConfig:
    code6 = _normalize_symbol_code(args.symbol)
    symbol_ts = _to_ts_symbol(code6)

    if bool(args.start_date) ^ bool(args.end_date):
        raise ValueError(
            "start-date and end-date must be provided together, or use --period"
        )

    if args.start_date and args.end_date:
        start_date = _to_iso_date(args.start_date)
        end_date = _to_iso_date(args.end_date)
    else:
        end_date = date.today()
        start_date = end_date - timedelta(days=_period_to_days(args.period))

    if start_date > end_date:
        raise ValueError("start-date cannot be later than end-date")

    return RequestConfig(
        symbol=args.symbol,
        symbol_ts=symbol_ts,
        symbol_ak=code6,
        mode=args.mode,
        adjust=args.adjust,
        start_date=start_date,
        end_date=end_date,
        min_rows=max(args.min_rows, 1),
        max_local_lag_days=max(args.max_local_lag_days, 0),
    )


def _local_view_exists(conn: Any, view_name: str) -> bool:
    row = conn.execute(
        """
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE lower(table_name) = lower(?)
        """,
        [view_name],
    ).fetchone()
    return bool(row and row[0] > 0)


def _history_view_candidates(adjust: str) -> list[str]:
    if adjust == "qfq":
        return ["v_bar_daily_qfq", "v_bar_daily_hfq", "v_bar_daily_raw"]
    if adjust == "hfq":
        return ["v_bar_daily_hfq", "v_bar_daily_raw"]
    return ["v_bar_daily_raw", "v_bar_daily_hfq"]


def _history_records_from_df(df: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        records.append(
            {
                "date": str(_to_jsonable(row.get("date")))[:10],
                "open": _to_jsonable(row.get("open")),
                "high": _to_jsonable(row.get("high")),
                "low": _to_jsonable(row.get("low")),
                "close": _to_jsonable(row.get("close")),
                "volume": _to_jsonable(row.get("volume")),
                "amount": _to_jsonable(row.get("amount")),
            }
        )
    return records


def _is_history_local_sufficient(
    records: list[dict[str, Any]],
    request: RequestConfig,
) -> tuple[bool, str | None]:
    if len(records) < request.min_rows:
        return False, f"local_rows<{request.min_rows}"

    if not records:
        return False, "local_empty"

    last_date_text = records[-1].get("date")
    if not isinstance(last_date_text, str) or len(last_date_text) < 10:
        return False, "local_last_date_missing"

    first_date_text = records[0].get("date")
    if not isinstance(first_date_text, str) or len(first_date_text) < 10:
        return False, "local_first_date_missing"

    local_last = datetime.strptime(last_date_text[:10], "%Y-%m-%d").date()
    local_first = datetime.strptime(first_date_text[:10], "%Y-%m-%d").date()

    coverage_tolerance_days = 7
    if local_first > (request.start_date + timedelta(days=coverage_tolerance_days)):
        return False, f"local_range_incomplete_first={local_first.isoformat()}"

    lag_days = (request.end_date - local_last).days
    if lag_days > request.max_local_lag_days:
        return False, f"local_stale_lag_days={lag_days}"

    return True, None


def _adjust_from_view(view_name: str) -> str:
    name = view_name.lower()
    if "qfq" in name:
        return "qfq"
    if "hfq" in name:
        return "hfq"
    return "none"


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _compose_attempt_reason(prefix: str, attempts: list[str]) -> str:
    if not attempts:
        return prefix
    selected = attempts[:6]
    suffix = ";..." if len(attempts) > 6 else ""
    return f"{prefix}:{';'.join(selected)}{suffix}"


def _fetch_local_history(
    db_path: Path, request: RequestConfig
) -> tuple[dict[str, Any] | None, str | None]:
    if not db_path.exists():
        return None, f"duckdb_not_found:{db_path}"

    import duckdb

    attempts: list[str] = []
    with duckdb.connect(str(db_path), read_only=True) as conn:
        for view_name in _history_view_candidates(request.adjust):
            if not _local_view_exists(conn, view_name):
                attempts.append(f"{view_name}=view_missing")
                continue

            try:
                df = conn.execute(
                    f"""
                    SELECT date, open, high, low, close, volume, amount
                    FROM {view_name}
                    WHERE symbol = ?
                      AND date BETWEEN ? AND ?
                    ORDER BY date
                    """,
                    [
                        request.symbol_ts,
                        request.start_date.isoformat(),
                        request.end_date.isoformat(),
                    ],
                ).fetchdf()
            except Exception as exc:
                attempts.append(f"{view_name}=query_error:{exc}")
                continue

            if df.empty:
                attempts.append(f"{view_name}=empty")
                continue

            records = _history_records_from_df(df)
            ok, reason = _is_history_local_sufficient(records, request)
            if not ok:
                attempts.append(f"{view_name}={reason or 'insufficient'}")
                continue

            return {
                "source": "local",
                "symbol": request.symbol_ts,
                "mode": "history",
                "data": records,
                "meta": {
                    "local_view": view_name,
                    "adjust_applied": _adjust_from_view(view_name),
                    "rows": len(records),
                    "start_date": records[0]["date"] if records else None,
                    "end_date": records[-1]["date"] if records else None,
                    "attempted_views": len(attempts) + 1,
                },
            }, None

    return None, _compose_attempt_reason("local_no_sufficient_history", attempts)


def _latest_candidates() -> list[str]:
    return [
        "v_daily_hfq_w_ind_dim",
        "v_daily_hfq_w_ind",
        "v_bar_daily_hfq",
        "v_bar_daily_raw",
    ]


def _fetch_local_latest(
    db_path: Path, request: RequestConfig
) -> tuple[dict[str, Any] | None, str | None]:
    if not db_path.exists():
        return None, f"duckdb_not_found:{db_path}"

    import duckdb

    attempts: list[str] = []
    with duckdb.connect(str(db_path), read_only=True) as conn:
        for view_name in _latest_candidates():
            if not _local_view_exists(conn, view_name):
                attempts.append(f"{view_name}=view_missing")
                continue

            if view_name == "v_daily_hfq_w_ind_dim":
                query = """
                    SELECT
                        symbol,
                        name,
                        industry,
                        date,
                        close,
                        pe_ttm,
                        pb,
                        turnover_rate,
                        total_mv_10k,
                        circ_mv_10k,
                        dividend_yield_ttm
                    FROM v_daily_hfq_w_ind_dim
                    WHERE symbol = ?
                    ORDER BY date DESC
                    LIMIT 1
                """
            elif view_name == "v_daily_hfq_w_ind":
                query = """
                    SELECT
                        d.symbol,
                        s.name,
                        s.industry,
                        d.date,
                        d.close,
                        d.pe_ttm,
                        d.pb,
                        d.turnover_rate,
                        d.total_mv_10k,
                        d.circ_mv_10k,
                        d.dividend_yield_ttm
                    FROM v_daily_hfq_w_ind d
                    LEFT JOIN v_dim_symbol s ON d.symbol = s.symbol
                    WHERE d.symbol = ?
                    ORDER BY d.date DESC
                    LIMIT 1
                """
            else:
                query = f"""
                    SELECT
                        b.symbol,
                        s.name,
                        s.industry,
                        b.date,
                        b.close,
                        NULL AS pe_ttm,
                        NULL AS pb,
                        NULL AS turnover_rate,
                        NULL AS total_mv_10k,
                        NULL AS circ_mv_10k,
                        NULL AS dividend_yield_ttm
                    FROM {view_name} b
                    LEFT JOIN v_dim_symbol s ON b.symbol = s.symbol
                    WHERE b.symbol = ?
                    ORDER BY b.date DESC
                    LIMIT 1
                """

            try:
                df = conn.execute(query, [request.symbol_ts]).fetchdf()
            except Exception as exc:
                attempts.append(f"{view_name}=query_error:{exc}")
                continue

            if df.empty:
                attempts.append(f"{view_name}=empty")
                continue

            row = df.iloc[0]
            date_text = str(_to_jsonable(row.get("date")))[:10]
            if len(date_text) != 10:
                attempts.append(f"{view_name}=date_missing")
                continue

            local_date = datetime.strptime(date_text, "%Y-%m-%d").date()
            lag_days = (request.end_date - local_date).days
            if lag_days > request.max_local_lag_days:
                attempts.append(f"{view_name}=local_stale_lag_days={lag_days}")
                continue

            payload = {
                "symbol": row.get("symbol") or request.symbol_ts,
                "name": row.get("name"),
                "industry": row.get("industry"),
                "date": date_text,
                "close": _to_jsonable(row.get("close")),
                "pe_ttm": _to_jsonable(row.get("pe_ttm")),
                "pb": _to_jsonable(row.get("pb")),
                "turnover_rate": _to_jsonable(row.get("turnover_rate")),
                "total_mv_10k": _to_jsonable(row.get("total_mv_10k")),
                "circ_mv_10k": _to_jsonable(row.get("circ_mv_10k")),
                "total_mv": None,
                "circ_mv": None,
                "dividend_yield_ttm": _to_jsonable(row.get("dividend_yield_ttm")),
            }

            total_mv_10k = _as_float(payload.get("total_mv_10k"))
            circ_mv_10k = _as_float(payload.get("circ_mv_10k"))
            if total_mv_10k is not None:
                payload["total_mv"] = total_mv_10k * 10000
            if circ_mv_10k is not None:
                payload["circ_mv"] = circ_mv_10k * 10000

            return {
                "source": "local",
                "symbol": request.symbol_ts,
                "mode": "latest",
                "data": payload,
                "meta": {
                    "local_view": view_name,
                    "adjust_applied": _adjust_from_view(view_name),
                    "local_date": date_text,
                    "lag_days": lag_days,
                    "attempted_views": len(attempts) + 1,
                },
            }, None

    return None, _compose_attempt_reason("local_no_sufficient_latest", attempts)


def _get_akshare(proxy_host: str, token: str, retry: int) -> Any:
    try:
        akshare_proxy_patch = importlib.import_module("akshare_proxy_patch")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "akshare_proxy_patch_missing: install via `.venv/bin/pip install akshare-proxy-patch`"
        ) from exc

    akshare_proxy_patch.install_patch(proxy_host, token, retry=retry)
    try:
        return importlib.import_module("akshare")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "akshare_missing: install via `.venv/bin/pip install akshare`"
        ) from exc


def _fetch_akshare_history(
    request: RequestConfig,
    proxy_host: str,
    token: str,
    retry: int,
    timeout: float,
) -> dict[str, Any]:
    ak = _get_akshare(proxy_host, token, retry)
    adjust = "" if request.adjust in {"", "none", "raw"} else request.adjust

    try:
        df = ak.stock_zh_a_hist(
            symbol=request.symbol_ak,
            period="daily",
            start_date=_date_to_ak(request.start_date),
            end_date=_date_to_ak(request.end_date),
            adjust=adjust,
            timeout=timeout,
        )
    except TypeError:
        df = ak.stock_zh_a_hist(
            symbol=request.symbol_ak,
            period="daily",
            start_date=_date_to_ak(request.start_date),
            end_date=_date_to_ak(request.end_date),
            adjust=adjust,
        )

    if df is None or df.empty:
        raise RuntimeError("akshare_history_empty")

    records: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        records.append(
            {
                "date": str(row.get("日期", "")),
                "open": _to_jsonable(row.get("开盘")),
                "high": _to_jsonable(row.get("最高")),
                "low": _to_jsonable(row.get("最低")),
                "close": _to_jsonable(row.get("收盘")),
                "volume": _to_jsonable(row.get("成交量")),
                "amount": _to_jsonable(row.get("成交额")),
                "turnover_rate": _to_jsonable(row.get("换手率")),
            }
        )

    return {
        "source": "akshare",
        "symbol": request.symbol_ts,
        "mode": "history",
        "data": records,
        "meta": {
            "rows": len(records),
            "start_date": records[0]["date"] if records else None,
            "end_date": records[-1]["date"] if records else None,
        },
    }


def _fetch_akshare_latest(
    request: RequestConfig, proxy_host: str, token: str, retry: int
) -> dict[str, Any]:
    ak = _get_akshare(proxy_host, token, retry)
    spot = ak.stock_zh_a_spot_em()
    if spot is None or spot.empty:
        raise RuntimeError("akshare_spot_empty")

    row_df = spot[spot["代码"] == request.symbol_ak]
    if row_df.empty:
        raise RuntimeError(f"akshare_symbol_not_found:{request.symbol_ak}")

    row = row_df.iloc[0]
    total_mv = _to_jsonable(row.get("总市值"))
    circ_mv = _to_jsonable(row.get("流通市值"))
    total_mv_num = _as_float(total_mv)
    circ_mv_num = _as_float(circ_mv)

    payload = {
        "symbol": request.symbol_ts,
        "name": row.get("名称"),
        "date": date.today().isoformat(),
        "close": _to_jsonable(row.get("最新价")),
        "pe_ttm": _to_jsonable(row.get("市盈率-动态")),
        "pb": _to_jsonable(row.get("市净率")),
        "turnover_rate": _to_jsonable(row.get("换手率")),
        "total_mv": total_mv,
        "circ_mv": circ_mv,
        "total_mv_10k": (total_mv_num / 10000) if total_mv_num is not None else None,
        "circ_mv_10k": (circ_mv_num / 10000) if circ_mv_num is not None else None,
        "pct_change": _to_jsonable(row.get("涨跌幅")),
        "volume": _to_jsonable(row.get("成交量")),
        "amount": _to_jsonable(row.get("成交额")),
    }
    return {
        "source": "akshare",
        "symbol": request.symbol_ts,
        "mode": "latest",
        "data": payload,
        "meta": {"adjust_applied": "none"},
    }


def _emit(payload: dict[str, Any], pretty: bool) -> None:
    indent = 2 if pretty else None
    print(json.dumps(payload, ensure_ascii=False, indent=indent))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch A-share stock data with local DuckDB priority and AkShare fallback."
    )
    parser.add_argument(
        "symbol", help="A-share symbol, e.g. 600519 / 600519.SH / sh600519"
    )
    parser.add_argument("--mode", choices=["history", "latest"], default="history")
    parser.add_argument("--adjust", choices=["hfq", "qfq", "none"], default="hfq")
    parser.add_argument(
        "--period", choices=["1m", "3m", "6m", "1y", "2y", "5y", "max"], default="1y"
    )
    parser.add_argument("--start-date", help="Start date in YYYY-MM-DD")
    parser.add_argument("--end-date", help="End date in YYYY-MM-DD")
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument(
        "--min-rows",
        type=int,
        default=1,
        help="Minimum rows required from local history",
    )
    parser.add_argument(
        "--max-local-lag-days",
        type=int,
        default=7,
        help="Max acceptable lag (days) before local data is considered stale",
    )
    parser.add_argument(
        "--force-remote", action="store_true", help="Skip local DB and force AkShare"
    )
    parser.add_argument("--proxy-host", default=DEFAULT_PROXY_HOST)
    parser.add_argument("--token", default=DEFAULT_TOKEN)
    parser.add_argument("--proxy-retry", type=int, default=30)
    parser.add_argument("--ak-timeout", type=float, default=30.0)
    parser.add_argument("--pretty", action="store_true")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    try:
        request = _build_request(args)
    except Exception as exc:
        _emit({"error": str(exc), "source": None}, pretty=True)
        return 2

    request_payload = {
        "symbol_input": request.symbol,
        "symbol": request.symbol_ts,
        "mode": request.mode,
        "adjust": request.adjust,
        "start_date": request.start_date.isoformat(),
        "end_date": request.end_date.isoformat(),
        "db_path": str(args.db_path),
        "min_rows": request.min_rows,
        "max_local_lag_days": request.max_local_lag_days,
        "ak_timeout": args.ak_timeout,
    }

    local_result: dict[str, Any] | None = None
    fallback_reason: str | None = "force_remote" if args.force_remote else None

    if not args.force_remote:
        try:
            if request.mode == "history":
                local_result, fallback_reason = _fetch_local_history(
                    args.db_path, request
                )
            else:
                local_result, fallback_reason = _fetch_local_latest(
                    args.db_path, request
                )
        except Exception as exc:
            fallback_reason = f"local_query_error:{exc}"

    if local_result is not None:
        local_result["request"] = request_payload
        _emit(local_result, pretty=args.pretty)
        return 0

    try:
        if request.mode == "history":
            remote_result = _fetch_akshare_history(
                request,
                args.proxy_host,
                args.token,
                args.proxy_retry,
                args.ak_timeout,
            )
        else:
            remote_result = _fetch_akshare_latest(
                request, args.proxy_host, args.token, args.proxy_retry
            )

        remote_result["request"] = request_payload
        remote_result.setdefault("meta", {})
        remote_result["meta"]["fallback_reason"] = (
            fallback_reason or "force_remote_or_local_unavailable"
        )
        _emit(remote_result, pretty=args.pretty)
        return 0
    except Exception as exc:
        _emit(
            {
                "error": f"remote_query_error:{exc}",
                "source": None,
                "request": request_payload,
                "meta": {"fallback_reason": fallback_reason},
            },
            pretty=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
