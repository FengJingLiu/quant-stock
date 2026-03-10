"""
本地数据访问模块 (Local Data Access)
优先从本地 DuckDB / Parquet Lake 获取数据，获取不到再使用 AKShare。

数据路径：project_root/data/duckdb/stock.duckdb
"""

import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

# 定位项目根目录和 DuckDB 文件
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
DUCKDB_PATH = PROJECT_ROOT / "data" / "duckdb" / "stock.duckdb"


def _normalize_symbol(symbol: str) -> str:
    """标准化 A 股代码为 TS 格式 (如 600519 -> 600519.SH)。"""
    sym = symbol.strip().replace(".SH", "").replace(".SZ", "").replace(".BJ", "")
    sym = sym.zfill(6)

    # 根据前缀判断交易所
    if sym.startswith("6"):
        return f"{sym}.SH"
    elif sym.startswith(("0", "3")):
        return f"{sym}.SZ"
    elif sym.startswith(("4", "8")):
        return f"{sym}.BJ"
    return f"{sym}.SZ"  # 默认深圳


def _normalize_symbol_list(symbols: list[str]) -> list[str]:
    """标准化多个 A 股代码。"""
    return [_normalize_symbol(s) for s in symbols]


def get_price_history(
    symbol: str,
    period: str = "1y",
    adjust: str = "qfq",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Optional[list[dict]]:
    """
    从本地 Lake 获取历史价格数据。

    Args:
        symbol: A 股代码 (如 "600519" 或 "600519.SH")
        period: 时间周期 ("1m", "3m", "6m", "1y", "2y", "5y", "max")
        adjust: 复权方式 ("qfq" - 前复权, "hfq" - 后复权, "" - 不复权)
        start_date: 开始日期 (YYYYMMDD)，覆盖 period
        end_date: 结束日期 (YYYYMMDD)，覆盖 period

    Returns:
        价格数据列表，失败返回 None
    """
    if not DUCKDB_PATH.exists():
        return None

    import duckdb

    sym = _normalize_symbol(symbol)

    # 计算日期范围
    if start_date and end_date:
        start = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]}"
        end = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:]}"
    else:
        period_map = {
            "1m": 30,
            "3m": 90,
            "6m": 180,
            "1y": 365,
            "2y": 730,
            "5y": 1825,
            "max": 7300,
        }
        days = period_map.get(period, 365)
        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    original_dir = os.getcwd()
    conn = None
    try:
        os.chdir(PROJECT_ROOT)
        conn = duckdb.connect(str(DUCKDB_PATH), read_only=True)

        # 选择表/视图
        if adjust == "hfq":
            query = f"""
                SELECT date, open, high, low, close, volume, amount
                FROM v_bar_daily_hfq
                WHERE symbol = ?
                  AND date BETWEEN ? AND ?
                ORDER BY date
            """
        elif adjust == "qfq":
            # 使用 v_bar_daily_qfq 视图（如果存在），否则计算
            query = f"""
                SELECT date, open, high, low, close, volume, amount
                FROM v_bar_daily_qfq
                WHERE symbol = ?
                  AND date BETWEEN ? AND ?
                ORDER BY date
            """
        else:
            query = f"""
                SELECT date, open, high, low, close, volume, amount
                FROM v_bar_daily_raw
                WHERE symbol = ?
                  AND date BETWEEN ? AND ?
                ORDER BY date
            """
        df = conn.execute(query, [sym, start, end]).fetchdf()
        conn.close()

        return _df_to_records(df) if not df.empty else None

    except Exception:
        return None
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass
        os.chdir(original_dir)


def get_financial_metrics(symbol: str) -> Optional[dict]:
    """
    从本地 Lake 获取最新财务指标。

    Returns:
        财务指标字典，失败返回 None
    """
    if not DUCKDB_PATH.exists():
        return None

    import duckdb

    sym = _normalize_symbol(symbol)

    original_dir = os.getcwd()
    conn = None
    try:
        os.chdir(PROJECT_ROOT)
        conn = duckdb.connect(str(DUCKDB_PATH), read_only=True)

        # 获取最新日期的指标
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
            FROM v_daily_hfq_w_ind_dim d
            LEFT JOIN v_dim_symbol s ON d.symbol = s.symbol
            WHERE d.symbol = ?
            ORDER BY d.date DESC
            LIMIT 1
        """
        df = conn.execute(query, [sym]).fetchdf()
        conn.close()

        if df.empty:
            return None

        row = df.iloc[0]
        return {
            "symbol": row["symbol"],
            "name": row.get("name", ""),
            "industry": row.get("industry", ""),
            "current_price": row.get("close"),
            "market_cap": row.get("total_mv_10k"),
            "circulating_cap": row.get("circ_mv_10k"),
            "valuation": {
                "pe_ttm": row.get("pe_ttm"),
                "pb": row.get("pb"),
            },
            "trading": {
                "turnover_rate": row.get("turnover_rate"),
            },
            "data_date": str(row.get("date", ""))[:10] if row.get("date") else "",
            "source": "local",
        }

    except Exception:
        return None
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass
        os.chdir(original_dir)


def get_basic_info(symbols: list[str]) -> Optional[list[dict]]:
    """
    从本地 Lake 获取股票基本信息。

    Returns:
        基本信息列表，失败返回 None
    """
    if not DUCKDB_PATH.exists():
        return None

    if not symbols:
        return None

    import duckdb

    syms = _normalize_symbol_list(symbols)

    original_dir = os.getcwd()
    conn = None
    try:
        os.chdir(PROJECT_ROOT)
        conn = duckdb.connect(str(DUCKDB_PATH), read_only=True)

        # 使用 IN 查询
        placeholders = ",".join(["?"] * len(syms))
        query = f"""
            SELECT
                symbol,
                name,
                industry,
                exchange,
                market_type,
                list_date
            FROM v_dim_symbol
            WHERE symbol IN ({placeholders})
        """
        df = conn.execute(query, syms).fetchdf()
        conn.close()

        if df.empty:
            return None

        results = []
        for _, row in df.iterrows():
            results.append(
                {
                    "symbol": row["symbol"],
                    "name": row.get("name", ""),
                    "industry": row.get("industry", ""),
                    "exchange": row.get("exchange", ""),
                    "market_type": row.get("market_type", ""),
                    "listing_date": str(row.get("list_date", ""))[:10]
                    if row.get("list_date")
                    else "",
                    "source": "local",
                }
            )

        return results

    except Exception:
        return None
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass
        os.chdir(original_dir)


def get_latest_date() -> Optional[str]:
    """获取本地数据的最新日期。"""
    if not DUCKDB_PATH.exists():
        return None

    import duckdb

    original_dir = os.getcwd()
    conn = None
    try:
        os.chdir(PROJECT_ROOT)
        conn = duckdb.connect(str(DUCKDB_PATH), read_only=True)
        query = "SELECT MAX(date) as max_date FROM v_bar_daily_raw"
        result = conn.execute(query).fetchone()
        conn.close()

        if result and result[0]:
            return str(result[0])[:10]
        return None
    except Exception:
        return None
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass
        os.chdir(original_dir)


def _df_to_records(df) -> list[dict]:
    """将 DataFrame 转换为记录列表。"""
    records = []
    for _, row in df.iterrows():
        record = {}
        for col in df.columns:
            val = row[col]
            if col == "date":
                record[col] = str(val)[:10]
            elif col in ["volume", "amount"]:
                record[col] = float(val) if val is not None else None
            else:
                record[col] = float(val) if val is not None else None
        records.append(record)
    return records


def is_local_data_available() -> bool:
    """检查本地数据是否可用。"""
    return DUCKDB_PATH.exists()


if __name__ == "__main__":
    # 测试代码
    print(f"DuckDB path: {DUCKDB_PATH}")
    print(f"Available: {is_local_data_available()}")

    if is_local_data_available():
        # 测试获取最新日期
        latest = get_latest_date()
        print(f"Latest date in local DB: {latest}")

        # 测试获取财务指标
        metrics = get_financial_metrics("600519")
        print(f"Metrics for 600519: {metrics}")
