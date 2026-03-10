#!/usr/bin/env python3
"""Build symbol dimension parquet from active + delisted stock lists.

Input
- A股数据_每日指标/股票列表.csv
- A股数据_每日指标/退市股票列表.csv

Output
- data/lake/dim_symbol/symbols.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import pandas as pd

try:
    from prepare_akquant_data import normalize_stock_code, read_csv_path
except ModuleNotFoundError:  # pragma: no cover
    from scripts.prepare_akquant_data import normalize_stock_code, read_csv_path  # type: ignore

DIM_COLUMNS = [
    "symbol",
    "code",
    "name",
    "region",
    "area",
    "industry",
    "full_name",
    "english_name",
    "pinyin_abbr",
    "market_type",
    "market",
    "exchange",
    "list_date",
    "controller_name",
    "controller_type",
    "is_delisted",
    "updated_at",
]

DIM_COL_MAP = {
    "TS代码": "symbol",
    "股票代码": "code",
    "股票名称": "name",
    "地域": "region",
    "所属行业": "industry",
    "股票全称": "full_name",
    "英文全称": "english_name",
    "拼音缩写": "pinyin_abbr",
    "市场类型": "market_type",
    "交易所代码": "exchange",
    "上市日期": "list_date",
    "实控人名称": "controller_name",
    "实控人企业性质": "controller_type",
}


def _standardize_symbol(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    s = str(value).strip().upper()
    if not s:
        return None
    if "." in s:
        return s
    try:
        return f"{normalize_stock_code(s)}.UNKNOWN"
    except Exception:  # noqa: BLE001
        return s


def _standardize_code(row: pd.Series) -> str | None:
    raw_code = row.get("code")
    if raw_code is not None and not pd.isna(raw_code):
        try:
            return normalize_stock_code(raw_code)
        except Exception:  # noqa: BLE001
            pass

    symbol = row.get("symbol")
    if symbol is None or pd.isna(symbol):
        return None
    try:
        return normalize_stock_code(symbol)
    except Exception:  # noqa: BLE001
        return None


def _read_dim_source(path: Path, *, is_delisted: int, source_rank: int) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"missing csv: {path}")

    df = read_csv_path(path)
    out = df.rename(columns=DIM_COL_MAP).copy()

    out["area"] = out.get("region")
    out["market"] = out.get("market_type")

    for col in DIM_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA

    out = out[DIM_COLUMNS].copy()

    out["symbol"] = out["symbol"].map(_standardize_symbol)
    out["code"] = out.apply(_standardize_code, axis=1)

    if "exchange" in out.columns:
        out["exchange"] = out["exchange"].astype(str).str.strip().str.upper()

    out["list_date"] = pd.to_datetime(out["list_date"].astype(str), format="%Y%m%d", errors="coerce")
    out["is_delisted"] = int(is_delisted)
    out["source_rank"] = int(source_rank)
    out["updated_at"] = pd.Timestamp.now("UTC").floor("s")

    out = out.dropna(subset=["symbol", "code"])
    out = out.drop_duplicates(subset=["symbol"], keep="last")
    return out


def build_dim_symbol_parquet(
    stock_list: Path,
    delisted_stock_list: Path,
    out_file: Path,
    *,
    dry_run: bool = False,
) -> dict[str, int]:
    active = _read_dim_source(stock_list, is_delisted=0, source_rank=2)
    delisted = _read_dim_source(delisted_stock_list, is_delisted=1, source_rank=1)

    merged = pd.concat([delisted, active], ignore_index=True)
    merged = merged.sort_values(["symbol", "source_rank"]).drop_duplicates(subset=["symbol"], keep="last")

    for col in DIM_COLUMNS:
        if col not in merged.columns:
            merged[col] = pd.NA

    merged = merged[DIM_COLUMNS].sort_values("symbol").reset_index(drop=True)

    stats = {
        "rows": int(len(merged)),
        "active_rows": int(len(active)),
        "delisted_rows": int(len(delisted)),
    }

    if dry_run:
        print(
            "[DRY-RUN] dim_symbol build: "
            f"rows={stats['rows']}, active={stats['active_rows']}, delisted={stats['delisted_rows']}"
        )
        return stats

    out_file.parent.mkdir(parents=True, exist_ok=True)
    if out_file.exists():
        out_file.unlink()

    con = duckdb.connect()
    try:
        con.register("dim_df", merged)
        con.execute(
            f"""
            COPY (
              SELECT
                symbol,
                code,
                name,
                region,
                area,
                industry,
                full_name,
                english_name,
                pinyin_abbr,
                market_type,
                market,
                exchange,
                list_date::DATE AS list_date,
                controller_name,
                controller_type,
                is_delisted,
                updated_at::TIMESTAMP AS updated_at
              FROM dim_df
            )
            TO '{out_file.as_posix()}'
            (FORMAT PARQUET, COMPRESSION ZSTD);
            """
        )
    finally:
        con.close()

    print(f"[DONE] wrote dim_symbol parquet -> {out_file} (rows={stats['rows']})")
    return stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build dim_symbol parquet")
    p.add_argument("--stock-list", type=Path, default=Path("A股数据_每日指标/股票列表.csv"))
    p.add_argument("--delisted-stock-list", type=Path, default=Path("A股数据_每日指标/退市股票列表.csv"))
    p.add_argument("--out-file", type=Path, default=Path("data/lake/dim_symbol/symbols.parquet"))
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    build_dim_symbol_parquet(
        stock_list=args.stock_list,
        delisted_stock_list=args.delisted_stock_list,
        out_file=args.out_file,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
