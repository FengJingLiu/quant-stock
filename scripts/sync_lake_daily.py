#!/usr/bin/env python3
"""Incremental sync for stock parquet lake (daily).

Features
- Read incremental files: A股数据_每日指标/增量数据/每日指标/YYYY-MM/YYYYMMDD.csv
- Upsert unadjusted bars to fact_bar_daily/adjust=none (month partition rewrite)
- Upsert indicators to fact_indicator_daily (month partition rewrite)
- Optional: update hfq/qfq factors from daily.zip/hfq/qfq per-code CSV
- Refresh dim_symbol parquet + DuckDB views after sync
- Support dry-run
"""

from __future__ import annotations

import argparse
import shutil
import uuid
import zipfile
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import duckdb
import pandas as pd

try:
    from build_dim_symbol_parquet import build_dim_symbol_parquet
    from init_duckdb_views_lake import init_views
    from prepare_akquant_data import (
        INDICATOR_COLUMN_MAP,
        clean_indicator_frame,
        normalize_stock_code,
        read_csv_bytes,
        read_csv_path,
    )
except ModuleNotFoundError:  # pragma: no cover
    from scripts.build_dim_symbol_parquet import build_dim_symbol_parquet  # type: ignore
    from scripts.init_duckdb_views_lake import init_views  # type: ignore
    from scripts.prepare_akquant_data import (  # type: ignore
        INDICATOR_COLUMN_MAP,
        clean_indicator_frame,
        normalize_stock_code,
        read_csv_bytes,
        read_csv_path,
    )

BAR_COLUMNS = ["date", "symbol", "open", "high", "low", "close", "volume", "amount", "year", "month"]
INDICATOR_VALUE_COLUMNS = [
    col
    for col in INDICATOR_COLUMN_MAP.values()
    if col not in {"symbol", "trade_date", "pct_chg"}
]
INDICATOR_COLUMNS = ["date", "symbol", *INDICATOR_VALUE_COLUMNS, "year", "month"]
FACTOR_COLUMNS = ["date", "symbol", "hfq_factor", "qfq_factor", "year", "month"]

BAR_INPUT_MAP = {
    "股票代码": "symbol",
    "交易日期": "date",
    "开盘价": "open",
    "最高价": "high",
    "最低价": "low",
    "收盘价": "close",
    "成交量(手)": "volume",
    "成交额(千元)": "amount_k",
}


@dataclass
class TargetFile:
    trade_date: date
    path: Path


def _parse_date_token(token: str) -> date:
    t = token.strip()
    if not t:
        raise ValueError("empty date token")

    if len(t) == 8 and t.isdigit():
        return datetime.strptime(t, "%Y%m%d").date()

    return datetime.strptime(t, "%Y-%m-%d").date()


def _date_to_token(d: date) -> str:
    return d.strftime("%Y%m%d")


def collect_incremental_files(incremental_root: Path) -> dict[date, Path]:
    if not incremental_root.exists():
        raise FileNotFoundError(f"missing incremental root: {incremental_root}")

    out: dict[date, Path] = {}
    for path in sorted(incremental_root.glob("*/*.csv")):
        try:
            trade_date = _parse_date_token(path.stem)
        except ValueError:
            continue
        out[trade_date] = path
    return out


def resolve_target_files(
    file_map: dict[date, Path],
    *,
    reprocess_days: int,
    dates: list[date] | None,
) -> list[TargetFile]:
    if not file_map:
        return []

    selected: list[date]
    if dates:
        uniq = sorted(set(dates))
        selected = [d for d in uniq if d in file_map]
        missing = [d for d in uniq if d not in file_map]
        for d in missing:
            print(f"[WARN] increment file not found for date={d}")
    else:
        n = max(1, int(reprocess_days))
        selected = sorted(file_map)[-n:]

    return [TargetFile(trade_date=d, path=file_map[d]) for d in selected]


def clean_incremental_bar_frame(raw_df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in BAR_INPUT_MAP if c not in raw_df.columns]
    if missing:
        raise ValueError(f"increment csv missing columns: {missing}")

    out = raw_df.rename(columns=BAR_INPUT_MAP).copy()
    out = out[list(BAR_INPUT_MAP.values())]

    out["symbol"] = out["symbol"].astype(str).str.strip().str.upper()
    out["date"] = pd.to_datetime(out["date"].astype(str), format="%Y%m%d", errors="coerce")

    for c in ["open", "high", "low", "close", "volume", "amount_k"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out["amount"] = out["amount_k"] * 1000.0
    out = out.drop(columns=["amount_k"])

    out = out.dropna(subset=["date", "symbol", "open", "high", "low", "close"])
    out["year"] = out["date"].dt.year.astype("int32")
    out["month"] = out["date"].dt.month.astype("int8")
    out = out[BAR_COLUMNS].drop_duplicates(subset=["date", "symbol"], keep="last")

    return out.reset_index(drop=True)


def clean_incremental_indicator_frame(raw_df: pd.DataFrame) -> pd.DataFrame:
    cleaned = clean_indicator_frame(raw_df)
    if cleaned.empty:
        return pd.DataFrame(columns=INDICATOR_COLUMNS)

    out = cleaned.copy()
    out["symbol"] = out["symbol"].astype(str).str.strip().str.upper()

    for col in INDICATOR_VALUE_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA

    out["date"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out = out.dropna(subset=["date", "symbol"])
    out["year"] = out["date"].dt.year.astype("int32")
    out["month"] = out["date"].dt.month.astype("int8")

    out = out[["date", "symbol", *INDICATOR_VALUE_COLUMNS, "year", "month"]]
    out = out.drop_duplicates(subset=["date", "symbol"], keep="last")
    return out.reset_index(drop=True)


def _read_month_partition(month_dir: Path, columns: list[str]) -> pd.DataFrame:
    files = sorted(month_dir.glob("*.parquet"))
    if not files:
        return pd.DataFrame(columns=columns)

    con = duckdb.connect()
    try:
        glob = month_dir.as_posix() + "/*.parquet"
        df = con.execute(f"SELECT * FROM read_parquet('{glob}')").df()
    finally:
        con.close()

    for c in columns:
        if c not in df.columns:
            df[c] = pd.NA
    return df[columns]


def _write_single_parquet(df: pd.DataFrame, file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    try:
        con.register("tmp_df", df)
        col_sql = ", ".join(df.columns)
        con.execute(
            f"""
            COPY (SELECT {col_sql} FROM tmp_df)
            TO '{file_path.as_posix()}'
            (FORMAT PARQUET, COMPRESSION ZSTD);
            """
        )
    finally:
        con.close()


def _replace_partition_dir_atomically(month_dir: Path, merged_df: pd.DataFrame) -> None:
    parent = month_dir.parent
    parent.mkdir(parents=True, exist_ok=True)

    tmp_dir = parent / f".{month_dir.name}.tmp-{uuid.uuid4().hex[:8]}"
    bak_dir = parent / f".{month_dir.name}.bak-{uuid.uuid4().hex[:8]}"

    tmp_dir.mkdir(parents=True, exist_ok=True)
    _write_single_parquet(merged_df, tmp_dir / "part-000.parquet")

    has_old = month_dir.exists()
    try:
        if has_old:
            month_dir.rename(bak_dir)
        tmp_dir.rename(month_dir)
        if has_old and bak_dir.exists():
            shutil.rmtree(bak_dir)
    except Exception:  # noqa: BLE001
        if month_dir.exists() and tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        elif tmp_dir.exists() and not month_dir.exists():
            tmp_dir.rename(month_dir)

        if has_old and bak_dir.exists() and not month_dir.exists():
            bak_dir.rename(month_dir)
        raise


def upsert_month_partitions(
    delta_df: pd.DataFrame,
    *,
    base_dir: Path,
    schema_columns: list[str],
    key_columns: list[str],
    dry_run: bool,
) -> dict[str, int]:
    if delta_df.empty:
        return {"months": 0, "delta_rows": 0, "written_rows": 0}

    out_stats = {"months": 0, "delta_rows": int(len(delta_df)), "written_rows": 0}

    grouped = delta_df.groupby(["year", "month"], sort=True)
    for (year, month), delta_m in grouped:
        year_i = int(year)
        month_i = int(month)
        month_dir = base_dir / f"year={year_i}" / f"month={month_i}"

        existing = _read_month_partition(month_dir, schema_columns)
        merged = pd.concat([existing, delta_m[schema_columns]], ignore_index=True)
        merged = merged.drop_duplicates(subset=key_columns, keep="last")
        merged = merged.sort_values(key_columns).reset_index(drop=True)
        merged["year"] = year_i
        merged["month"] = month_i
        merged = merged[schema_columns]

        out_stats["months"] += 1
        out_stats["written_rows"] += int(len(merged))

        if dry_run:
            print(
                "[DRY-RUN] upsert month "
                f"{year_i}-{month_i:02d}: existing={len(existing)}, delta={len(delta_m)}, merged={len(merged)}"
            )
            continue

        _replace_partition_dir_atomically(month_dir, merged)

    return out_stats


def _build_zip_member_map(zf: zipfile.ZipFile) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for name in zf.namelist():
        if not name.lower().endswith(".csv"):
            continue
        stem = Path(name).stem
        raw_code = stem.split("_")[0]
        try:
            code = normalize_stock_code(raw_code)
        except Exception:  # noqa: BLE001
            continue
        mapping[code] = name
    return mapping


def _read_close_series(
    zf: zipfile.ZipFile,
    member: str,
    target_dates: set[date],
) -> dict[date, float]:
    try:
        df = read_csv_bytes(zf.read(member))
    except Exception:  # noqa: BLE001
        return {}

    if df is None or df.empty or "日期" not in df.columns or "收盘" not in df.columns:
        return {}

    out = df[["日期", "收盘"]].copy()
    out["日期"] = pd.to_datetime(out["日期"], errors="coerce")
    out["收盘"] = pd.to_numeric(out["收盘"], errors="coerce")
    out = out.dropna(subset=["日期", "收盘"])
    if out.empty:
        return {}

    out["d"] = out["日期"].dt.date
    out = out[out["d"].isin(target_dates)]

    return {d: float(v) for d, v in zip(out["d"], out["收盘"], strict=False)}


def build_factor_delta_from_zips(
    *,
    raw_zip: Path,
    hfq_zip: Path,
    qfq_zip: Path | None,
    symbols: list[str],
    target_dates: list[date],
) -> pd.DataFrame:
    if not symbols or not target_dates:
        return pd.DataFrame(columns=FACTOR_COLUMNS)

    if not raw_zip.exists() or not hfq_zip.exists():
        raise FileNotFoundError("factor update requires --raw-zip and --hfq-zip")
    if qfq_zip is not None and not qfq_zip.exists():
        raise FileNotFoundError(f"missing qfq zip: {qfq_zip}")

    target_date_set = set(target_dates)
    rows: list[dict[str, object]] = []

    with zipfile.ZipFile(raw_zip) as raw_zf, zipfile.ZipFile(hfq_zip) as hfq_zf:
        qfq_zf = zipfile.ZipFile(qfq_zip) if qfq_zip is not None else None
        try:
            raw_map = _build_zip_member_map(raw_zf)
            hfq_map = _build_zip_member_map(hfq_zf)
            qfq_map = _build_zip_member_map(qfq_zf) if qfq_zf is not None else {}

            for symbol in sorted(set(symbols)):
                try:
                    code = normalize_stock_code(symbol)
                except Exception:  # noqa: BLE001
                    continue

                raw_member = raw_map.get(code)
                hfq_member = hfq_map.get(code)
                if not raw_member or not hfq_member:
                    continue

                raw_close = _read_close_series(raw_zf, raw_member, target_date_set)
                hfq_close = _read_close_series(hfq_zf, hfq_member, target_date_set)
                qfq_close = {}
                qfq_member = qfq_map.get(code)
                if qfq_zf is not None and qfq_member:
                    qfq_close = _read_close_series(qfq_zf, qfq_member, target_date_set)

                for d in sorted(target_date_set):
                    base = raw_close.get(d)
                    if base is None or base <= 0:
                        continue

                    hfq_val = hfq_close.get(d)
                    qfq_val = qfq_close.get(d)
                    hfq_factor = (hfq_val / base) if hfq_val is not None else pd.NA
                    qfq_factor = (qfq_val / base) if qfq_val is not None else pd.NA

                    if pd.isna(hfq_factor) and pd.isna(qfq_factor):
                        continue

                    rows.append(
                        {
                            "date": pd.Timestamp(d),
                            "symbol": symbol,
                            "hfq_factor": hfq_factor,
                            "qfq_factor": qfq_factor,
                            "year": d.year,
                            "month": d.month,
                        }
                    )
        finally:
            if qfq_zf is not None:
                qfq_zf.close()

    if not rows:
        return pd.DataFrame(columns=FACTOR_COLUMNS)

    out = pd.DataFrame(rows)
    out = out.drop_duplicates(subset=["date", "symbol"], keep="last")
    return out[FACTOR_COLUMNS].reset_index(drop=True)


def sync_lake_daily(
    *,
    incremental_root: Path,
    reprocess_days: int,
    dates: list[date] | None,
    raw_base_dir: Path,
    indicator_base_dir: Path,
    factor_base_dir: Path,
    update_factors: bool,
    raw_zip: Path,
    hfq_zip: Path,
    qfq_zip: Path | None,
    stock_list: Path,
    delisted_stock_list: Path,
    dim_out_file: Path,
    db_path: Path,
    raw_glob: str,
    factor_glob: str,
    indicator_glob: str,
    dim_glob: str,
    dry_run: bool,
) -> dict[str, object]:
    file_map = collect_incremental_files(incremental_root)
    targets = resolve_target_files(file_map, reprocess_days=reprocess_days, dates=dates)
    if not targets:
        raise RuntimeError("no incremental files selected")

    print(
        "[INFO] target increment dates: "
        + ", ".join(f"{t.trade_date} ({t.path.name})" for t in targets)
    )

    bar_frames: list[pd.DataFrame] = []
    ind_frames: list[pd.DataFrame] = []

    for t in targets:
        raw = read_csv_path(t.path)
        bar_frames.append(clean_incremental_bar_frame(raw))
        ind_frames.append(clean_incremental_indicator_frame(raw))

    bar_delta = pd.concat(bar_frames, ignore_index=True) if bar_frames else pd.DataFrame(columns=BAR_COLUMNS)
    ind_delta = pd.concat(ind_frames, ignore_index=True) if ind_frames else pd.DataFrame(columns=INDICATOR_COLUMNS)

    bar_delta = bar_delta.drop_duplicates(subset=["date", "symbol"], keep="last")
    ind_delta = ind_delta.drop_duplicates(subset=["date", "symbol"], keep="last")

    bar_stats = upsert_month_partitions(
        bar_delta,
        base_dir=raw_base_dir,
        schema_columns=BAR_COLUMNS,
        key_columns=["date", "symbol"],
        dry_run=dry_run,
    )
    ind_stats = upsert_month_partitions(
        ind_delta,
        base_dir=indicator_base_dir,
        schema_columns=INDICATOR_COLUMNS,
        key_columns=["date", "symbol"],
        dry_run=dry_run,
    )

    factor_stats = {"months": 0, "delta_rows": 0, "written_rows": 0}
    if update_factors:
        target_dates = [t.trade_date for t in targets]
        symbols = sorted(set(bar_delta["symbol"].astype(str).tolist()))
        factor_delta = build_factor_delta_from_zips(
            raw_zip=raw_zip,
            hfq_zip=hfq_zip,
            qfq_zip=qfq_zip,
            symbols=symbols,
            target_dates=target_dates,
        )
        factor_stats = upsert_month_partitions(
            factor_delta,
            base_dir=factor_base_dir,
            schema_columns=FACTOR_COLUMNS,
            key_columns=["date", "symbol"],
            dry_run=dry_run,
        )
    else:
        print("[INFO] skip factor update (use --update-factors to enable)")

    dim_stats: dict[str, int] = {}
    if dry_run:
        print("[DRY-RUN] skip dim_symbol rebuild + duckdb view refresh")
    else:
        dim_stats = build_dim_symbol_parquet(
            stock_list=stock_list,
            delisted_stock_list=delisted_stock_list,
            out_file=dim_out_file,
            dry_run=False,
        )
        init_views(
            db_path=db_path,
            raw_glob=raw_glob,
            factor_glob=factor_glob,
            indicator_glob=indicator_glob,
            dim_glob=dim_glob,
        )

    return {
        "target_files": len(targets),
        "target_dates": [_date_to_token(t.trade_date) for t in targets],
        "bar_delta_rows": int(len(bar_delta)),
        "indicator_delta_rows": int(len(ind_delta)),
        "bar_stats": bar_stats,
        "indicator_stats": ind_stats,
        "factor_stats": factor_stats,
        "dim_stats": dim_stats,
        "dry_run": dry_run,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Incremental sync to lake (daily)")
    p.add_argument(
        "--incremental-root",
        type=Path,
        default=Path("A股数据_每日指标/增量数据/每日指标"),
        help="增量数据根目录",
    )
    p.add_argument(
        "--reprocess-days",
        type=int,
        default=3,
        help="默认回刷最近 N 个交易日（按增量文件日期排序）",
    )
    p.add_argument(
        "--dates",
        nargs="*",
        default=None,
        help="指定日期列表（YYYYMMDD 或 YYYY-MM-DD；支持逗号分隔）",
    )

    p.add_argument("--raw-base-dir", type=Path, default=Path("data/lake/fact_bar_daily/adjust=none"))
    p.add_argument("--indicator-base-dir", type=Path, default=Path("data/lake/fact_indicator_daily"))
    p.add_argument("--factor-base-dir", type=Path, default=Path("data/lake/adj_factor_daily"))

    p.add_argument(
        "--update-factors",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="是否从 daily.zip/hfq/qfq 增量更新 hfq_factor/qfq_factor",
    )
    p.add_argument("--raw-zip", type=Path, default=Path("A股数据_zip/daily.zip"))
    p.add_argument("--hfq-zip", type=Path, default=Path("A股数据_zip/daily_hfq.zip"))
    p.add_argument("--qfq-zip", type=Path, default=Path("A股数据_zip/daily_qfq.zip"))

    p.add_argument("--stock-list", type=Path, default=Path("A股数据_每日指标/股票列表.csv"))
    p.add_argument("--delisted-stock-list", type=Path, default=Path("A股数据_每日指标/退市股票列表.csv"))
    p.add_argument("--dim-out-file", type=Path, default=Path("data/lake/dim_symbol/symbols.parquet"))

    p.add_argument("--db", type=Path, default=Path("data/duckdb/stock.duckdb"))
    p.add_argument(
        "--raw-glob",
        type=str,
        default="data/lake/fact_bar_daily/adjust=none/**/*.parquet",
    )
    p.add_argument(
        "--factor-glob",
        type=str,
        default="data/lake/adj_factor_daily/**/*.parquet",
    )
    p.add_argument(
        "--indicator-glob",
        type=str,
        default="data/lake/fact_indicator_daily/**/*.parquet",
    )
    p.add_argument(
        "--dim-glob",
        type=str,
        default="data/lake/dim_symbol/**/*.parquet",
    )

    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _parse_dates_arg(raw_dates: list[str] | None) -> list[date] | None:
    if not raw_dates:
        return None

    tokens: list[str] = []
    for v in raw_dates:
        tokens.extend([t.strip() for t in str(v).split(",") if t.strip()])

    if not tokens:
        return None

    return [_parse_date_token(t) for t in tokens]


def main() -> None:
    args = parse_args()
    dates = _parse_dates_arg(args.dates)

    summary = sync_lake_daily(
        incremental_root=args.incremental_root,
        reprocess_days=args.reprocess_days,
        dates=dates,
        raw_base_dir=args.raw_base_dir,
        indicator_base_dir=args.indicator_base_dir,
        factor_base_dir=args.factor_base_dir,
        update_factors=bool(args.update_factors),
        raw_zip=args.raw_zip,
        hfq_zip=args.hfq_zip,
        qfq_zip=args.qfq_zip,
        stock_list=args.stock_list,
        delisted_stock_list=args.delisted_stock_list,
        dim_out_file=args.dim_out_file,
        db_path=args.db,
        raw_glob=args.raw_glob,
        factor_glob=args.factor_glob,
        indicator_glob=args.indicator_glob,
        dim_glob=args.dim_glob,
        dry_run=args.dry_run,
    )

    print("[DONE] sync summary:")
    for k, v in summary.items():
        print(f"  - {k}: {v}")


if __name__ == "__main__":
    main()
