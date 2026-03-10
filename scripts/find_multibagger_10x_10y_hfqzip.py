#!/usr/bin/env python3
"""Find A-share multibaggers from local hfq zip (buy&hold over N years).

Definition (default)
- end_date: max date inferred from incremental dir, unless overridden by --end-date
- start_date: end_date - N years (default N=10)
- entry: first trading day close_hfq in [start_date, end_date]
- exit: last trading day close_hfq in [start_date, end_date]
- multiple = exit / entry

Data source
- A股数据_zip/daily_hfq.zip (per-code CSV)
- data/lake/dim_symbol/symbols.parquet (auto-built if missing)
"""

from __future__ import annotations

import argparse
import csv
import io
import zipfile
from datetime import date, datetime
from pathlib import Path

import duckdb
import pandas as pd

try:
    from build_dim_symbol_parquet import build_dim_symbol_parquet
    from prepare_akquant_data import normalize_stock_code
except ModuleNotFoundError:  # pragma: no cover
    from scripts.build_dim_symbol_parquet import build_dim_symbol_parquet  # type: ignore
    from scripts.prepare_akquant_data import normalize_stock_code  # type: ignore

ENCODINGS = ("utf-8-sig", "gb18030", "gbk")
DATE_COL_CANDIDATES = ("日期", "date", "Date")
CLOSE_COL_CANDIDATES = ("收盘", "close", "Close")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Find 10x+ multibaggers from daily_hfq.zip")
    p.add_argument("--hfq-zip", type=Path, default=Path("A股数据_zip/daily_hfq.zip"))
    p.add_argument(
        "--incremental-root",
        type=Path,
        default=Path("A股数据_每日指标/增量数据/每日指标"),
        help="Used to infer end_date from max YYYYMMDD.csv",
    )
    p.add_argument("--end-date", type=str, default=None, help="Override end date (YYYYMMDD or YYYY-MM-DD)")

    p.add_argument("--years", type=int, default=10)
    p.add_argument("--multiple", type=float, default=10.0)

    p.add_argument("--dim-file", type=Path, default=Path("data/lake/dim_symbol/symbols.parquet"))
    p.add_argument("--stock-list", type=Path, default=Path("A股数据_每日指标/股票列表.csv"))
    p.add_argument("--delisted-stock-list", type=Path, default=Path("A股数据_每日指标/退市股票列表.csv"))

    p.add_argument("--out-list", type=Path, default=Path("data/multibagger_10x_10y_list.csv"))
    p.add_argument("--out-summary", type=Path, default=Path("data/multibagger_10x_10y_summary.csv"))
    p.add_argument("--top-samples", type=int, default=10, help="Top samples kept in summary sample list")
    p.add_argument("--progress-every", type=int, default=500)
    return p.parse_args()


def _parse_date_token(token: str) -> date:
    s = str(token).strip()
    if not s:
        raise ValueError("empty date token")
    if len(s) == 8 and s.isdigit():
        return datetime.strptime(s, "%Y%m%d").date()
    return datetime.strptime(s, "%Y-%m-%d").date()


def _to_ymd(d: date) -> str:
    return d.strftime("%Y-%m-%d")


def infer_end_date_from_incremental(incremental_root: Path) -> date:
    if not incremental_root.exists():
        raise FileNotFoundError(f"incremental root not found: {incremental_root}")

    dates: list[date] = []
    for p in incremental_root.glob("*/*.csv"):
        stem = p.stem.strip()
        if len(stem) == 8 and stem.isdigit():
            dates.append(datetime.strptime(stem, "%Y%m%d").date())

    if not dates:
        raise RuntimeError(f"no YYYYMMDD.csv found under: {incremental_root}")

    return max(dates)


def classify_board(code: str) -> str:
    c = str(code).strip()
    if c.startswith("68"):
        return "科创板"
    if c.startswith(("60", "601", "603", "605")):
        return "沪主板"
    if c.startswith("30"):
        return "创业板"
    if c.startswith("00"):
        return "深主板"
    if c.startswith(("83", "87", "88", "43", "92")):
        return "北交/新三板"
    return "其他"


def _normalize_date_str(raw: str) -> str | None:
    if raw is None:
        return None
    s = str(raw).strip().replace("/", "-")
    if len(s) >= 10:
        s = s[:10]
    try:
        datetime.strptime(s, "%Y-%m-%d")
        return s
    except Exception:  # noqa: BLE001
        return None


def _safe_float(x: str) -> float | None:
    try:
        v = float(str(x).replace(",", "").strip())
    except Exception:  # noqa: BLE001
        return None
    if pd.isna(v):
        return None
    return v


def _extract_code_from_member(member: str) -> str | None:
    stem = Path(member).stem
    raw_code = stem.split("_")[0]
    try:
        return normalize_stock_code(raw_code)
    except Exception:  # noqa: BLE001
        return None


def _resolve_colname(keys: list[str], candidates: tuple[str, ...]) -> str | None:
    norm = {str(k).strip().lstrip("\ufeff"): k for k in keys}
    for c in candidates:
        if c in norm:
            return norm[c]
    return None


def scan_member_entry_exit(
    zf: zipfile.ZipFile,
    member: str,
    *,
    start_date_s: str,
    end_date_s: str,
) -> tuple[str, float, str, float] | None:
    for enc in ENCODINGS:
        try:
            with zf.open(member, "r") as f:
                wrapper = io.TextIOWrapper(f, encoding=enc, newline="", errors="strict")
                reader = csv.DictReader(wrapper)

                if not reader.fieldnames:
                    return None

                date_col = _resolve_colname(list(reader.fieldnames), DATE_COL_CANDIDATES)
                close_col = _resolve_colname(list(reader.fieldnames), CLOSE_COL_CANDIDATES)
                if not date_col or not close_col:
                    return None

                entry_date: str | None = None
                entry_close: float | None = None
                exit_date: str | None = None
                exit_close: float | None = None

                for row in reader:
                    d_raw = row.get(date_col)
                    c_raw = row.get(close_col)
                    d = _normalize_date_str(d_raw)
                    if d is None or d < start_date_s or d > end_date_s:
                        continue

                    c = _safe_float(c_raw)
                    if c is None or c <= 0:
                        continue

                    if entry_date is None or d < entry_date or (d == entry_date):
                        entry_date = d
                        entry_close = c
                    if exit_date is None or d > exit_date or (d == exit_date):
                        exit_date = d
                        exit_close = c

                if entry_date and exit_date and entry_close and exit_close:
                    return entry_date, entry_close, exit_date, exit_close
                return None
        except UnicodeDecodeError:
            continue

    return None


def ensure_dim_symbol(
    dim_file: Path,
    stock_list: Path,
    delisted_stock_list: Path,
) -> None:
    if dim_file.exists():
        return
    print(f"[INFO] dim file missing, building: {dim_file}")
    build_dim_symbol_parquet(
        stock_list=stock_list,
        delisted_stock_list=delisted_stock_list,
        out_file=dim_file,
        dry_run=False,
    )


def load_dim_map(dim_file: Path) -> dict[str, dict[str, object]]:
    con = duckdb.connect()
    try:
        df = con.execute(
            f"""
            SELECT
              symbol,
              code,
              name,
              industry,
              exchange,
              COALESCE(market_type, market) AS market_type,
              is_delisted,
              COALESCE(region, area) AS region
            FROM read_parquet('{dim_file.as_posix()}')
            """
        ).df()
    finally:
        con.close()

    out: dict[str, dict[str, object]] = {}
    for _, r in df.iterrows():
        code_raw = r.get("code")
        if pd.isna(code_raw):
            continue
        try:
            code = normalize_stock_code(code_raw)
        except Exception:  # noqa: BLE001
            continue

        out[code] = {
            "symbol": None if pd.isna(r.get("symbol")) else str(r.get("symbol")),
            "name": "" if pd.isna(r.get("name")) else str(r.get("name")),
            "industry": "" if pd.isna(r.get("industry")) else str(r.get("industry")),
            "exchange": "" if pd.isna(r.get("exchange")) else str(r.get("exchange")),
            "market_type": "" if pd.isna(r.get("market_type")) else str(r.get("market_type")),
            "is_delisted": None if pd.isna(r.get("is_delisted")) else int(r.get("is_delisted")),
            "region": "" if pd.isna(r.get("region")) else str(r.get("region")),
        }

    return out


def summarize_hits(df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["board", "industry", "count", "top_samples"])

    def sample_list(g: pd.DataFrame) -> str:
        show = g.sort_values("multiple", ascending=False).head(top_k)
        return ";".join(
            f"{r['symbol']}{r['name']}({float(r['multiple']):.2f}x)"
            for _, r in show.iterrows()
        )

    grp = (
        df.groupby(["board", "industry"], dropna=False)
        .apply(lambda g: pd.Series({"count": len(g), "top_samples": sample_list(g)}))
        .reset_index()
        .sort_values(["count", "board", "industry"], ascending=[False, True, True])
        .reset_index(drop=True)
    )
    return grp


def main() -> None:
    args = parse_args()

    if not args.hfq_zip.exists():
        raise FileNotFoundError(f"hfq zip not found: {args.hfq_zip}")

    if args.end_date:
        end_date = _parse_date_token(args.end_date)
    else:
        end_date = infer_end_date_from_incremental(args.incremental_root)

    start_date = (pd.Timestamp(end_date) - pd.DateOffset(years=int(args.years))).date()
    start_date_s = _to_ymd(start_date)
    end_date_s = _to_ymd(end_date)

    ensure_dim_symbol(
        dim_file=args.dim_file,
        stock_list=args.stock_list,
        delisted_stock_list=args.delisted_stock_list,
    )
    dim_map = load_dim_map(args.dim_file)

    print(
        f"[INFO] window: {start_date_s} -> {end_date_s}; years={args.years}; multiple>={args.multiple}; dim={len(dim_map)}"
    )

    hits: list[dict[str, object]] = []

    with zipfile.ZipFile(args.hfq_zip) as zf:
        members = sorted([m for m in zf.namelist() if m.lower().endswith(".csv")])
        total = len(members)
        print(f"[INFO] scanning hfq zip members: {total}")

        for i, member in enumerate(members, start=1):
            code = _extract_code_from_member(member)
            if not code:
                continue

            entry_exit = scan_member_entry_exit(
                zf,
                member,
                start_date_s=start_date_s,
                end_date_s=end_date_s,
            )
            if entry_exit is None:
                continue

            entry_date, entry_close, exit_date, exit_close = entry_exit
            if entry_close <= 0:
                continue
            multiple = float(exit_close / entry_close)
            if multiple < float(args.multiple):
                continue

            dim = dim_map.get(code, {})
            symbol = str(dim.get("symbol") or f"{code}.UNKNOWN")
            rec = {
                "symbol": symbol,
                "code": code,
                "name": str(dim.get("name") or ""),
                "board": classify_board(code),
                "industry": str(dim.get("industry") or ""),
                "start_date": entry_date,
                "start_close_hfq": float(entry_close),
                "end_date": exit_date,
                "end_close_hfq": float(exit_close),
                "multiple": multiple,
                # extra dim fields (for downstream convenience)
                "exchange": str(dim.get("exchange") or ""),
                "market_type": str(dim.get("market_type") or ""),
                "is_delisted": dim.get("is_delisted"),
                "region": str(dim.get("region") or ""),
            }
            hits.append(rec)

            if i % max(1, int(args.progress_every)) == 0 or i == total:
                print(f"[INFO] progress: {i}/{total}, hits={len(hits)}")

    list_df = pd.DataFrame(hits)
    if not list_df.empty:
        list_df = list_df.sort_values(["multiple", "symbol"], ascending=[False, True]).reset_index(drop=True)

    # Keep required output columns first (plus extra dim fields afterwards)
    base_cols = [
        "symbol",
        "code",
        "name",
        "board",
        "industry",
        "start_date",
        "start_close_hfq",
        "end_date",
        "end_close_hfq",
        "multiple",
    ]
    extra_cols = ["exchange", "market_type", "is_delisted", "region"]
    for c in base_cols + extra_cols:
        if c not in list_df.columns:
            list_df[c] = pd.NA

    args.out_list.parent.mkdir(parents=True, exist_ok=True)
    list_df[base_cols + extra_cols].to_csv(args.out_list, index=False, encoding="utf-8-sig")

    summary_df = summarize_hits(list_df, top_k=max(1, int(args.top_samples)))
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(args.out_summary, index=False, encoding="utf-8-sig")

    print(f"[DONE] hits={len(list_df)}")
    print(f"[DONE] wrote list: {args.out_list}")
    print(f"[DONE] wrote summary: {args.out_summary}")

    if not list_df.empty:
        print("\n[TOP20] by multiple")
        show = list_df.head(20)
        for _, r in show.iterrows():
            print(
                f"{r['symbol']} {r['name']} | {r['board']} | {r['industry']} | "
                f"{float(r['multiple']):.2f}x | {r['start_date']}->{r['end_date']}"
            )


if __name__ == "__main__":
    main()
