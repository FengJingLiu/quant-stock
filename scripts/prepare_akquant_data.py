#!/usr/bin/env python3
"""A 股数据解压与清洗脚本，输出可用于 Akquant 回测的数据集。"""

from __future__ import annotations

import argparse
import io
import re
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

PRICE_COLUMN_MAP = {
    "日期": "timestamp",
    "开盘": "open",
    "最高": "high",
    "最低": "low",
    "收盘": "close",
    "成交量": "volume",
    "成交额": "amount",
    "涨跌幅": "pct_chg",
    "换手率": "turnover_rate",
}

INDICATOR_COLUMN_MAP = {
    "股票代码": "symbol",
    "交易日期": "trade_date",
    "换手率": "turnover_rate",
    "换手率(自由流通股)": "turnover_rate_free",
    "量比": "volume_ratio",
    "市盈率": "pe",
    "市盈率TTM": "pe_ttm",
    "市净率": "pb",
    "市销率": "ps",
    "市销率TTM": "ps_ttm",
    "股息率": "dividend_yield",
    "股息率TTM": "dividend_yield_ttm",
    "总股本(万股)": "total_share_10k",
    "流通股本(万股)": "float_share_10k",
    "自由流通股本(万股)": "free_float_share_10k",
    "总市值(万元)": "total_mv_10k",
    "流通市值(万元)": "circ_mv_10k",
}

TECH_FACTOR_COLUMN_MAP = {
    "股票代码": "symbol",
    "交易日期": "trade_date",
    "复权因子": "adj_factor",
    "涨跌幅(%)": "tech_pct_chg",
    "换手率(%)": "tech_turnover_rate",
    "量比": "tech_volume_ratio",
    "MA5": "ma5",
    "MA10": "ma10",
    "MA20": "ma20",
    "MA60": "ma60",
    "MACD_DIF": "macd_dif",
    "MACD_DEA": "macd_dea",
    "MACD": "macd",
    "RSI6": "rsi6",
    "RSI12": "rsi12",
    "RSI24": "rsi24",
    "KDJ_K": "kdj_k",
    "KDJ_D": "kdj_d",
    "KDJ_J": "kdj_j",
    "BOLL_UPPER": "boll_upper",
    "BOLL_MID": "boll_mid",
    "BOLL_LOWER": "boll_lower",
    "ATR": "atr",
    "OBV": "obv",
    "连涨天数": "up_streak_days",
    "连跌天数": "down_streak_days",
    "阶段新高天数": "days_from_new_high",
    "阶段新低天数": "days_from_new_low",
}

AKQ_BASE_COLUMNS = [
    "timestamp",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
]

CSV_ENCODINGS = ("utf-8", "utf-8-sig", "gb18030", "gbk")


class DataCleanError(RuntimeError):
    """数据清洗异常。"""


def normalize_stock_code(raw: Any) -> str:
    """将股票代码归一化到 6 位数字字符串。"""
    digits = re.sub(r"\D", "", str(raw))
    if not digits:
        raise ValueError(f"无法从 {raw!r} 提取股票代码")
    return digits[-6:].zfill(6)


def read_csv_bytes(raw: bytes) -> pd.DataFrame:
    """自动识别编码读取 CSV。"""
    for encoding in CSV_ENCODINGS:
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(io.BytesIO(raw))


def read_csv_path(path: Path) -> pd.DataFrame:
    """读取文件路径 CSV（自动编码）。"""
    raw = path.read_bytes()
    return read_csv_bytes(raw)


def clean_price_frame(df: pd.DataFrame, ts_code: str) -> pd.DataFrame:
    """清洗行情数据，输出 Akquant 基础 OHLCV 列。"""
    out = df.rename(columns=PRICE_COLUMN_MAP).copy()
    allowed = set(PRICE_COLUMN_MAP.values())
    out = out[[c for c in out.columns if c in allowed]]

    if "timestamp" not in out.columns:
        raise DataCleanError("行情数据缺少日期列")

    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out["symbol"] = ts_code

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "pct_chg",
        "turnover_rate",
    ]
    for col in numeric_columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["timestamp", "open", "high", "low", "close"])
    out = out.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")

    base = [c for c in AKQ_BASE_COLUMNS if c in out.columns]
    extras = [c for c in out.columns if c not in base]
    return out[base + extras].reset_index(drop=True)


def _clean_indicator_common(
    df: pd.DataFrame,
    column_map: dict[str, str],
    force_symbol: str | None = None,
) -> pd.DataFrame:
    out = df.rename(columns=column_map).copy()
    allowed = set(column_map.values())
    out = out[[c for c in out.columns if c in allowed]]

    if "trade_date" in out.columns:
        out["timestamp"] = pd.to_datetime(
            out["trade_date"].astype(str),
            format="%Y%m%d",
            errors="coerce",
        )
        out = out.drop(columns=["trade_date"])
    elif "timestamp" in out.columns:
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    else:
        raise DataCleanError("指标数据缺少交易日期")

    if force_symbol:
        out["symbol"] = force_symbol
    elif "symbol" not in out.columns:
        raise DataCleanError("指标数据缺少股票代码")

    for col in out.columns:
        if col in {"timestamp", "symbol"}:
            continue
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["timestamp"])
    dedup_subset = ["timestamp", "symbol"] if "symbol" in out.columns else ["timestamp"]
    out = out.sort_values(dedup_subset).drop_duplicates(subset=dedup_subset, keep="last")

    base = [c for c in ["timestamp", "symbol"] if c in out.columns]
    extras = [c for c in out.columns if c not in base]
    return out[base + extras].reset_index(drop=True)


def clean_indicator_frame(df: pd.DataFrame, force_symbol: str | None = None) -> pd.DataFrame:
    """清洗每日指标数据。"""
    return _clean_indicator_common(df, INDICATOR_COLUMN_MAP, force_symbol=force_symbol)


def clean_tech_factor_frame(df: pd.DataFrame, force_symbol: str | None = None) -> pd.DataFrame:
    """清洗技术因子数据。"""
    return _clean_indicator_common(df, TECH_FACTOR_COLUMN_MAP, force_symbol=force_symbol)


def merge_symbol_frames(
    price_df: pd.DataFrame,
    indicator_df: pd.DataFrame | None = None,
    tech_factor_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """按日期合并行情 + 每日指标 + 技术因子。"""
    merged = price_df.copy()

    for extra in (indicator_df, tech_factor_df):
        if extra is None or extra.empty:
            continue

        add = extra.copy()
        overlap = [
            c
            for c in add.columns
            if c in merged.columns and c not in {"timestamp", "symbol"}
        ]
        if overlap:
            add = add.drop(columns=overlap)

        if "symbol" in add.columns:
            add = add.drop(columns=["symbol"])

        merged = merged.merge(add, on="timestamp", how="left")

    merged = merged.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    base = [c for c in AKQ_BASE_COLUMNS if c in merged.columns]
    extras = [c for c in merged.columns if c not in base]
    return merged[base + extras].reset_index(drop=True)


def extract_zip_file(zip_path: Path, target_dir: Path, overwrite: bool = False) -> tuple[int, int]:
    """解压 zip 到目录，返回（本次写入文件数, 总文件数）。"""
    target_dir.mkdir(parents=True, exist_ok=True)

    extracted = 0
    with zipfile.ZipFile(zip_path) as zf:
        files = [name for name in zf.namelist() if not name.endswith("/")]
        for name in files:
            target = target_dir / name
            if target.exists() and not overwrite:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(name) as src, target.open("wb") as dst:
                dst.write(src.read())
            extracted += 1

    return extracted, len(files)


def build_member_map(zf: zipfile.ZipFile, kind: str) -> dict[str, str]:
    """构建代码到压缩包成员名的映射。"""
    mapping: dict[str, str] = {}
    for name in zf.namelist():
        if not name.lower().endswith(".csv"):
            continue
        stem = Path(name).stem
        if kind == "price":
            code = normalize_stock_code(stem.split("_")[0])
        else:
            code = normalize_stock_code(stem.split(".")[0])
        mapping[code] = name
    return mapping


def load_stock_map(stock_list_path: Path) -> dict[str, str]:
    """加载 6 位代码 -> TS 代码 映射。"""
    df = read_csv_path(stock_list_path)
    if "TS代码" not in df.columns:
        raise DataCleanError(f"{stock_list_path} 缺少 TS代码 列")

    if "股票代码" in df.columns:
        key_series = df["股票代码"].map(normalize_stock_code)
    else:
        key_series = df["TS代码"].map(normalize_stock_code)

    value_series = df["TS代码"].astype(str)
    return dict(zip(key_series, value_series, strict=False))


def load_from_zip(zf: zipfile.ZipFile, member: str) -> pd.DataFrame:
    with zf.open(member) as file_obj:
        return read_csv_bytes(file_obj.read())


def run_clean(
    price_zip: Path,
    indicator_zip: Path,
    output_dir: Path,
    stock_list_path: Path,
    tech_factor_zip: Path | None = None,
    limit: int | None = None,
) -> tuple[int, int]:
    """执行清洗并写出按股票拆分的 CSV。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    stock_map = load_stock_map(stock_list_path)

    success = 0
    failures = 0
    manifest_rows: list[dict[str, Any]] = []

    with (
        zipfile.ZipFile(price_zip) as price_zf,
        zipfile.ZipFile(indicator_zip) as ind_zf,
        zipfile.ZipFile(tech_factor_zip) if tech_factor_zip else io.BytesIO() as tech_obj,
    ):
        tech_zf = tech_obj if isinstance(tech_obj, zipfile.ZipFile) else None

        price_map = build_member_map(price_zf, kind="price")
        ind_map = build_member_map(ind_zf, kind="indicator")
        tech_map = build_member_map(tech_zf, kind="indicator") if tech_zf else {}

        codes = sorted(set(price_map) & set(ind_map))
        if limit is not None:
            codes = codes[: max(0, limit)]

        for index, code in enumerate(codes, start=1):
            ts_code = stock_map.get(code) or f"{code}.UNKNOWN"
            try:
                price_raw = load_from_zip(price_zf, price_map[code])
                ind_raw = load_from_zip(ind_zf, ind_map[code])

                price_df = clean_price_frame(price_raw, ts_code=ts_code)
                ind_df = clean_indicator_frame(ind_raw, force_symbol=ts_code)

                tech_df = None
                if tech_zf and code in tech_map:
                    tech_raw = load_from_zip(tech_zf, tech_map[code])
                    tech_df = clean_tech_factor_frame(tech_raw, force_symbol=ts_code)

                merged = merge_symbol_frames(price_df, ind_df, tech_df)
                export_df = merged.copy()
                export_df.insert(0, "date", export_df["timestamp"])

                out_file = output_dir / f"{ts_code}.csv"
                export_df.to_csv(out_file, index=False)

                manifest_rows.append(
                    {
                        "symbol": ts_code,
                        "rows": int(len(export_df)),
                        "start": merged["timestamp"].min().strftime("%Y-%m-%d")
                        if not merged.empty
                        else None,
                        "end": merged["timestamp"].max().strftime("%Y-%m-%d")
                        if not merged.empty
                        else None,
                        "file": out_file.name,
                    }
                )
                success += 1
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"[WARN] {ts_code} 清洗失败: {exc}")

            if index % 200 == 0:
                print(f"[INFO] 已处理 {index}/{len(codes)}")

    manifest = pd.DataFrame(manifest_rows)
    manifest = manifest.sort_values("symbol") if not manifest.empty else manifest
    manifest.to_csv(output_dir / "manifest.csv", index=False)

    return success, failures


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="A 股数据解压与清洗 (Akquant)")
    parser.add_argument(
        "--price-zip",
        type=Path,
        default=Path("A股数据_zip/daily_hfq.zip"),
        help="行情压缩包（默认后复权日线）",
    )
    parser.add_argument(
        "--indicator-zip",
        type=Path,
        default=Path("A股数据_每日指标/每日指标.zip"),
        help="每日指标压缩包",
    )
    parser.add_argument(
        "--tech-factor-zip",
        type=Path,
        default=None,
        help="可选：技术因子压缩包（如 技术因子_前复权.zip）",
    )
    parser.add_argument(
        "--stock-list",
        type=Path,
        default=Path("A股数据_zip/股票列表.csv"),
        help="股票列表 CSV，用于映射 TS 代码",
    )
    parser.add_argument(
        "--extract-dir",
        type=Path,
        default=Path("data/raw_extracted"),
        help="解压目录",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/akquant/daily_hfq_with_indicators"),
        help="清洗结果输出目录",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="仅清洗，不执行解压",
    )
    parser.add_argument(
        "--overwrite-extract",
        action="store_true",
        help="解压时覆盖已存在文件",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="仅处理前 N 只股票（调试用）",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    if not args.price_zip.exists():
        raise FileNotFoundError(f"未找到行情压缩包: {args.price_zip}")
    if not args.indicator_zip.exists():
        raise FileNotFoundError(f"未找到指标压缩包: {args.indicator_zip}")
    if not args.stock_list.exists():
        raise FileNotFoundError(f"未找到股票列表: {args.stock_list}")
    if args.tech_factor_zip and not args.tech_factor_zip.exists():
        raise FileNotFoundError(f"未找到技术因子压缩包: {args.tech_factor_zip}")

    if not args.skip_extract:
        datasets = [
            (args.price_zip, args.extract_dir / "price"),
            (args.indicator_zip, args.extract_dir / "indicator"),
        ]
        if args.tech_factor_zip:
            datasets.append((args.tech_factor_zip, args.extract_dir / "tech_factor"))

        for zip_path, target in datasets:
            extracted, total = extract_zip_file(
                zip_path,
                target,
                overwrite=args.overwrite_extract,
            )
            print(f"[INFO] 解压 {zip_path} -> {target} ({extracted}/{total})")

    success, failures = run_clean(
        price_zip=args.price_zip,
        indicator_zip=args.indicator_zip,
        output_dir=args.output_dir,
        stock_list_path=args.stock_list,
        tech_factor_zip=args.tech_factor_zip,
        limit=args.limit,
    )
    print(f"[DONE] 清洗完成，成功 {success}，失败 {failures}，输出目录: {args.output_dir}")


if __name__ == "__main__":
    main()
