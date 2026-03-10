"""
A股价值策略回测（本地后复权 + 每日动态筛选 + 满仓）

数据源: data/akquant/daily_hfq_with_indicators
每日筛选条件:
- PE(TTM) < 20
- 股息率(TTM/静态) > 3%
- 剔除 ST/退市风险名称
交易规则:
- 每个交易日根据当日可见因子重新筛选
- 若有候选标的，按“股息率降序 + PE升序”选前 1 只并满仓
- 其余持仓清仓
"""

from pathlib import Path

import numpy as np
import pandas as pd
from akquant import Strategy, run_backtest

# ==================== 参数配置 ====================
INITIAL_CASH = 200_000.0
PE_THRESHOLD = 20
DIV_THRESHOLD = 3.0
TOP_SYMBOLS_PER_DAY = 1  # 用户要求“全仓买入”，默认每日只选1只
START_DATE = "2021-03-01"
END_DATE = "2026-02-26"

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data/akquant/daily_hfq_with_indicators"
STOCK_LIST_FILE = BASE_DIR / "A股数据_zip/股票列表.csv"


def read_csv_auto(path: Path, **kwargs) -> pd.DataFrame:
    """带编码兜底的 CSV 读取。"""
    for enc in ("utf-8", "utf-8-sig", "gb18030", "gbk"):
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, **kwargs)


def load_stock_name_map() -> dict[str, str]:
    """加载 TS 代码到股票名称映射。"""
    if not STOCK_LIST_FILE.exists():
        return {}

    df = read_csv_auto(STOCK_LIST_FILE, usecols=["TS代码", "股票名称"])
    df = df.dropna(subset=["TS代码"]).drop_duplicates(subset=["TS代码"], keep="last")
    return dict(zip(df["TS代码"].astype(str), df["股票名称"].astype(str), strict=False))


def is_bad_name(name: str) -> bool:
    """过滤 ST、退市风险标的。"""
    if not name:
        return False
    upper = name.upper()
    return "ST" in upper or "退" in name


def get_trading_dates(data_dir: Path, start_date: str, end_date: str) -> list[pd.Timestamp]:
    """获取交易日期序列。"""
    csv_files = sorted(p for p in data_dir.glob("*.csv") if p.name != "manifest.csv")
    if not csv_files:
        return []

    benchmark = data_dir / "000001.SZ.csv"
    if not benchmark.exists():
        benchmark = csv_files[0]

    df = pd.read_csv(benchmark, usecols=["date"])
    dates = pd.to_datetime(df["date"], errors="coerce").dropna()
    dates = dates[(dates >= pd.Timestamp(start_date)) & (dates <= pd.Timestamp(end_date))]
    dates = pd.Series(dates).drop_duplicates().sort_values()
    return [pd.Timestamp(x).normalize() for x in dates]


def build_daily_selection(
    data_dir: Path,
    trading_dates: list[pd.Timestamp],
    name_map: dict[str, str],
    top_n: int,
) -> dict[pd.Timestamp, list[str]]:
    """构建每日无前视选股结果。

    关键点：对每只股票因子按交易日 ffill 到当日，只使用当日及历史可见数据。
    """
    if not trading_dates:
        return {}

    date_index = pd.Index(trading_dates, name="date")
    candidates_by_day: dict[pd.Timestamp, list[tuple[str, float, float]]] = {
        d: [] for d in trading_dates
    }

    csv_files = sorted(p for p in data_dir.glob("*.csv") if p.name != "manifest.csv")
    print(f"[选股] 每日筛选，扫描 {len(csv_files)} 只股票...")

    for i, csv_file in enumerate(csv_files, 1):
        symbol = csv_file.stem
        if is_bad_name(name_map.get(symbol, "")):
            continue

        try:
            df = pd.read_csv(
                csv_file,
                usecols=["date", "pe_ttm", "dividend_yield_ttm", "dividend_yield"],
            )
        except Exception:
            continue

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["pe_ttm"] = pd.to_numeric(df["pe_ttm"], errors="coerce")
        dy_ttm = pd.to_numeric(df["dividend_yield_ttm"], errors="coerce")
        dy_static = pd.to_numeric(df["dividend_yield"], errors="coerce")
        df["div_yield"] = dy_ttm.where(dy_ttm.notna(), dy_static)

        factors = df[["date", "pe_ttm", "div_yield"]].dropna(subset=["date"])
        if factors.empty:
            continue

        factors = factors.sort_values("date").drop_duplicates(subset=["date"], keep="last")
        factors = factors.set_index("date").reindex(date_index, method="ffill")

        valid = (
            (factors["pe_ttm"] > 0)
            & (factors["pe_ttm"] < PE_THRESHOLD)
            & (factors["div_yield"] > DIV_THRESHOLD)
        )

        for dt, row in factors[valid].iterrows():
            candidates_by_day[dt].append((symbol, float(row["div_yield"]), float(row["pe_ttm"])))

        if i % 1000 == 0:
            print(f"  已扫描 {i}/{len(csv_files)}")

    plan: dict[pd.Timestamp, list[str]] = {}
    for dt in trading_dates:
        rows = candidates_by_day[dt]
        rows.sort(key=lambda x: (-x[1], x[2], x[0]))
        plan[dt] = [x[0] for x in rows[:top_n]]

    return plan


def summarize_daily_plan(plan: dict[pd.Timestamp, list[str]]) -> None:
    """打印每日选股计划摘要。"""
    if not plan:
        print("[选股] 每日计划为空")
        return

    counts = [len(v) for v in plan.values()]
    union_symbols = sorted({s for syms in plan.values() for s in syms})

    print(f"[选股] 交易日数: {len(plan)}")
    print(f"[选股] 每日入选数(最小/平均/最大): {min(counts)}/{np.mean(counts):.2f}/{max(counts)}")
    print(f"[选股] 覆盖股票数(并集): {len(union_symbols)}")


def load_local_data(
    symbols: list[str],
    start_date: str = START_DATE,
    end_date: str = END_DATE,
) -> dict[str, pd.DataFrame]:
    """从本地后复权目录加载回测数据。"""
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    print(f"\n[数据] 从本地加载 {len(symbols)} 只股票...")
    data: dict[str, pd.DataFrame] = {}

    for i, sym in enumerate(symbols, 1):
        csv_file = DATA_DIR / f"{sym}.csv"
        print(f"  [{i}/{len(symbols)}] {sym} ... ", end="", flush=True)

        if not csv_file.exists():
            print("✗ 文件不存在")
            continue

        try:
            df = pd.read_csv(csv_file)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
            elif "timestamp" in df.columns:
                df["date"] = pd.to_datetime(df["timestamp"], errors="coerce")
            else:
                print("✗ 缺少日期列")
                continue

            keep_cols = ["date", "open", "high", "low", "close", "volume"]
            if not set(keep_cols).issubset(df.columns):
                print("✗ 缺少OHLCV字段")
                continue

            df = df[keep_cols].copy()
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            df = df.dropna(subset=["date", "open", "high", "low", "close", "volume"])
            df = df[(df["date"] >= start_ts) & (df["date"] <= end_ts)]
            df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last")
            df["symbol"] = sym

            if len(df) > 0:
                data[sym] = df
                print(f"✓ {len(df)} 条")
            else:
                print("✗ 区间内无数据")
        except Exception as exc:  # noqa: BLE001
            print(f"✗ {exc}")

    print(f"[数据] 成功加载: {len(data)} 只")
    return data


class DailyFullPositionFactorStrategy(Strategy):
    """每日动态筛选 + 满仓策略。"""

    warmup_period = 1

    def __init__(self, daily_plan: dict[pd.Timestamp, list[str]]):
        self.warmup_period = 1
        self.daily_plan = {pd.Timestamp(k).normalize(): list(v) for k, v in daily_plan.items()}
        self.current_date: pd.Timestamp | None = None
        self.current_targets: set[str] = set()

    def _switch_day(self, bar_date: pd.Timestamp) -> None:
        if self.current_date == bar_date:
            return
        self.current_date = bar_date
        self.current_targets = set(self.daily_plan.get(bar_date, []))

    def on_bar(self, bar):
        bar_date = pd.Timestamp(bar.timestamp).tz_localize(None).normalize()
        self._switch_day(bar_date)

        symbol = bar.symbol
        pos = self.get_position(symbol)

        if not self.current_targets:
            if pos > 0:
                self.order_target_percent(0.0, symbol)
            return

        if symbol in self.current_targets:
            target_weight = 1.0 / len(self.current_targets)
            self.order_target_percent(target_weight, symbol)
        elif pos > 0:
            self.order_target_percent(0.0, symbol)


def run_backtest_daily(
    start_date: str = START_DATE,
    end_date: str = END_DATE,
    top_symbols_per_day: int = TOP_SYMBOLS_PER_DAY,
    initial_cash: float = INITIAL_CASH,
):
    """构建每日选股并执行回测，返回 result 供 Jupyter 分析与绘图。"""
    trading_dates = get_trading_dates(DATA_DIR, start_date, end_date)
    if not trading_dates:
        raise RuntimeError("没有可用交易日, 无法回测!")

    name_map = load_stock_name_map()
    daily_plan = build_daily_selection(
        data_dir=DATA_DIR,
        trading_dates=trading_dates,
        name_map=name_map,
        top_n=top_symbols_per_day,
    )
    summarize_daily_plan(daily_plan)

    candidate_symbols = sorted({sym for syms in daily_plan.values() for sym in syms})
    if not candidate_symbols:
        raise RuntimeError("没有符合条件的股票, 无法回测!")

    data = load_local_data(candidate_symbols, start_date=start_date, end_date=end_date)
    if not data:
        raise RuntimeError("没有有效的历史数据!")

    valid_symbols = set(data.keys())
    daily_plan = {
        d: [s for s in syms if s in valid_symbols]
        for d, syms in daily_plan.items()
    }

    print(
        f"\n[回测] 本金={initial_cash:,.0f}  "
        f"每日入选={top_symbols_per_day}  "
        f"PE<{PE_THRESHOLD}  股息率>{DIV_THRESHOLD}%"
    )

    result = run_backtest(
        data=data,
        strategy=DailyFullPositionFactorStrategy(daily_plan=daily_plan),
        symbol=list(data.keys()),
        initial_cash=initial_cash,
        commission_rate=0.0003,
    )
    return result, daily_plan, data


def run():
    result, _daily_plan, _data = run_backtest_daily(
        start_date=START_DATE,
        end_date=END_DATE,
        top_symbols_per_day=TOP_SYMBOLS_PER_DAY,
        initial_cash=INITIAL_CASH,
    )

    print(f"\n{'=' * 60}")
    print("回测结果")
    print(f"{'=' * 60}")
    print(f"  总收益率:  {result.metrics.total_return_pct:.2f}%")
    print(f"  夏普比率:  {result.metrics.sharpe_ratio:.2f}")
    print(f"  最大回撤:  {result.metrics.max_drawdown_pct:.2f}%")

    print("\n--- 绩效指标 ---")
    print(result.metrics_df)

    print("\n--- 交易记录 ---")
    print(result.trades_df)


if __name__ == "__main__":
    run()
