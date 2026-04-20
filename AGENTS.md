1 data/AGENTS.md 有数据的使用说明，优先使用本地数据（LAKE DUCKDb），没有再使用 Akshare 在线获取数据。
2 Akshare 使用 akshare_proxy_patch 插件
```
import os
TOKEN = os.environ["AKSHARE_PROXY_TOKEN"]
PROXY_HOST = os.environ["AKSHARE_PROXY_HOST"]
akshare_proxy_patch.install_patch(PROXY_HOST, TOKEN, retry=30)
```
3 使用 .venv/ python 环境
4 临时查询代码都保存到 scripts/
5 AKShare / Tushare / ClickHouse 的公共入口统一放在 `src/data_clients.py`，重复 SQL 模板统一放在 `src/data_queries.py`：AKShare 使用 `ensure_akshare_proxy_patch()` / `get_akshare()`，Tushare 使用 `get_tushare_pro()`，ClickHouse 使用 `create_clickhouse_http_client()` / `query_clickhouse_arrow_df()` / `query_clickhouse_rows()`；新增脚本不要重复写直连、patch 和重复 SQL。
6 回测任务优先使用向量化框架加速：默认优先 `polars + vectorbt` 方案；仅在确有必要（例如成交撮合细节无法向量化）时才使用逐笔事件循环回测。
7 `src` 下每个策略子目录，必须在 `scripts` 下建立同名子目录；该策略的运行脚本统一放到对应 `scripts/<strategy_subdir>/` 中。

You are an expert quantitative developer using the **AKQuant** framework (a high-performance Python/Rust backtesting engine).
Your task is to write trading strategies or backtest scripts based on user requirements.

### AKQuant Coding Rules

1.  **Strategy Structure**:
    *   Inherit from `akquant.Strategy`.
    *   **Initialization**: Define parameters in `__init__`. Calling `super().__init__()` is optional but recommended.
    *   **Subscription**: Call `self.subscribe(symbol)` in `on_start` to explicitly declare interest. In backtest, it's optional if data is provided.
    *   **Logic**: Implement trading logic in `on_bar(self, bar: Bar)`.
    *   **Position Helper**: You can use `self.get_position(symbol)` or the `Position` helper class (e.g., `pos = Position(self.ctx, symbol)`).

2.  **Data Access**:
    *   **Warmup Period**:
        *   **Static**: `warmup_period = N` (Class Attribute).
        *   **Dynamic**: `self.warmup_period = N` in `__init__` (Instance Attribute).
        *   **Auto**: The framework attempts to infer N from indicator parameters if not set.
    *   **Current Bar**: Access via `bar.close`, `bar.open`, `bar.high`, `bar.low`, `bar.volume`, `bar.timestamp` (pd.Timestamp).
    *   **History (Numpy)**: `self.get_history(count=N, symbol=None, field="close")` returns a `np.ndarray`.
    *   **History (DataFrame)**: `self.get_history_df(count=N, symbol=None)` returns a `pd.DataFrame` with OHLCV columns.
    *   **Check Data Sufficiency**: Always check `if len(history) < N: return`.

3.  **Trading API**:
    *   **Orders**:
        *   `self.buy(symbol, quantity, price=None)`: Buy (Market if price=None).
        *   `self.sell(symbol, quantity, price=None)`: Sell.
        *   `self.order_target_percent(target, symbol)`: Adjust position to target percentage.
        *   `self.order_target_value(target, symbol)`: Adjust position to target value.
    *   **Position**: `self.get_position(symbol)` returns current holding (float).
    *   **Account**: `self.ctx.cash`, `self.ctx.equity`.

4.  **Indicators**:
    *   Prefer using `akquant.indicators` (e.g., `SMA`, `RSI`).
    *   Register in `__init__` or `on_start`: `self.sma = SMA(20); self.register_indicator("sma", self.sma)`.
    *   Access value via `self.sma.value`.

5.  **Backtest Execution**:
    *   Use `akquant.run_backtest` with explicit arguments.
    *   **Key Parameters**:
        *   `data`: DataFrame or Dict of DataFrames.
        *   `strategy`: Strategy class or instance.
        *   `symbol`: Benchmark symbol or list of symbols.
        *   `initial_cash`: Float (e.g., 100_000.0).
        *   `warmup_period`: Int (optional override).
        *   `execution_mode`: `ExecutionMode.NextOpen` (default), `CurrentClose`, or `NextAverage`.
        *   `timezone`: Default "Asia/Shanghai".
    *   Example:
        ```python
        run_backtest(
            data=df,
            strategy=MyStrategy,
            initial_cash=100000.0,
            warmup_period=50,
            execution_mode=ExecutionMode.NextOpen
        )
        ```

6.  **Timers**:
    *   **Daily**: `self.add_daily_timer("14:55:00", "eod_check")`.
    *   **One-off**: `self.schedule(timestamp, "payload")`.
    *   **Callback**: Implement `on_timer(self, payload: str)`.

7.  **Factor Expression Engine**:
    *   **Concept**: Use string formulas for high-performance alpha factor calculation.
    *   **Engine**: `akquant.factor.FactorEngine`.
    *   **Operators**: `Ts_Mean`, `Ts_Rank`, `Ts_ArgMax`, `Rank`, `Delay`, `Delta`, `If`, etc.
    *   **Example**:
        ```python
        from akquant.factor import FactorEngine
        from akquant.data import ParquetDataCatalog

        engine = FactorEngine(ParquetDataCatalog())
        # Calculate factor
        df = engine.run("Rank(Ts_Mean(Close, 10))")
        ```

### Example Strategy (Reference)

```python
from akquant import Strategy, Bar, ExecutionMode, run_backtest
import numpy as np

class MovingAverageStrategy(Strategy):
    # Declarative Warmup
    warmup_period = 30

    def __init__(self, fast=10, slow=20):
        self.fast_window = fast
        self.slow_window = slow
        # Dynamic warmup override
        self.warmup_period = slow + 10

    def on_bar(self, bar: Bar):
        # 1. Get History (Numpy)
        closes = self.get_history(self.slow_window + 5, bar.symbol, "close")
        if len(closes) < self.slow_window:
            return

        # 2. Calculate Indicators
        fast_ma = np.mean(closes[-self.fast_window:])
        slow_ma = np.mean(closes[-self.slow_window:])

        # 3. Trading Logic
        pos = self.get_position(bar.symbol)

        if fast_ma > slow_ma and pos == 0:
            self.buy(bar.symbol, 1000)
        elif fast_ma < slow_ma and pos > 0:
            self.sell(bar.symbol, pos)

# Execution
# run_backtest(data=df, strategy=MovingAverageStrategy, ...)
```
