# AKQuant 示例映射（按任务选模板）

本表用于在生成代码前快速定位最接近的官方示例，避免“从零想象”实现。

## 1) 策略与回测基础

| 场景 | 首选示例 | 关键 API / 结构 |
|---|---|---|
| 单标的最小回测 | `examples/01_quickstart.py` | `Strategy` + `on_bar` + `run_backtest` |
| 双均线趋势跟踪 | `examples/strategies/01_stock_dual_moving_average.py` | `warmup_period` + `get_history` + `order_target_percent` |
| ATR 突破（避免未来函数） | `examples/strategies/03_stock_atr_breakout.py` | `N+1` + `[:-1]` |

## 2) 多标的与组合调仓

| 场景 | 首选示例 | 关键 API / 结构 |
|---|---|---|
| 动量轮动（定时调仓） | `examples/strategies/05_stock_momentum_rotation_timer.py` | `add_daily_timer` + `on_timer` |
| 动量轮动（收齐时间片） | `examples/strategies/06_stock_momentum_rotation_bucket.py` | `timestamp -> set(symbol)` 桶缓存 |
| TopN 等权组合再平衡 | `examples/43_target_weights_rebalance.py` | `order_target_weights` + `liquidate_unmentioned` |

## 3) 指标、因子、分析输出

| 场景 | 首选示例 | 关键 API / 结构 |
|---|---|---|
| TA-Lib 指标组合 | `examples/45_talib_indicator_playbook_demo.py` | `akquant.talib` + 信号组合 |
| 因子表达式计算 | `examples/19_factor_expression.py` | `FactorEngine` + `ParquetDataCatalog` |
| 报告与结构化分析输出 | `examples/33_report_and_analysis_outputs.py` | `result.report()` + `exposure_df()` |

## 4) 参数化与优化

| 场景 | 首选示例 | 关键 API / 结构 |
|---|---|---|
| 基础参数优化 | `examples/02_parameter_optimization.py` | 参数网格 + 回测评估 |
| 高级参数优化 | `examples/03_parameter_optimization_advanced.py` | 扩展指标与过滤 |
| 运行时参数映射 | `examples/22_strategy_runtime_config_demo.py` | runtime kwargs 与策略参数分离 |

## 5) 数据准备约定（推荐）

1. 若输入来自 AKShare：统一列名小写，确保字段存在：
   - `date/open/high/low/close/volume/symbol`
2. `date` 转 `pd.to_datetime`，时区建议 `Asia/Shanghai`。
3. 多标的回测传 `dict[str, DataFrame]`。
4. 生产或研究环境优先走本地 DuckDB；仅在不足时回退 AkShare。
