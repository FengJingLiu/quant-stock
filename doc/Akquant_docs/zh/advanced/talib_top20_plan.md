# AKQuant 指标兼容与迁移总览（用户版）

## 更新信息

- 更新时间（UTC+8）：2026-03-13 00:00
- 口径版本号：plan-baseline-v1
- 数据来源口径：以当前仓库主干代码与测试为准，核心来源包括 `src/indicators.rs`、`python/akquant/talib/funcs.py`、`tests/test_talib_backend.py`、`tests/test_talib_compat.py`。

## 这份计划给用户什么价值

- 快速判断：`akquant.talib` 是否覆盖你的常用 TA-Lib 指标。
- 快速迁移：先对齐历史结果，再切高性能后端。
- 快速落地：按场景选择指标组合，减少试错成本。

## 当前状态（基于本仓库 2026-03）

- `akquant.talib` 支持 `backend=auto/python/rust`。
- Top20 高频迁移指标已全部完成双后端兼容。
- Top20 之外扩展已完成到批次 `T`，当前总计支持 **103 个指标**。
- 多输出指标结构稳定：
  - `MACD -> (macd, signal, hist)`
  - `BBANDS -> (upper, middle, lower)`
  - `STOCH -> (slowk, slowd)`
  - `AROON -> (aroondown, aroonup)`

## 用户迁移建议（建议顺序）

1. 用 `backend="python"` 对齐老策略结果。
2. 切到高性能后端复核信号一致性。
3. 重点检查 warmup 区段与多输出解包顺序。
4. 最后再做参数微调与回测优化。

## Top20 指标（已全部完成）

| 指标 | Python backend | 高性能 backend | 输出形态 |
| :--- | :--- | :--- | :--- |
| SMA | ✅ | ✅ | 单序列 |
| EMA | ✅ | ✅ | 单序列 |
| MACD | ✅ | ✅ | 三序列 |
| RSI | ✅ | ✅ | 单序列 |
| BBANDS | ✅ | ✅ | 三序列 |
| ATR | ✅ | ✅ | 单序列 |
| ROC | ✅ | ✅ | 单序列 |
| WILLR | ✅ | ✅ | 单序列 |
| CCI | ✅ | ✅ | 单序列 |
| ADX | ✅ | ✅ | 单序列 |
| STOCH | ✅ | ✅ | 双序列 |
| MFI | ✅ | ✅ | 单序列 |
| OBV | ✅ | ✅ | 单序列 |
| TRIX | ✅ | ✅ | 单序列 |
| MOM | ✅ | ✅ | 单序列 |
| DEMA | ✅ | ✅ | 单序列 |
| TEMA | ✅ | ✅ | 单序列 |
| KAMA | ✅ | ✅ | 单序列 |
| NATR | ✅ | ✅ | 单序列 |
| SAR | ✅ | ✅ | 单序列 |

## 扩展批次进度（A-T）

| 批次 | 内容 | 状态 |
| :--- | :--- | :--- |
| A | `ADX/CCI/STOCH/WILLR/ROC` | 已完成 |
| B | `MFI/OBV/TRIX/MOM/DEMA` | 已完成 |
| C | `TEMA/KAMA/NATR/SAR` | 已完成 |
| D | `WMA/ADXR/CMO/TRANGE/BOP` | 已完成 |
| E | `PPO/APO/ULTOSC/PLUS_DI/MINUS_DI` | 已完成 |
| F | `DX/AD/ADOSC/AROON/AROONOSC` | 已完成 |
| G | `TRIMA/STDDEV/VAR/LINEARREG/LINEARREG_SLOPE` | 已完成 |
| H | `LINEARREG_INTERCEPT/LINEARREG_ANGLE/TSF/CORREL/BETA` | 已完成 |
| I | `LINEARREG_R2/T3/MEDPRICE/TYPPRICE/WCLPRICE` | 已完成 |
| J | `AVGPRICE/MIDPOINT/MIDPRICE/HT_TRENDLINE/MAMA` | 已完成 |
| K | `MAX/MIN/MAXINDEX/MININDEX/MINMAX` | 已完成 |
| L | `SUM/AVGDEV/RANGE/STDDEV扩展/VAR扩展` | 已完成 |
| M | `MINMAXINDEX/ROCP/ROCR/ROCR100/COVAR` | 已完成 |
| N | `LN/LOG10/SQRT/CEIL/FLOOR` | 已完成 |
| O | `SIN/COS/TAN/ASIN/ACOS` | 已完成 |
| P | `ATAN/SINH/COSH/TANH/EXP` | 已完成 |
| Q | `ABS/SIGN/ADD/SUB/MULT` | 已完成 |
| R | `DIV/MAX2/MIN2/CLIP/ROUND` | 已完成 |
| S | `POW/MOD/CLAMP01/SQ/CUBE` | 已完成 |
| T | `RECIP/INV_SQRT/LOG1P/EXPM1/DEG2RAD` | 已完成 |

## 建议阅读顺序

- 先看全量词典： [AKQuant 指标全量说明（103 个）](../guide/rust_indicator_reference.md)
- 再看场景速查： [按策略场景选指标速查表（四象限）](../guide/indicator_scenario_quickref.md)
- 最后看组合实战： [指标组合实战手册](../guide/talib_indicator_playbook.md)

## 测试入口

- [test_talib_compat.py](https://github.com/akfamily/akquant/blob/main/tests/test_talib_compat.py)
- [test_talib_backend.py](https://github.com/akfamily/akquant/blob/main/tests/test_talib_backend.py)
