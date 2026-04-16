# 第 16 章：Rust 指标全景与工程化使用

本章目标：让你能够把 AKQuant 的 Rust 指标体系当作“可教学、可迁移、可工程化复用”的指标库来使用。

## 1. 为什么要学习 Rust 指标体系

- 同一套函数签名可切换 `backend="python"` 与 `backend="rust"`，便于迁移与性能升级。
- Rust 实现覆盖了完整的趋势、动量、波动率、量能与数学变换指标。
- 对策略开发而言，最关键的是理解三件事：输入结构、输出结构、warmup 行为。

## 2. 全量覆盖范围（当前 103 个）

- Momentum：8 个
- Moving Average & Transforms：59 个
- Trend：20 个
- Volatility：11 个
- Volume：5 个

完整逐项解释请使用课程配套词典页：
- [Rust 指标全量说明（103 个）](../guide/rust_indicator_reference.md)

## 3. 学会“读指标”的统一方法

每个指标都用同一模板理解：

1. 输入是什么（`close` 还是 `high, low, close`）
2. 输出是什么（单序列、双序列还是三序列）
3. warmup 需要几根 K 线
4. 指标在策略里承担哪一种角色（主信号、过滤器、风控、确认）

## 4. 五大类指标怎么教、怎么用

### 4.1 趋势类（方向与强度）

- 方向判别：`EMA/SMA/TEMA/KAMA/SAR`
- 强度过滤：`ADX/ADXR/DX`
- 典型组合：`EMA + ADX + NATR`

### 4.2 动量类（变化速度）

- 速度/斜率：`ROC/ROCP/ROCR/ROCR100/MOM`
- 过热过冷：`RSI/CMO/WILLR`
- 典型组合：`BBANDS + RSI + MOM`

### 4.3 波动率类（风险尺度）

- 范围与波动：`ATR/NATR/TRANGE/STDDEV/VAR`
- 通道类：`BollingerBands`
- 价格派生：`MEDPRICE/TYPPRICE/WCLPRICE/AVGPRICE/MIDPRICE`

### 4.4 量价类（成交量确认）

- 趋势确认：`OBV/AD`
- 动量确认：`MFI/ADOSC`
- K 线力量：`BOP`

### 4.5 数学变换类（特征工程）

- 对数/指数：`LN/LOG10/LOG1P/EXP/EXPM1`
- 三角与双曲：`SIN/COS/TAN/ASIN/ACOS/ATAN/SINH/COSH/TANH`
- 代数运算：`ADD/SUB/MULT/DIV/MOD/POW/MAX2/MIN2`
- 规整变换：`ABS/SIGN/ROUND/CLIP/CLAMP01/SQ/CUBE/RECIP/INV_SQRT/DEG2RAD`

## 5. 教学中的核心坑位

### 5.1 warmup 区段误用

- 窗口不足时指标值无效，必须在教学与作业中强调空值过滤。

### 5.2 多输出解包顺序错误

- `MACD -> (macd, signal, hist)`
- `BollingerBands -> (upper, middle, lower)`
- `STOCH -> (slowk, slowd)`
- `AROON -> (aroondown, aroonup)`

### 5.3 迁移时直接切 Rust

- 正确流程是先 `python` 对齐，再切 `rust` 提速。

## 6. 标准教学脚手架（可直接放实验课）

```python
import numpy as np
from akquant import talib as ta

close = np.asarray(df["close"], dtype=float)
high = np.asarray(df["high"], dtype=float)
low = np.asarray(df["low"], dtype=float)
volume = np.asarray(df["volume"], dtype=float)

ema_fast = np.asarray(ta.EMA(close, timeperiod=20, backend="rust"), dtype=float)
ema_slow = np.asarray(ta.EMA(close, timeperiod=60, backend="rust"), dtype=float)
adx = np.asarray(ta.ADX(high, low, close, timeperiod=14, backend="rust"), dtype=float)
natr = np.asarray(ta.NATR(high, low, close, timeperiod=14, backend="rust"), dtype=float)
rsi = np.asarray(ta.RSI(close, timeperiod=14, backend="rust"), dtype=float)

if np.isnan([ema_fast[-1], ema_slow[-1], adx[-1], natr[-1], rsi[-1]]).any():
    return
```

## 7. 推荐教学路径

1. 第 1 周：先教 `EMA/RSI/ATR` 的输入输出与 warmup。
2. 第 2 周：加入 `MACD/BBANDS/STOCH` 的多输出处理。
3. 第 3 周：引入 `ADX/NATR/SAR` 做风险过滤。
4. 第 4 周：做一次 `python -> rust` 迁移实验与回归验证。
5. 第 5 周：用数学变换类指标做简单特征工程实验。

## 8. 本章小结

- Rust 指标体系已经覆盖策略开发所需的核心技术面能力。
- 真正决定教学效果的不是指标数量，而是“结构化理解 + 可复现实验”。
- 完整指标字典与解释请始终以参考页为准：
  - [Rust 指标全量说明（103 个）](../guide/rust_indicator_reference.md)
