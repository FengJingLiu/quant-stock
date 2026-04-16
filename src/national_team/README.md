# 国家队跟踪策略 — 买卖信号因子 v2

## 策略概述

通过分析宽基 ETF 分钟级行情数据，识别"国家队"（汇金、证金等机构）护盘买入和高位出货的行为特征，输出实时买卖概率信号。

**核心假设**：国家队操作体量大，必然在分钟级 K 线上留下量价异常痕迹——买入时体现为"放量托住"，卖出时体现为"放量打压"。

**设计原则**：

| 原则 | 买入因子 | 卖出因子 |
|------|---------|---------|
| 信号偏好 | **重精度** — 宁可漏掉，不能误报 | **重召回** — 宁可多报，不能漏掉 |
| T+1 约束 | 信号当天不可能执行，必须次日可验证 | 同上 |
| 标签定义 | 绑定执行端（高冲击比率股票篮子）| 绑定 ETF 相对弱势 |
| 组合方法 | v1 线性 + sigmoid（可后续接 isotonic 校准）| 同上 |

**标的**：沪深 300 ETF (`510300.SH`)，参考指数 `000300`

---

## 数据源

- **ClickHouse** `astock` 库
  - `klines_1m_etf` — ETF 分钟 K 线（symbol, datetime, trade_date, open, high, low, close, volume, amount）
  - `klines_1m_index` — 指数分钟 K 线（同上字段）
- **舰队 ETF**（买入共振用）：510300.SH / 510050.SH / 510500.SH / 512100.SH

---

## 买入因子 (NT_Buy_Prob)

> 文件：`nt_buy_prob.py`  
> 类名：`NTBuyProb`

### 5 个特征

| # | 特征名 | 取值范围 | 计算逻辑 | 物理含义 |
|---|--------|---------|---------|---------|
| 1 | `stress_context` | 0 ~ 1 | `mean(ma_dev_score, drawdown_score)` | 市场压力背景：指数 20 日均线偏离 + 近 5 日累计回撤。偏离 ≤ -3% 或回撤 ≤ -5% 时趋近 1 |
| 2 | `vol_shock` | ℝ (z-score) | `robust_zscore(log(amount+1))` 按时间切片对齐 | 成交量异常冲击。log 变换使分布对称，按分钟时段标准化消除开盘/收盘效应 |
| 3 | `absorption` | ≥ 0 | `vol_shock × max(alpha, 0)` | 被托住的程度。放量同时 ETF 跑赢指数 → 有资金在接盘 |
| 4 | `resonance_count` | 0 ~ N (整数) | 跨舰队 ETF 中满足 `vol_shock > 2.326 AND alpha > 0` 的个数 | 多只 ETF 同时出现护盘信号 → 行为非偶发 |
| 5 | `lead_gap` | ℝ | `ret_etf - ret_index` | ETF-指数剪刀差。正值 = ETF 领先于指数，可能是机构抢跑 |

### 信号合成

```
logit = 2.0 × stress_context
      + 1.5 × vol_shock / 10
      + 2.0 × absorption / 0.1
      + 1.5 × resonance_count / 4
      + 1.0 × lead_gap × 100
      - 5.0  (bias)

nt_buy_prob = sigmoid(logit) = 1 / (1 + exp(-logit))
```

- **bias = -5.0** → 基准概率 ~0.7%，无信号时极低
- **归一化**：各特征缩放到近似 0~1 量级后加权

### 使用

```python
from src.national_team.nt_buy_prob import NTBuyProb
from src.national_team.ch_client import get_etf_1m, get_index_1m

factor = NTBuyProb()

etf_1m = get_etf_1m("510300.SH", "2024-01-01", "2024-06-30")
idx_1m = get_index_1m("000300", "2024-01-01", "2024-06-30")

# 舰队数据（可选，用于共振计算）
fleet = {
    "510050.SH": get_etf_1m("510050.SH", "2024-01-01", "2024-06-30"),
    "510500.SH": get_etf_1m("510500.SH", "2024-01-01", "2024-06-30"),
}

result = factor.compute(etf_1m, idx_1m, fleet)
signals = result.filter(pl.col("nt_buy_prob") > 0.20)
```

---

## 卖出因子 (NT_Sell_Prob)

> 文件：`nt_sell_prob.py`  
> 类名：`NTSellProb`

### 5 个特征

| # | 特征名 | 取值范围 | 计算逻辑 | 物理含义 |
|---|--------|---------|---------|---------|
| 1 | `distribution` | ≥ 0 | `vol_shock × max(-alpha, 0)` | 高位放量低效。放量但 ETF 跑输指数 → 有资金在出货 |
| 2 | `wick_ratio` | 0 ~ 1 | `(high - max(open, close)) / (high - low + ε)` | 上影线占比。连续值比布尔更细腻，数值大 = 上方抛压重 |
| 3 | `vwap_fail` | 0 或 1 | `1[close < session_vwap]` | 价格跌破日内累计 VWAP → 买方力量不足 |
| 4 | `lead_reversal` | 0 或 1 | `1[lead_gap_{t-1} > 0 AND lead_gap_t < 0]` | ETF 领先反转。上一分钟还领先指数，这一分钟落后 → 领先优势消失 |
| 5 | `propagation_fail` | 0 或 1 | `1[max(vol_shock, 5bar) > 2.326 AND sum(ret, 5bar) < 0]` | 传导断裂。近 5 bar 有大放量 (>q99) 但累计收益为负 → 放量没推上去 |

### 信号合成

```
logit = 2.0 × distribution / 0.1
      + 1.5 × wick_ratio
      + 1.5 × vwap_fail
      + 2.0 × lead_reversal
      + 1.5 × propagation_fail
      - 5.0  (bias)

nt_sell_prob = sigmoid(logit)
```

### 多日累积

```python
# 每日取 nt_sell_prob 最大值，做 3 日滚动求和
rolling = NTSellProb.rolling_daily_prob(result, window=3)
```

### 使用

```python
from src.national_team.nt_sell_prob import NTSellProb
from src.national_team.ch_client import get_etf_1m, get_index_1m

factor = NTSellProb()
etf_1m = get_etf_1m("510300.SH", "2024-01-01", "2024-06-30")
idx_1m = get_index_1m("000300", "2024-01-01", "2024-06-30")

result = factor.compute(etf_1m, idx_1m)
signals = result.filter(pl.col("nt_sell_prob") > 0.20)
```

---

## 模块文件结构

```
src/national_team/
├── README.md                  ← 本文件
├── __init__.py                ← 模块导出
├── nt_buy_prob.py             ← 买入因子 v2 (NTBuyProb)
├── nt_sell_prob.py            ← 卖出因子 v2 (NTSellProb)
├── ch_client.py               ← ClickHouse 查询层
├── etf_resonance.py           ← ETF 舰队共振（v1 遗留，buy v2 内置）
├── etf_sell_resonance.py      ← ETF 卖出共振（v1 遗留，sell v2 内置）
├── spread_spike.py            ← ETF-指数剪刀差（v1 遗留，v2 内置）
├── ts_client.py               ← Tushare API 客户端
└── market_constraints.yaml    ← A 股执行模型约束配置
```

---

## 回测脚本

| 脚本 | 说明 | 输出 |
|------|------|------|
| `scripts/backtest_nt_buy_signals.py` | 买入信号 15 年回测 | `doc/nt_buy_signals_backtest_v2.md` |
| `scripts/backtest_nt_sell_signals.py` | 卖出信号 15 年回测 | `doc/nt_sell_signals_backtest_v2.md` |

运行方式：

```bash
cd /home/autumn/quant/stock
.venv/bin/python scripts/backtest_nt_buy_signals.py
.venv/bin/python scripts/backtest_nt_sell_signals.py
```

---

## v1 → v2 变更对照

### 买入因子

| v1 特征 | v2 替代 | 改进原因 |
|---------|---------|---------|
| `macro_bear` (单日日内跌幅) | `stress_context` (多日 MA 偏离 + 回撤) | 单日跌幅不够稳健，多日指标更能反映趋势压力 |
| `vol_zscore` (原始 z-score) | `vol_shock` (log + 季节性标准化) | log 变换使分布对称，时间切片消除日内模式 |
| `price_resilience` (布尔/弱连续) | `absorption` (vol_shock × alpha) | 量化"被托住"的强度而非仅判断有/无 |
| `resonance_count` (放量+收阳) | `resonance_count` (vol_shock>q99 + alpha>0) | 阈值统一为 z-score，加入 alpha 过滤假阳性 |
| `spread_zscore_max` (z-scored) | `lead_gap` (原始剪刀差) | 原始值更直观，避免 z-score 延迟 |

### 卖出因子

| v1 特征 | v2 替代 | 改进原因 |
|---------|---------|---------|
| `vol_price_divergence` (15m 滚动) | `distribution` (vol_shock × |-alpha|) | 与 buy 侧 absorption 对称，更直接 |
| `ceiling_hit` (布尔计数) | `wick_ratio` (连续 0~1) | 连续值比布尔更细腻 |
| — (无) | `vwap_fail` | 新增：VWAP 失败是重要的卖方信号 |
| — (无) | `lead_reversal` | 新增：领先反转捕获机构撤退瞬间 |
| — (无) | `propagation_fail` | 新增：传导断裂 = 放量无效 = 出货 |

### 组合方法

| 项目 | v1 | v2 |
|------|----|----|
| 组合方式 | 贝叶斯似然比 | 线性加权 + sigmoid |
| 权重来源 | 条件概率统计 | 手动设置（后续可接 isotonic 校准）|
| 输出 | 后验概率 | sigmoid 概率 |

---

## 可调参数

### NTBuyProb

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `vol_lookback` | 20 | vol_shock 回看天数 |
| `stress_ma_window` | 20 | 均线偏离窗口 |
| `stress_drawdown_days` | 5 | 回撤观察天数 |
| `stress_ma_dev_thresh` | -0.03 | 均线偏离阈值 |
| `stress_drawdown_thresh` | -0.05 | 回撤阈值 |
| `vol_shock_q99` | 2.326 | 共振 vol_shock 阈值 |
| `w_stress` / `w_vol_shock` / `w_absorption` / `w_resonance` / `w_lead_gap` | 2.0 / 1.5 / 2.0 / 1.5 / 1.0 | sigmoid 权重 |
| `bias` | -5.0 | sigmoid 偏置 |

### NTSellProb

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `vol_lookback` | 20 | vol_shock 回看天数 |
| `prop_fail_window` | 5 | 传导断裂回看 bar 数 |
| `prop_fail_vol_q` | 2.326 | 传导断裂 vol_shock 阈值 (q99) |
| `w_distribution` / `w_wick` / `w_vwap_fail` / `w_lead_rev` / `w_prop_fail` | 2.0 / 1.5 / 1.5 / 2.0 / 1.5 | sigmoid 权重 |
| `bias` | -5.0 | sigmoid 偏置 |

---

## 后续计划

1. **标签生成**：基于执行端（高冲击比率股票篮子）生成真实买卖标签
2. **权重校准**：用历史标签 + isotonic regression 替代手动权重
3. **全策略集成**：结合 `market_constraints.yaml` 执行模型，输出实际仓位建议
