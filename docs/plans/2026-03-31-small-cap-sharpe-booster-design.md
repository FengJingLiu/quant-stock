# 小市值 Sharpe Booster 设计

**日期**: 2026-03-31

**目标**

在现有 [`full_combo`](/home/autumn/quant/stock/scripts/backtest_small_cap_lake.py) 基础上，新增一个独立的 `sharpe_booster` 消融模式，只引入两项温和过滤：

1. 真实价格低价过滤：`raw_close >= 2.5`
2. 截面低波剔除：在最终候选池中剔除过去 20 日波动率最高的前 20%

目标不是继续大幅压缩换手，而是在尽量保住 `full_combo` 收益结构的前提下，进一步提升夏普并平滑净值曲线。

## 一、 为什么要单独做一个 Booster 模式

现有四组模式结论已经清楚：

- `baseline` 收益高，但摩擦和回撤过重
- `buffer_only`、`buffer_risk` 过度牺牲收益
- `full_combo` 在低换手、低手续费、较低回撤之间取得了最好的平衡

因此下一步不应该推倒重来，而应该在 `full_combo` 的信号结构上，做两把“选股手术刀”，只去除最容易制造净值毛刺的尾部标的。

## 二、 策略改动范围

### 1. 低价过滤

过滤条件：

- `raw_close >= 2.5`

关键点：

- 必须使用真实原始收盘价，不允许使用后复权价或前复权价
- 该字段来自本地 DuckDB 的 [`v_bar_daily_raw`](/home/autumn/quant/stock/scripts/init_duckdb_views_lake.py) 表

原因：

- 近似历史 ST / 退市过滤存在口径缺陷
- 极低价股是面值退市、重大风险、流动性枯竭的高风险代理变量
- 这个过滤极其便宜，且不需要任何在线数据

### 2. 低波剔除

先为每个信号日横截面计算：

- `vol20`: 使用后复权 `close` 计算最近 20 个交易日收益率标准差

在满足 `full_combo` 的最终候选池里，再执行：

- 计算当期候选池内 `vol20` 的横截面分位
- 剔除 `vol20_rank > 0.8` 的股票

保留：

- 波动率更低的 80% 候选

原因：

- `full_combo` 已经通过动量过滤避免了明显左侧飞刀
- 但仍可能保留高波动的游资妖股
- 低波过滤的目的不是追求保守，而是切掉最容易 A 字回落的尾部高波票

## 三、 数据设计

### 1. 信号快照字段扩展

当前周度信号快照已包含：

- `close`（后复权）
- `ma20_stock`
- `pe_ttm`
- `pb`
- `total_mv_10k`
- `list_date`

需要新增：

- `raw_close`：来自 `v_bar_daily_raw.close`
- `ret_1d`：`close / lag(close) - 1`
- `vol20`：最近 20 日 `ret_1d` 的标准差

### 2. 过滤顺序

对于 `sharpe_booster` 模式：

1. 先继承 `full_combo` 的全部条件：
   - Rank Buffer
   - 风险过滤
   - 择时滞后带
   - 动量过滤
2. 再加：
   - `raw_close >= 2.5`
   - 低波剔除（候选池内剔除波动率最高前 20%）

这样可以保证 Booster 的效果可直接归因于这两个新增过滤，而不是和其他逻辑混杂。

## 四、 模式设计

在 [`resolve_mode_configs`](/home/autumn/quant/stock/scripts/backtest_small_cap_lake.py) 中新增：

- `sharpe_booster`

参数特征：

- 继承 `full_combo`
- `price_floor = 2.5`
- `low_vol_enabled = True`
- `low_vol_exclude_pct = 0.2`
- 不启用额外广度控仓
- 不启用硬止损

## 五、 验收标准

- `sharpe_booster` 模式可独立运行
- 输出新增：
  - `*_metrics_sharpe_booster.csv`
  - `*_trades_sharpe_booster.csv`
  - `*_weekly_picks_sharpe_booster.csv`
  - `*_equity_sharpe_booster.csv`
- 汇总表新增 `sharpe_booster`
- 消融结果可直接与 `full_combo` 对比：
  - `trade_count`
  - `total_return_pct`
  - `max_drawdown_pct`
  - `sharpe_ratio`
  - `total_commission`

## 六、 预期

合理预期不是“收益暴增”，而是：

- 交易次数与 `full_combo` 接近
- 手续费变化不大
- 极端回撤进一步收敛
- 夏普有机会进一步提升

若 Booster 效果不佳，也必须保留结果，作为失败消融样本，而不是覆盖 `full_combo`。
