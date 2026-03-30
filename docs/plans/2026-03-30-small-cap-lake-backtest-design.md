# 小市值本地 Lake 回测设计

**日期**: 2026-03-30

**目标**

基于 [`doc/小市值.md`](/home/autumn/quant/stock/doc/%E5%B0%8F%E5%B8%82%E5%80%BC.md) 实现一个“本地数据优先版”小市值回测脚本，使用本地 DuckDB/Lake 完成股票横截面筛选，仅在本地缺失时在线补充中证 1000 指数日线，用 `AKQuant` 回测 `2010-01-01` 至最新交易日、`200000` 初始资金的策略表现。

**范围**

- 纳入：
  - 小市值选股
  - 行业内 `PE/PB` 相对低估分位过滤
  - 次新过滤
  - 中证 1000 `MA20` 择时
  - 每周轮动、等权持仓
  - 结果落盘
- 不纳入：
  - 历史 `ROE/营业收入` 过滤
  - 历史时点 `ST/退市` 状态过滤

**为什么不做完整版**

当前本地 Lake 具备以下历史字段：

- 行情：`open/high/low/close/volume`
- 估值与市值：`pe_ttm/pb/total_mv_10k/circ_mv_10k`
- 维表：`industry/list_date/is_delisted/name`

但缺少：

- 历史时点 `ROE`
- 历史时点 `营业收入`
- 中证 1000 指数本地日线
- 历史时点 `ST/退市整理` 状态

因此如果强行复刻原文“完整版”，会引入大量在线依赖和未来函数风险，不符合“本地数据优先”的选择。

**策略定义**

1. 每周以“该周最后一个交易日”为信号日。
2. 在信号日，对全市场股票使用本地 `v_daily_hfq_w_ind_dim` 做筛选：
   - `pe_ttm > 0`
   - `pb > 0`
   - `industry` 非空
   - `total_mv_10k` 非空
   - `list_date <= signal_date - 250天`
3. 在行业内分别计算：
   - `pe_rank = rank(pe_ttm, ascending=True, pct=True)`
   - `pb_rank = rank(pb, ascending=True, pct=True)`
4. 仅保留 `pe_rank <= 0.4` 且 `pb_rank <= 0.4` 的股票。
5. 在剩余股票中按 `total_mv_10k` 升序排序，取前 `10` 只作为下周目标持仓。
6. 使用中证 1000 指数 `MA20` 作为择时：
   - 若信号日收盘 `< MA20`，则下周空仓。
   - 否则按目标池等权持仓。
7. 实际交易日使用“下一个交易日开盘”，通过 `AKQuant ExecutionMode.NextOpen` 落地。

**数据流**

1. DuckDB 读取股票历史横截面，计算周度选股结果。
2. Akshare 拉取中证 1000 历史日线，并缓存到 `data/cache/index_000852.csv`。
3. 合并股票周度选股与指数择时结果，生成 `rebalance_date -> symbols` 的周度计划。
4. 从本地 `v_bar_daily_hfq` 读取所有入选股票所需行情，构造成 `dict[str, DataFrame]`。
5. 用 `AKQuant` 执行“按计划调仓”的简单策略。

**执行口径**

- 资金：`200000`
- 手续费：
  - `commission_rate=0.0003`
  - `stamp_tax_rate=0.0005`
  - `min_commission=5`
- 滑点：`0.002`
- 市场规则：`t_plus_one=True`
- 时区：`Asia/Shanghai`

**风险与偏差**

- 缺少历史 `ST/退市` 状态，若用当前维表过滤历史会产生未来函数；本版不使用该条件。
- 行业字段取自当前维表映射，历史行业迁移无法完全还原。
- 指数择时依赖在线补数一次，但会缓存，后续可复现。
- 小市值策略容量低，回测结果只适用于小资金口径。

**输出**

- `data/backtest_small_cap_lake_metrics.csv`
- `data/backtest_small_cap_lake_trades.csv`
- `data/backtest_small_cap_lake_weekly_picks.csv`
- `data/backtest_small_cap_lake_equity.csv`

**验收标准**

- 脚本可在 `.venv` 环境运行。
- 能成功生成 `2010-至今` 的周度计划和回测结果。
- 输出总收益、年化、最大回撤、夏普等核心指标。
- 结果文件完整落盘。
