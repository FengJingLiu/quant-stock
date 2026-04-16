---
name: ashare-data-duckdb-first
description: 当用户在 /home/autumn/quant/stock 里需要 A 股历史行情、最新快照、估值相关字段或统一数据入口时使用本技能。必须先查询本地 DuckDB（/home/autumn/quant/stock/data/duckdb/stock.duckdb），仅在本地缺失、滞后或不足时才回退 AkShare，并先安装 akshare_proxy_patch。
---

# A股数据获取（DuckDB 优先，AkShare 兜底）

这个技能用于稳定获取 A 股股票数据，核心目标是：

1. **优先走本地 Lake/DuckDB**（快、稳、口径一致）；
2. **本地不满足再回退 AkShare**，并且必须先安装 `akshare_proxy_patch`。

## 何时触发

当用户出现以下需求时直接使用本技能：

- “查 A 股历史行情 / K 线 / 日线数据”
- “查某只股票最新价、估值、换手率、市值”
- “从本地 DuckDB 取数，不够再在线补”
- “拉取 600519/000001 这类 A 股代码的数据”

## 查询与分析工作流

推荐按下面顺序执行，保证口径一致且可追踪：

1. **先取数（本地优先）**：历史 `history` 或快照 `latest`
2. **再分析**：基于统一 JSON 输出做指标/因子/报告计算
3. **最后解释结果**：明确数据来源（`source=local|akshare`）与回退原因

> 说明：本技能是数据路由层，不直接做“买卖建议”。分析应由下游脚本完成。

## 执行流程（必须遵守）

### 1) 先标准化请求

- 识别 `symbol`（如 `600519` / `600519.SH` / `sh600519`）
- 识别模式：
  - `history`：历史日线 OHLCV
  - `latest`：最新快照（价格 + 指标）
- 识别日期范围、复权类型（`hfq` / `qfq` / `none`）
- 注意：`--start-date` 与 `--end-date` 必须成对出现；若不传则使用 `--period`

### 2) 本地优先（必须）

使用脚本：

```bash
.venv/bin/python skills/ashare-data-duckdb-first/scripts/ashare_data_router.py 600519 --mode history --period 1y --adjust hfq
```

本地默认读取：

- `data/duckdb/stock.duckdb`
- 视图优先级：
  - history: `v_bar_daily_hfq / v_bar_daily_qfq / v_bar_daily_raw`（按请求与可用性选择）
  - latest: `v_daily_hfq_w_ind_dim` → `v_daily_hfq_w_ind` → bar 视图兜底

并进行充分性检查：

- 行数下限：`--min-rows`
- 时效下限：`--max-local-lag-days`

### 3) 不满足再回退 AkShare

以下情况才允许回退：

- 本地 DB/视图不存在
- 本地查询结果为空
- 本地数据滞后超阈值（`--max-local-lag-days`）
- 本地返回行数少于要求（`--min-rows`）

回退时必须执行：

```python
import os
import akshare_proxy_patch

TOKEN = os.environ["AKSHARE_PROXY_TOKEN"]
PROXY_HOST = os.environ["AKSHARE_PROXY_HOST"]
akshare_proxy_patch.install_patch(PROXY_HOST, TOKEN, retry=30)
```

然后再调用 AkShare 接口（如 `stock_zh_a_hist`, `stock_zh_a_spot_em`）。

## 命令矩阵（.venv 环境）

```bash
# 1) 拉历史日线（默认优先本地）
.venv/bin/python skills/ashare-data-duckdb-first/scripts/ashare_data_router.py 600519 --mode history --period 6m --adjust hfq --pretty

# 2) 拉最新快照（价格+估值）
.venv/bin/python skills/ashare-data-duckdb-first/scripts/ashare_data_router.py 000001 --mode latest --pretty

# 3) 指定日期区间
.venv/bin/python skills/ashare-data-duckdb-first/scripts/ashare_data_router.py 300750 --mode history --start-date 2025-01-01 --end-date 2025-12-31

# 4) 提高本地严格度（至少 120 行，且最多滞后 3 天）
.venv/bin/python skills/ashare-data-duckdb-first/scripts/ashare_data_router.py 600519 --mode history --period 1y --min-rows 120 --max-local-lag-days 3 --pretty

# 5) 强制在线（仅用于对照，不作为默认）
.venv/bin/python skills/ashare-data-duckdb-first/scripts/ashare_data_router.py 600519 --mode history --force-remote --pretty

# 6) 指定自定义 DuckDB 文件
.venv/bin/python skills/ashare-data-duckdb-first/scripts/ashare_data_router.py 600519 --mode latest --db-path /tmp/stock.duckdb --pretty
```

## 输出约定

返回 JSON，关键字段：

- `source`: `"local"` 或 `"akshare"`
- `symbol`: 统一 TS 格式（如 `600519.SH`）
- `request`: 入参回显
- `meta.local_view`: 本地命中的视图（若走本地）
- `meta.adjust_applied`: 实际使用的复权口径（`hfq/qfq/none`）
- `meta.fallback_reason`: 回退原因（若走 AkShare）

`latest` 模式下统一返回：`total_mv` / `circ_mv` 与 `total_mv_10k` / `circ_mv_10k`（便于不同下游使用）。

### 输出示例（摘要）

```json
{
  "source": "local",
  "symbol": "600519.SH",
  "mode": "history",
  "meta": {
    "local_view": "v_bar_daily_hfq",
    "rows": 180
  }
}
```

## 常见回退原因

| fallback_reason 样例 | 含义 |
|---|---|
| `duckdb_not_found:...` | 本地库文件不存在 |
| `local_no_sufficient_history:...` | 历史视图存在但数据不足/滞后 |
| `local_no_sufficient_latest:...` | 快照视图存在但数据不足/滞后 |
| `local_query_error:...` | 本地查询异常 |
| `force_remote` | 显式要求跳过本地 |

## 失败处理

- 若本地与在线都失败，返回 `error`，并附带 `fallback_reason`。
- 避免静默失败，必须给出可追踪原因。

## 排错清单

1. 先检查 DuckDB 文件是否存在：`/home/autumn/quant/stock/data/duckdb/stock.duckdb`
2. 再检查本地是否过旧：调大 `--max-local-lag-days` 做对照
3. 若回退 AkShare 失败，确认代理补丁参数（环境变量）：
   - `AKSHARE_PROXY_TOKEN`
   - `AKSHARE_PROXY_HOST`
   - `retry = 30`
4. `--force-remote` 仅用于对照，不应作为默认数据路径

## 数据增强（可选）

当你拿到本技能输出后，如果还需要资金流、财务报表、宏观指标等扩展字段，可继续调用：

- `skills/findata-toolkit-cn/SKILL.md`

## 重要注意事项

- 使用仓库 `.venv` Python 环境。
- 临时查询脚本放在 `/home/autumn/quant/stock/scripts/`。
- 读取数据时优先本地 DuckDB，只有不满足需求才回退 AkShare。

> **免责声明**：本技能基于本地 DuckDB 与 AkShare 公开数据，可能存在数据滞后、接口变更或临时不可用。输出仅用于信息参考与研究，不构成任何投资建议。
