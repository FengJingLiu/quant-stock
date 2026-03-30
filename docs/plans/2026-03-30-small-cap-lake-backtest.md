# Small-Cap Lake Backtest Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a local-data-priority small-cap backtest script based on DuckDB/Lake stock data and a cached CSI 1000 timing index, then run it from 2010 to the latest available date with 200,000 initial cash.

**Architecture:** The script will compute weekly rebalance targets in DuckDB first, using local valuation and market-cap history. It will fetch and cache only the CSI 1000 timing series online, then hand a `rebalance_date -> symbols` plan plus local bar data into an `AKQuant` strategy that rebalances at next open.

**Tech Stack:** Python 3.13, DuckDB, pandas, Akshare with `akshare_proxy_patch`, AKQuant, unittest

---

### Task 1: Add failing tests for weekly selection and timing plan

**Files:**
- Create: `tests/test_backtest_small_cap_lake.py`
- Modify: none
- Test: `tests/test_backtest_small_cap_lake.py`

**Step 1: Write the failing test**

```python
def test_build_weekly_selection_keeps_industry_relative_value_and_smallest_market_cap():
    ...

def test_build_rebalance_plan_clears_targets_when_index_below_ma20():
    ...
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=/home/autumn/quant/stock .venv/bin/python -m unittest tests.test_backtest_small_cap_lake -v`
Expected: FAIL with import or missing function errors

**Step 3: Write minimal implementation**

Create the smallest importable module and placeholder functions in `scripts/backtest_small_cap_lake.py`.

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=/home/autumn/quant/stock .venv/bin/python -m unittest tests.test_backtest_small_cap_lake -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_backtest_small_cap_lake.py scripts/backtest_small_cap_lake.py
git commit -m "test: add small-cap backtest planning tests"
```

### Task 2: Implement local selection, weekly scheduling, and index cache helpers

**Files:**
- Modify: `scripts/backtest_small_cap_lake.py`
- Test: `tests/test_backtest_small_cap_lake.py`

**Step 1: Write the failing test**

Add tests for:
- weekly rebalance dates use next trading day after signal day
- cache loader normalizes date/close series
- price-data loader returns AKQuant-ready dataframes

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=/home/autumn/quant/stock .venv/bin/python -m unittest tests.test_backtest_small_cap_lake -v`
Expected: FAIL on new helper expectations

**Step 3: Write minimal implementation**

Implement:
- index cache read/write helpers
- weekly selection query builder
- rebalance plan builder
- local bar loader

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=/home/autumn/quant/stock .venv/bin/python -m unittest tests.test_backtest_small_cap_lake -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_backtest_small_cap_lake.py scripts/backtest_small_cap_lake.py
git commit -m "feat: implement small-cap weekly planning helpers"
```

### Task 3: Implement AKQuant strategy and CLI entrypoint

**Files:**
- Modify: `scripts/backtest_small_cap_lake.py`
- Test: `tests/test_backtest_small_cap_lake.py`

**Step 1: Write the failing test**

Add a test for the strategy day-switch behavior:

```python
def test_strategy_switches_to_daily_targets_from_plan():
    ...
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=/home/autumn/quant/stock .venv/bin/python -m unittest tests.test_backtest_small_cap_lake -v`
Expected: FAIL because strategy methods are incomplete

**Step 3: Write minimal implementation**

Implement:
- `WeeklyPlanStrategy`
- CLI args
- orchestration in `main()`
- result exports

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=/home/autumn/quant/stock .venv/bin/python -m unittest tests.test_backtest_small_cap_lake -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_backtest_small_cap_lake.py scripts/backtest_small_cap_lake.py
git commit -m "feat: add small-cap lake backtest runner"
```

### Task 4: Run end-to-end backtest and verify outputs

**Files:**
- Modify: `scripts/backtest_small_cap_lake.py` if needed
- Output: `data/backtest_small_cap_lake_metrics.csv`
- Output: `data/backtest_small_cap_lake_trades.csv`
- Output: `data/backtest_small_cap_lake_weekly_picks.csv`
- Output: `data/backtest_small_cap_lake_equity.csv`

**Step 1: Run the backtest**

```bash
cd /home/autumn/quant/stock
PYTHONPATH=/home/autumn/quant/stock .venv/bin/python scripts/backtest_small_cap_lake.py --start-date 2010-01-01 --initial-cash 200000
```

**Step 2: Verify outputs**

Confirm result files exist and inspect summary metrics.

**Step 3: If needed, make minimal fixes**

Only patch issues found during the real backtest run.

**Step 4: Re-run verification**

Repeat the backtest command until it succeeds cleanly.

**Step 5: Commit**

```bash
git add scripts/backtest_small_cap_lake.py tests/test_backtest_small_cap_lake.py docs/plans/2026-03-30-small-cap-lake-backtest*.md
git commit -m "feat: add local small-cap backtest workflow"
```
