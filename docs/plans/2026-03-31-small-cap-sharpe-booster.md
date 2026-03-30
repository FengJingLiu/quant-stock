# Small-Cap Sharpe Booster Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a new `sharpe_booster` ablation mode that keeps the `full_combo` structure but adds a raw-price floor and a low-volatility exclusion filter.

**Architecture:** Extend the existing single-runner script instead of creating a new file. Enrich the weekly snapshot query with raw close and 20-day realized volatility, then apply the extra filters only when the selected mode is `sharpe_booster`.

**Tech Stack:** Python 3.13, DuckDB, pandas, AKQuant, unittest

---

### Task 1: Add failing tests for raw-price floor and low-volatility exclusion

**Files:**
- Modify: `tests/test_backtest_small_cap_lake.py`
- Modify: `scripts/backtest_small_cap_lake.py`
- Test: `tests/test_backtest_small_cap_lake.py`

**Step 1: Write the failing test**

Add tests for:
- candidates with `raw_close < 2.5` are excluded
- the highest-volatility 20% names are excluded from the final candidate pool
- `resolve_mode_configs("all")` now includes `sharpe_booster`

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=/home/autumn/quant/stock .venv/bin/python -m unittest tests.test_backtest_small_cap_lake -v`
Expected: FAIL because the new fields and mode are not implemented yet

**Step 3: Write minimal implementation**

Implement helper behavior for:
- raw-price filter
- volatility percentile filter
- mode expansion

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=/home/autumn/quant/stock .venv/bin/python -m unittest tests.test_backtest_small_cap_lake -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_backtest_small_cap_lake.py scripts/backtest_small_cap_lake.py
git commit -m "test: add sharpe booster filter coverage"
```

### Task 2: Extend weekly snapshot loading with raw close and vol20

**Files:**
- Modify: `scripts/backtest_small_cap_lake.py`
- Modify: `tests/test_backtest_small_cap_lake.py`
- Test: `tests/test_backtest_small_cap_lake.py`

**Step 1: Write the failing test**

Add tests for:
- `rank_signal_snapshot` respects `raw_close`
- `vol20` is available for downstream filtering

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=/home/autumn/quant/stock .venv/bin/python -m unittest tests.test_backtest_small_cap_lake -v`
Expected: FAIL on missing snapshot fields or missing filter logic

**Step 3: Write minimal implementation**

Modify the DuckDB query to include:
- `raw_close`
- 20-day return volatility

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=/home/autumn/quant/stock .venv/bin/python -m unittest tests.test_backtest_small_cap_lake -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_backtest_small_cap_lake.py scripts/backtest_small_cap_lake.py
git commit -m "feat: expose raw price and vol20 in weekly snapshots"
```

### Task 3: Wire sharpe_booster into mode execution

**Files:**
- Modify: `scripts/backtest_small_cap_lake.py`
- Modify: `tests/test_backtest_small_cap_lake.py`
- Test: `tests/test_backtest_small_cap_lake.py`

**Step 1: Write the failing test**

Add tests for:
- `sharpe_booster` inherits `full_combo` behavior flags
- `sharpe_booster` applies extra raw-price and low-vol filters

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=/home/autumn/quant/stock .venv/bin/python -m unittest tests.test_backtest_small_cap_lake -v`
Expected: FAIL on mode config or candidate filtering

**Step 3: Write minimal implementation**

Implement:
- new mode config
- additional filter application in ranked snapshots / final candidate pool
- output suffix handling for `sharpe_booster`

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=/home/autumn/quant/stock .venv/bin/python -m unittest tests.test_backtest_small_cap_lake -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_backtest_small_cap_lake.py scripts/backtest_small_cap_lake.py
git commit -m "feat: add sharpe booster mode"
```

### Task 4: Run ablation and compare sharpe_booster vs full_combo

**Files:**
- Modify: `scripts/backtest_small_cap_lake.py` if runtime reveals issues
- Output: `data/backtest_small_cap_lake_*_sharpe_booster.csv`
- Output: `data/backtest_small_cap_lake_metrics_summary.csv`

**Step 1: Run short-window smoke test**

```bash
cd /home/autumn/quant/stock
PYTHONPATH=/home/autumn/quant/stock .venv/bin/python scripts/backtest_small_cap_lake.py --start-date 2024-01-01 --end-date 2024-03-31 --initial-cash 200000 --mode all
```

**Step 2: Run full-window ablation**

```bash
PYTHONPATH=/home/autumn/quant/stock .venv/bin/python scripts/backtest_small_cap_lake.py --start-date 2010-01-01 --initial-cash 200000 --mode all
```

**Step 3: Verify summary comparison**

Check whether `sharpe_booster`:
- keeps trade count close to `full_combo`
- improves or preserves Sharpe
- does not blow up drawdown

**Step 4: Apply minimal fixes if needed**

Only patch behavior proven broken by test or runtime evidence.

**Step 5: Commit**

```bash
git add scripts/backtest_small_cap_lake.py tests/test_backtest_small_cap_lake.py docs/plans/2026-03-31-small-cap-sharpe-booster*.md MEMORY.md
git commit -m "feat: add sharpe booster small-cap mode"
```
