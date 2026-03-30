# Small-Cap Lake Optimization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Upgrade the local small-cap backtest with rank buffering, Akshare-backed historical quality filters, timing hysteresis, momentum filters, and ablation modes.

**Architecture:** Keep a single runner script and extend it with cached online enrichment for missing historical fields. Use DuckDB for local cross-section assembly, then apply a stateful weekly portfolio update layer that separates buy thresholds from sell thresholds.

**Tech Stack:** Python 3.13, DuckDB, pandas, Akshare with proxy patch, AKQuant, unittest, parquet cache files

---

### Task 1: Add failing tests for rank buffer and timing hysteresis

**Files:**
- Modify: `tests/test_backtest_small_cap_lake.py`
- Modify: `scripts/backtest_small_cap_lake.py`
- Test: `tests/test_backtest_small_cap_lake.py`

**Step 1: Write the failing test**

Add tests for:
- holdings survive when rank slips from buy zone to buffer zone
- holdings are removed when they fall outside sell buffer
- timing hysteresis keeps previous state in the neutral band

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=/home/autumn/quant/stock .venv/bin/python -m unittest tests.test_backtest_small_cap_lake -v`
Expected: FAIL on missing or incorrect buffer/hysteresis functions

**Step 3: Write minimal implementation**

Implement helper functions for:
- market timing state machine
- buffered hold/replace portfolio update

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=/home/autumn/quant/stock .venv/bin/python -m unittest tests.test_backtest_small_cap_lake -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_backtest_small_cap_lake.py scripts/backtest_small_cap_lake.py
git commit -m "test: add buffer and timing hysteresis coverage"
```

### Task 2: Add failing tests for momentum and historical quality filters

**Files:**
- Modify: `tests/test_backtest_small_cap_lake.py`
- Modify: `scripts/backtest_small_cap_lake.py`
- Test: `tests/test_backtest_small_cap_lake.py`

**Step 1: Write the failing test**

Add tests for:
- candidate selection requires `close > ma20_stock`
- candidate selection requires visible `roe > 0`
- candidate selection excludes `is_st_like = True`

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=/home/autumn/quant/stock .venv/bin/python -m unittest tests.test_backtest_small_cap_lake -v`
Expected: FAIL on new filter expectations

**Step 3: Write minimal implementation**

Extend selection helpers to incorporate:
- momentum
- roe visibility
- risk flags

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=/home/autumn/quant/stock .venv/bin/python -m unittest tests.test_backtest_small_cap_lake -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_backtest_small_cap_lake.py scripts/backtest_small_cap_lake.py
git commit -m "feat: add momentum and quality filters"
```

### Task 3: Add cache builders for Akshare-backed quality and risk data

**Files:**
- Modify: `scripts/backtest_small_cap_lake.py`
- Modify: `tests/test_backtest_small_cap_lake.py`
- Test: `tests/test_backtest_small_cap_lake.py`

**Step 1: Write the failing test**

Add tests for:
- financial cache row normalization
- announcement-date fallback logic
- risk flag cache merge behavior

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=/home/autumn/quant/stock .venv/bin/python -m unittest tests.test_backtest_small_cap_lake -v`
Expected: FAIL on missing cache helper behavior

**Step 3: Write minimal implementation**

Implement:
- parquet cache load/save helpers
- Akshare fetch normalization
- visible-as-of-date merge logic for financial quality
- risk-flag join preparation

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=/home/autumn/quant/stock .venv/bin/python -m unittest tests.test_backtest_small_cap_lake -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_backtest_small_cap_lake.py scripts/backtest_small_cap_lake.py
git commit -m "feat: cache historical quality and risk flags"
```

### Task 4: Add ablation modes and end-to-end runner changes

**Files:**
- Modify: `scripts/backtest_small_cap_lake.py`
- Modify: `tests/test_backtest_small_cap_lake.py`
- Test: `tests/test_backtest_small_cap_lake.py`

**Step 1: Write the failing test**

Add tests for:
- ablation config parsing
- output suffix naming
- daily plan generation in `baseline`, `buffer_only`, `buffer_risk`, `full_combo`

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=/home/autumn/quant/stock .venv/bin/python -m unittest tests.test_backtest_small_cap_lake -v`
Expected: FAIL on missing ablation wiring

**Step 3: Write minimal implementation**

Implement:
- mode config presets
- per-mode output naming
- end-to-end orchestration for one mode or all modes

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=/home/autumn/quant/stock .venv/bin/python -m unittest tests.test_backtest_small_cap_lake -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_backtest_small_cap_lake.py scripts/backtest_small_cap_lake.py
git commit -m "feat: add ablation modes to small-cap backtest"
```

### Task 5: Run optimized ablation backtests and verify output tables

**Files:**
- Modify: `scripts/backtest_small_cap_lake.py` if needed
- Output: `data/backtest_small_cap_lake_*`

**Step 1: Run the full ablation**

```bash
cd /home/autumn/quant/stock
PYTHONPATH=/home/autumn/quant/stock .venv/bin/python scripts/backtest_small_cap_lake.py --start-date 2010-01-01 --initial-cash 200000 --mode all
```

**Step 2: Verify outputs**

Check:
- baseline outputs still exist
- optimized mode outputs exist
- summary comparison table exists

**Step 3: Make minimal fixes if runtime reveals issues**

Only patch behavior proven broken by the end-to-end run.

**Step 4: Re-run verification**

Repeat the full ablation command until it succeeds cleanly.

**Step 5: Commit**

```bash
git add scripts/backtest_small_cap_lake.py tests/test_backtest_small_cap_lake.py docs/plans/2026-03-30-small-cap-lake-optimization*.md MEMORY.md
git commit -m "feat: optimize small-cap backtest with buffered rotation"
```
