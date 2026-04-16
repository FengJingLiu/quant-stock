# AKQuant Indicator Compatibility and Migration Overview

This page is maintained in Chinese first.
English page provides a concise status summary for quick navigation.

- Chinese source: [AKQuant 指标兼容与迁移总览（用户版）](../../zh/advanced/talib_top20_plan.md)

## Update Metadata

- Last updated (UTC+8): 2026-03-13 00:00
- Scope version: plan-baseline-v1
- Data source scope: current repository trunk and regression tests, mainly `src/indicators.rs`, `python/akquant/talib/funcs.py`, `tests/test_talib_backend.py`, and `tests/test_talib_compat.py`.

## Current Status

- `akquant.talib` supports `backend=auto/python/rust`.
- Top20 migration indicators are fully available on both backends.
- Extended batches are completed through batch `T`.
- Total supported indicators: **103**.

## Recommended Migration Flow

1. Align baseline with `backend="python"`.
2. Switch to high-performance backend and verify signal consistency.
3. Check warmup segments and tuple unpacking order for multi-output indicators.

## Where to Read Next

- [AKQuant Indicator Reference (103)](../guide/rust_indicator_reference.md)
- [Indicator Scenario Quickref](../guide/indicator_scenario_quickref.md)
- [Indicator Playbook](../guide/talib_indicator_playbook.md)
