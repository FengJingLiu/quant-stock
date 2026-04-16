# AKQuant Capability Boost Plan (User-facing)

This page is maintained in Chinese first.
English page provides a concise capability board and next-focus items.

- Chinese source: [AKQuant 能力补强计划（用户视角）](../../zh/advanced/capability_boost_plan.md)

## Update Metadata

- Last updated (UTC+8): 2026-03-13 00:00
- Scope version: plan-baseline-v1
- Data source scope: current repository implementation and tests, mainly under `src/`, `python/akquant/`, `tests/`, and related advanced docs.

## Capability Board

| Area | Current State | Next Focus |
| :--- | :--- | :--- |
| Indicator compatibility | Top20 done, extended batches through `T`, 103 indicators total | More scenario templates and teaching materials |
| Complex orders | Strategy-level helpers are available (`OCO/Bracket/Trailing`) | Move to engine-native order graph |
| Data adapters & MTF | `DataFeedAdapter`, `CSV/Parquet`, `resample/replay/align` are available | More official adapters and validation tools |
| Broker extensibility | Local broker adapters + registry are in place | First international broker minimum loop |
| Analyzer plugins | Plugin protocol and backtest integration are available | Packaging/distribution and version constraints |
| Streaming kernel | Unified entry with event-based flow is available | Better observability and production diagnostics |

## Recommended Reading

- [AKQuant Indicator Reference](../guide/rust_indicator_reference.md)
- [Indicator Scenario Quickref](../guide/indicator_scenario_quickref.md)
- [Indicator Playbook](../guide/talib_indicator_playbook.md)
