# Project Memory

- 2026-03-30: For the small-cap backtest based on `doc/小市值.md`, user chose the local-data-priority variant:
  use local DuckDB/Lake data for stock selection, allow online补数 only for the CSI 1000 timing index if needed, and do not require historical ROE/revenue filters.
- 2026-03-30: For the next small-cap optimization round, missing historical fields may be fetched online with AKShare and should be treated as acceptable inputs rather than blockers.
