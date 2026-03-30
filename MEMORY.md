# Project Memory

- 2026-03-30: For the small-cap backtest based on `doc/小市值.md`, user chose the local-data-priority variant:
  use local DuckDB/Lake data for stock selection, allow online补数 only for the CSI 1000 timing index if needed, and do not require historical ROE/revenue filters.
- 2026-03-30: For the next small-cap optimization round, missing historical fields may be fetched online with AKShare and should be treated as acceptable inputs rather than blockers.
- 2026-03-31: For the next optimization round after `full_combo`, user prioritizes preserving return profile while improving Sharpe toward 1.0+.
  Preferred changes: add a low-price floor (`close >= 2.5` or similar real-price filter), add a low-volatility filter that removes the top 20% 20-day volatility names in the final candidate pool, and if breadth is used it should be a mild position dampener rather than binary timing. User explicitly does not want a hard single-stock stop-loss.
