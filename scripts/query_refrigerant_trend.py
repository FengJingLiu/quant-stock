#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import duckdb


DB_PATH = Path("/home/autumn/quant/stock/data/duckdb/stock.duckdb")


@dataclass(frozen=True)
class Target:
    symbol: str
    name: str


TARGETS = [
    Target("600160.SH", "巨化股份"),
    Target("603379.SH", "三美股份"),
    Target("605020.SH", "永和股份"),
    Target("600673.SH", "东阳光"),
    Target("002407.SZ", "多氟多"),
    Target("002915.SZ", "中欣氟材"),
]


def pct(cur: float | None, base: float | None) -> float | None:
    if cur is None or base is None or base == 0:
        return None
    return (cur / base - 1.0) * 100.0


def main() -> None:
    con = duckdb.connect(str(DB_PATH))
    symbols = [t.symbol for t in TARGETS]
    in_clause = ",".join(f"'{s}'" for s in symbols)

    sql = f"""
    WITH ranked AS (
      SELECT
        symbol,
        date,
        close,
        amount,
        row_number() OVER (PARTITION BY symbol ORDER BY date DESC) AS rn,
        avg(amount) OVER (
          PARTITION BY symbol
          ORDER BY date
          ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
        ) AS amount_ma20
      FROM v_bar_daily_hfq
      WHERE symbol IN ({in_clause})
    ), agg AS (
      SELECT
        symbol,
        max(CASE WHEN rn = 1 THEN date END) AS latest_date,
        max(CASE WHEN rn = 1 THEN close END) AS close_latest,
        max(CASE WHEN rn = 6 THEN close END) AS close_5d,
        max(CASE WHEN rn = 21 THEN close END) AS close_20d,
        max(CASE WHEN rn = 61 THEN close END) AS close_60d,
        max(CASE WHEN rn = 1 THEN amount END) AS amount_latest,
        max(CASE WHEN rn = 1 THEN amount_ma20 END) AS amount_ma20_latest
      FROM ranked
      GROUP BY symbol
    )
    SELECT
      a.symbol,
      a.latest_date,
      a.close_latest,
      a.close_5d,
      a.close_20d,
      a.close_60d,
      a.amount_latest,
      a.amount_ma20_latest,
      d.name,
      d.industry
    FROM agg a
    LEFT JOIN v_dim_symbol d ON a.symbol = d.symbol
    ORDER BY a.symbol;
    """

    rows = con.execute(sql).fetchall()
    cols = [d[0] for d in con.description]
    data = [dict(zip(cols, row)) for row in rows]

    for item in data:
        item["ret_5d_pct"] = pct(item["close_latest"], item["close_5d"])
        item["ret_20d_pct"] = pct(item["close_latest"], item["close_20d"])
        item["ret_60d_pct"] = pct(item["close_latest"], item["close_60d"])
        item["amount_vs_ma20"] = (
            item["amount_latest"] / item["amount_ma20_latest"]
            if item["amount_latest"] and item["amount_ma20_latest"]
            else None
        )

    payload = {
        "source": "local_duckdb",
        "db_path": str(DB_PATH),
        "universe": [t.__dict__ for t in TARGETS],
        "rows": len(data),
        "data": data,
    }
    print(json.dumps(payload, ensure_ascii=False, default=str, indent=2))


if __name__ == "__main__":
    main()
