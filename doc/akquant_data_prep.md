# A股数据清洗到 Akquant 回测格式

## 1. 一次性解压 + 清洗（后复权日线 + 每日指标）

```bash
.venv/bin/python scripts/prepare_akquant_data.py
```

默认行为：
- 解压到 `data/raw_extracted/price` 与 `data/raw_extracted/indicator`
- 清洗输出到 `data/akquant/daily_hfq_with_indicators`

## 2. 仅清洗（跳过解压）

```bash
.venv/bin/python scripts/prepare_akquant_data.py --skip-extract
```

## 3. 加入技术因子

```bash
.venv/bin/python scripts/prepare_akquant_data.py \
  --tech-factor-zip A股数据_每日指标/技术因子_前复权.zip
```

## 4. 小样本调试

```bash
.venv/bin/python scripts/prepare_akquant_data.py --skip-extract --limit 20
```

## 5. Akquant 回测加载示例

```python
from pathlib import Path
import pandas as pd
from akquant import Strategy, run_backtest

class BuyHold(Strategy):
    def on_bar(self, bar):
        if self.get_position(bar.symbol) == 0:
            self.buy(bar.symbol, 100)

root = Path("data/akquant/daily_hfq_with_indicators")
files = [p for p in root.glob("*.csv") if p.name != "manifest.csv"]

# 示例仅取第一只股票
df = pd.read_csv(files[0], parse_dates=["date", "timestamp"])

result = run_backtest(
    data=df,
    strategy=BuyHold,
    symbol=df["symbol"].iloc[0],
    initial_cash=100000.0,
)

print(result.metrics_df)
```

> 输出数据至少包含 `date/timestamp/symbol/open/high/low/close/volume/amount`，可直接用于 Akquant。
