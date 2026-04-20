#!/usr/bin/env python3
"""
A股多因子选股模型
================
使用本地DuckDB数据，基于以下因子维度进行选股：

1. 估值因子（Value Factors）
   - PE_TTM（市盈率TTM）
   - PB（市净率）
   - PS_TTM（市销率TTM）
   - Dividend Yield（股息率）

2. 质量因子（Quality Factors）
   - ROE（净资产收益率）- 需从财务数据获取
   - Gross Margin（毛利率）
   - Debt Ratio（资产负债率）
   - Cash Flow（经营现金流）

3. 动量因子（Momentum Factors）
   - Price vs MA120（价格相对于120日均线的位置）
   - 20日涨跌幅
   - 60日涨跌幅

4. 流动性因子（Liquidity Factors）
   - Market Cap（总市值）
   - Turnover Rate（换手率）
   - Volume Ratio（量比）

评分规则：各因子标准化后加权计算综合得分
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional
import warnings

import duckdb
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# 项目根目录
PROJECT_ROOT = Path("/home/autumn/quant/stock")
DUCKDB_PATH = PROJECT_ROOT / "data" / "duckdb" / "stock.duckdb"


@dataclass
class FactorWeights:
    """因子权重配置"""

    # 估值因子权重（总分40分）
    pe_weight: float = 10.0  # PE越低越好
    pb_weight: float = 10.0  # PB越低越好
    ps_weight: float = 10.0  # PS越低越好
    dividend_weight: float = 10.0  # 股息率越高越好

    # 动量因子权重（总分30分）
    momentum_20d_weight: float = 15.0  # 20日动量
    momentum_60d_weight: float = 15.0  # 60日动量

    # 流动性因子权重（总分20分）
    turnover_weight: float = 10.0  # 换手率适中
    volume_ratio_weight: float = 10.0  # 量比

    # 质量因子权重（总分10分）- 基础版使用可用代理变量
    stability_weight: float = 10.0  # 价格稳定性


@dataclass
class FilterCriteria:
    """筛选条件配置"""

    # 基础筛选
    min_market_cap: float = 30.0  # 最小市值30亿
    max_market_cap: float = 2000.0  # 最大市值2000亿
    min_price: float = 3.0  # 最小股价3元
    min_turnover: float = 0.2  # 最小换手率0.2%

    # 估值筛选
    max_pe: float = 50.0  # 最大PE
    min_pe: float = 0.0  # 排除负PE（亏损股）
    max_pb: float = 5.0  # 最大PB
    max_ps: float = 10.0  # 最大PS

    # 流动性筛选
    min_volume_ratio: float = 0.5  # 最小量比

    # 动量筛选
    max_price_vs_ma120: float = 1.5  # 价格不能超过MA120的50%
    min_price_vs_ma120: float = 0.5  # 价格不能低于MA120的50%


class MultiFactorScreener:
    """多因子选股器"""

    def __init__(
        self,
        weights: Optional[FactorWeights] = None,
        criteria: Optional[FilterCriteria] = None,
        db_path: Optional[Path] = None,
    ):
        self.weights = weights or FactorWeights()
        self.criteria = criteria or FilterCriteria()
        self.db_path = db_path or DUCKDB_PATH
        self.conn: Optional[duckdb.DuckDBPyConnection] = None

    def connect(self):
        """连接数据库"""
        os.chdir(PROJECT_ROOT)
        self.conn = duckdb.connect(str(self.db_path), read_only=True)
        return self

    def close(self):
        """关闭连接"""
        if self.conn:
            self.conn.close()
            self.conn = None

    def __enter__(self):
        return self.connect()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def get_latest_date(self) -> str:
        """获取最新交易日期"""
        result = self.conn.execute(
            "SELECT MAX(date) as max_date FROM v_daily_hfq_w_ind_dim"
        ).fetchone()
        return result[0]

    def get_stock_universe(self, date: Optional[str] = None) -> pd.DataFrame:
        """获取基础股票池"""
        if date is None:
            date = self.get_latest_date()

        query = f"""
        SELECT 
            d.symbol,
            s.name,
            s.industry,
            s.exchange,
            b.close as price,
            d.pe_ttm,
            d.pb,
            d.ps_ttm,
            d.turnover_rate,
            d.volume_ratio,
            d.total_mv_10k as market_cap_10k,
            d.circ_mv_10k as circulating_cap_10k,
            d.dividend_yield_ttm,
            d.date
        FROM v_indicator_daily d
        LEFT JOIN v_dim_symbol s ON d.symbol = s.symbol
        LEFT JOIN v_bar_daily_raw b ON d.symbol = b.symbol AND d.date = b.date
        WHERE d.date = '{date}'
          AND s.is_delisted = false  -- 排除退市股票
          AND b.close >= {self.criteria.min_price}
          AND d.total_mv_10k >= {self.criteria.min_market_cap * 10000}
          AND d.total_mv_10k <= {self.criteria.max_market_cap * 10000}
          AND d.pe_ttm > 0  -- 排除亏损股
          AND d.pe_ttm <= {self.criteria.max_pe}
          AND d.pb <= {self.criteria.max_pb}
          AND d.turnover_rate >= {self.criteria.min_turnover}
        ORDER BY d.total_mv_10k DESC
        """

        df = self.conn.execute(query).fetchdf()
        # 转换市值为亿元
        df["market_cap"] = df["market_cap_10k"] / 10000
        df["circulating_cap"] = df["circulating_cap_10k"] / 10000
        return df

    def get_historical_prices(
        self, symbols: list, days: int = 120, end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """获取历史价格计算动量指标"""
        if end_date is None:
            end_date = self.get_latest_date()

        symbols_str = "', '".join(symbols)

        query = f"""
        SELECT
            symbol,
            date,
            close,
            volume
        FROM v_bar_daily_raw
        WHERE symbol IN ('{symbols_str}')
          AND date >= DATE '{end_date}' - {days}
          AND date <= DATE '{end_date}'
        ORDER BY symbol, date
        """

        return self.conn.execute(query).fetchdf()

    def calculate_momentum_factors(self, prices_df: pd.DataFrame) -> pd.DataFrame:
        """计算动量因子"""
        results = []

        for symbol in prices_df["symbol"].unique():
            df = prices_df[prices_df["symbol"] == symbol].copy()
            df = df.sort_values("date")

            if len(df) < 60:  # 需要至少60天数据
                continue

            latest_price = df["close"].iloc[-1]

            # 计算移动平均线
            df["ma20"] = df["close"].rolling(window=20).mean()
            df["ma60"] = df["close"].rolling(window=60).mean()
            df["ma120"] = df["close"].rolling(window=120).mean()

            # 计算动量指标
            momentum_20d = (
                (latest_price / df["close"].iloc[-20] - 1) * 100
                if len(df) >= 20
                else None
            )
            momentum_60d = (
                (latest_price / df["close"].iloc[-60] - 1) * 100
                if len(df) >= 60
                else None
            )

            # 价格相对于均线的位置
            price_vs_ma120 = (
                latest_price / df["ma120"].iloc[-1]
                if pd.notna(df["ma120"].iloc[-1])
                else None
            )
            price_vs_ma60 = (
                latest_price / df["ma60"].iloc[-1]
                if pd.notna(df["ma60"].iloc[-1])
                else None
            )

            # 计算波动率（20日）
            returns = df["close"].pct_change().dropna()
            volatility_20d = (
                returns.tail(20).std() * np.sqrt(252) * 100
                if len(returns) >= 20
                else None
            )

            results.append(
                {
                    "symbol": symbol,
                    "momentum_20d": momentum_20d,
                    "momentum_60d": momentum_60d,
                    "price_vs_ma120": price_vs_ma120,
                    "price_vs_ma60": price_vs_ma60,
                    "volatility_20d": volatility_20d,
                }
            )

        return pd.DataFrame(results)

    def get_industry_medians(self, date: Optional[str] = None) -> pd.DataFrame:
        """获取行业中位数（用于相对估值）"""
        if date is None:
            date = self.get_latest_date()

        query = f"""
        SELECT
            s.industry,
            median(d.pe_ttm) as median_pe,
            median(d.pb) as median_pb,
            median(d.ps_ttm) as median_ps,
            count(*) as stock_count
        FROM v_indicator_daily d
        LEFT JOIN v_dim_symbol s ON d.symbol = s.symbol
        WHERE d.date = '{date}'
          AND d.pe_ttm > 0 AND d.pe_ttm < 100
          AND d.pb > 0 AND d.pb < 10
          AND d.ps_ttm > 0 AND d.ps_ttm < 50
          AND s.is_delisted = false
        GROUP BY s.industry
        HAVING count(*) >= 5
        """

        return self.conn.execute(query).fetchdf()

    def _normalize_factor(
        self, series: pd.Series, higher_is_better: bool = True, method: str = "rank"
    ) -> pd.Series:
        """
        因子标准化
        method: rank（排名标准化）或 zscore（Z-score标准化）
        """
        if method == "rank":
            # 排名标准化到0-100分
            ranks = series.rank(pct=True, na_option="bottom")
            normalized = ranks * 100
        else:  # zscore
            mean = series.mean()
            std = series.std()
            if std > 0:
                normalized = (series - mean) / std
                # 限制在-3到3之间，然后映射到0-100
                normalized = ((normalized.clip(-3, 3) + 3) / 6) * 100
            else:
                normalized = pd.Series(50, index=series.index)

        if not higher_is_better:
            normalized = 100 - normalized

        return normalized

    def calculate_factor_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算各因子得分"""
        # 估值因子（越低越好，所以higher_is_better=False）
        df["pe_score"] = self._normalize_factor(df["pe_ttm"], higher_is_better=False)
        df["pb_score"] = self._normalize_factor(df["pb"], higher_is_better=False)
        df["ps_score"] = self._normalize_factor(
            df["ps_ttm"].fillna(df["ps_ttm"].median()), higher_is_better=False
        )

        # 股息率（越高越好）
        df["dividend_score"] = self._normalize_factor(
            df["dividend_yield_ttm"].fillna(0), higher_is_better=True
        )

        # 动量因子
        df["momentum_20d_score"] = self._normalize_factor(
            df["momentum_20d"].fillna(0), higher_is_better=True
        )
        df["momentum_60d_score"] = self._normalize_factor(
            df["momentum_60d"].fillna(0), higher_is_better=True
        )

        # 流动性因子（换手率适中最好，这里用绝对偏差）
        turnover_ideal = 1.5  # 理想换手率1.5%
        df["turnover_deviation"] = abs(df["turnover_rate"] - turnover_ideal)
        df["turnover_score"] = self._normalize_factor(
            df["turnover_deviation"], higher_is_better=False
        )

        # 量比（适中最好）
        volume_ratio_ideal = 1.2
        df["volume_ratio_deviation"] = abs(
            df["volume_ratio"].fillna(1) - volume_ratio_ideal
        )
        df["volume_ratio_score"] = self._normalize_factor(
            df["volume_ratio_deviation"], higher_is_better=False
        )

        # 稳定性因子（波动率越低越好）
        df["stability_score"] = self._normalize_factor(
            df["volatility_20d"].fillna(df["volatility_20d"].median()),
            higher_is_better=False,
        )

        return df

    def calculate_total_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算综合得分"""
        w = self.weights

        # 计算加权总分
        df["total_score"] = (
            df["pe_score"] * w.pe_weight
            + df["pb_score"] * w.pb_weight
            + df["ps_score"] * w.ps_weight
            + df["dividend_score"] * w.dividend_weight
            + df["momentum_20d_score"] * w.momentum_20d_weight
            + df["momentum_60d_score"] * w.momentum_60d_weight
            + df["turnover_score"] * w.turnover_weight
            + df["volume_ratio_score"] * w.volume_ratio_weight
            + df["stability_score"] * w.stability_weight
        ) / 100.0  # 归一化到满分100

        return df

    def apply_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """应用筛选条件"""
        criteria = self.criteria

        # 基础筛选
        filtered = df[
            (df["price"] >= criteria.min_price)
            & (df["market_cap"] >= criteria.min_market_cap)
            & (df["market_cap"] <= criteria.max_market_cap)
            & (df["pe_ttm"] > criteria.min_pe)
            & (df["pe_ttm"] <= criteria.max_pe)
            & (df["pb"] <= criteria.max_pb)
            & (df["ps_ttm"].fillna(999) <= criteria.max_ps)
            & (df["turnover_rate"] >= criteria.min_turnover)
            & (df["volume_ratio"].fillna(0) >= criteria.min_volume_ratio)
        ].copy()

        # 动量筛选（价格相对于MA120不能偏离太多，但允许空值通过）
        if "price_vs_ma120" in filtered.columns:
            price_vs_ma120 = filtered["price_vs_ma120"]
            # 保留空值或符合条件的记录
            mask = price_vs_ma120.isna() | (
                (price_vs_ma120 >= criteria.min_price_vs_ma120)
                & (price_vs_ma120 <= criteria.max_price_vs_ma120)
            )
            filtered = filtered[mask]

        return filtered

    def run_screening(self, top_n: int = 50) -> pd.DataFrame:
        """执行多因子选股"""
        print("=" * 80)
        print("A股多因子选股模型")
        print("=" * 80)

        # 1. 获取基础股票池
        print("\n[1/5] 获取基础股票池...")
        universe = self.get_stock_universe()
        print(f"      初始股票池: {len(universe)} 只")

        # 2. 获取历史价格计算动量因子
        print("\n[2/5] 计算动量因子...")
        symbols = universe["symbol"].tolist()
        prices = self.get_historical_prices(symbols, days=130)
        momentum = self.calculate_momentum_factors(prices)
        print(f"      计算动量因子: {len(momentum)} 只股票")

        # 3. 合并数据
        print("\n[3/5] 合并数据...")
        merged = universe.merge(momentum, on="symbol", how="inner")
        print(f"      合并后: {len(merged)} 只股票")

        # 4. 应用筛选条件
        print("\n[4/5] 应用筛选条件...")
        filtered = self.apply_filters(merged)
        print(f"      通过筛选: {len(filtered)} 只股票")

        # 5. 计算因子得分
        print("\n[5/5] 计算因子得分...")
        scored = self.calculate_factor_scores(filtered)
        final = self.calculate_total_score(scored)

        # 排序并返回前N名
        final = final.sort_values("total_score", ascending=False)

        return final.head(top_n)

    def print_results(self, results: pd.DataFrame, top_n: int = 20):
        """打印结果"""
        print("\n" + "=" * 80)
        print(f"多因子选股结果（前{top_n}名）")
        print("=" * 80)

        display_cols = [
            "symbol",
            "name",
            "industry",
            "price",
            "market_cap",
            "pe_ttm",
            "pb",
            "dividend_yield_ttm",
            "momentum_20d",
            "momentum_60d",
            "price_vs_ma120",
            "turnover_rate",
            "total_score",
        ]

        display_df = results.head(top_n)[display_cols].copy()
        display_df["price"] = display_df["price"].round(2)
        display_df["market_cap"] = display_df["market_cap"].round(1)
        display_df["pe_ttm"] = display_df["pe_ttm"].round(2)
        display_df["pb"] = display_df["pb"].round(2)
        display_df["dividend_yield_ttm"] = (
            display_df["dividend_yield_ttm"].fillna(0).round(2)
        )
        display_df["momentum_20d"] = display_df["momentum_20d"].fillna(0).round(2)
        display_df["momentum_60d"] = display_df["momentum_60d"].fillna(0).round(2)
        display_df["price_vs_ma120"] = display_df["price_vs_ma120"].fillna(0).round(3)
        display_df["turnover_rate"] = display_df["turnover_rate"].round(2)
        display_df["total_score"] = display_df["total_score"].round(2)

        print(display_df.to_string(index=False))

    def save_results(self, results: pd.DataFrame, filename: Optional[str] = None):
        """保存结果到CSV"""
        if filename is None:
            date_str = datetime.now().strftime("%Y%m%d")
            filename = f"multifactor_screening_{date_str}.csv"

        output_path = PROJECT_ROOT / "scripts" / filename
        results.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"\n[保存] 完整结果已保存至: {output_path}")
        return output_path


def main():
    """主函数"""
    # 检查数据库
    if not DUCKDB_PATH.exists():
        print(f"错误：找不到数据库 {DUCKDB_PATH}")
        print("请先运行: uv run python scripts/sync_lake_daily.py --reprocess-days 5")
        sys.exit(1)

    # 创建选股器并运行
    with MultiFactorScreener() as screener:
        # 运行选股
        results = screener.run_screening(top_n=100)

        # 打印结果
        screener.print_results(results, top_n=30)

        # 保存结果
        output_path = screener.save_results(results)

        # 输出前10名代码
        print("\n" + "=" * 80)
        print("Top 10 推荐股票代码：")
        print("=" * 80)
        top10 = results.head(10)["symbol"].tolist()
        print(", ".join(top10))

    return results


if __name__ == "__main__":
    results = main()
