"""
组合构建器 — 从弹性打分结果构建事件驱动个股组合。

功能:
  - Top N 选股 (可配 3/5/8)
  - 权重分配: 等权 / 分数加权
  - 单票上限约束
  - 行业集中度上限 (可选)

使用:
    from src.national_team.portfolio_builder import PortfolioBuilder
    builder = PortfolioBuilder(top_n=5, weighting="score")
    portfolio = builder.build(scored_df)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import polars as pl


class Weighting(Enum):
    EQUAL = "equal"
    SCORE = "score"


@dataclass
class PortfolioBuilder:
    """事件驱动弹性组合构建器。"""

    top_n: int = 5
    """选股数量"""

    weighting: str = "equal"
    """权重方式: 'equal' 等权 | 'score' 分数加权"""

    max_single_weight: float = 0.40
    """单票最大权重"""

    max_industry_weight: float = 1.0
    """单行业最大权重 (1.0 = 不限制)"""

    min_score: float = 0.0
    """最低分数线，低于此不入选"""

    def build(
        self,
        scored: pl.DataFrame,
        industry_map: pl.DataFrame | None = None,
    ) -> pl.DataFrame:
        """
        根据弹性打分构建组合。

        Parameters
        ----------
        scored : ElasticScorer.score() 的输出，必须包含 symbol, elastic_score
        industry_map : 可选 DataFrame[symbol, industry]，用于行业集中度控制

        Returns
        -------
        DataFrame[symbol, elastic_score, weight]
        权重之和 = 1.0
        """
        if scored.height == 0:
            return pl.DataFrame(schema={
                "symbol": pl.Utf8,
                "elastic_score": pl.Float64,
                "weight": pl.Float64,
            })

        # 过滤最低分数线 & 不可交易 (tradability == 0)
        pool = scored.filter(pl.col("elastic_score") > self.min_score)
        if "tradability" in pool.columns:
            pool = pool.filter(pl.col("tradability") > 0)

        # Top N
        pool = pool.sort("elastic_score", descending=True).head(self.top_n)

        if pool.height == 0:
            return pl.DataFrame(schema={
                "symbol": pl.Utf8,
                "elastic_score": pl.Float64,
                "weight": pl.Float64,
            })

        # 行业集中度约束
        if industry_map is not None and self.max_industry_weight < 1.0:
            pool = self._apply_industry_constraint(pool, industry_map)

        # 权重分配
        if self.weighting == "score":
            pool = pool.with_columns(
                (pl.col("elastic_score") / pl.col("elastic_score").sum()).alias("weight"),
            )
        else:
            n = pool.height
            pool = pool.with_columns(
                pl.lit(1.0 / n).alias("weight"),
            )

        # 单票上限约束 (迭代重分配)
        pool = self._cap_weights(pool)

        return pool.select("symbol", "elastic_score", "weight")

    def _cap_weights(self, df: pl.DataFrame) -> pl.DataFrame:
        """迭代截断超限权重并重分配。"""
        for _ in range(10):
            over = df.filter(pl.col("weight") > self.max_single_weight)
            if over.height == 0:
                break

            under = df.filter(pl.col("weight") <= self.max_single_weight)
            excess = over.select(
                (pl.col("weight") - self.max_single_weight).sum()
            ).item()

            df = pl.concat([
                over.with_columns(pl.lit(self.max_single_weight).alias("weight")),
                under,
            ])

            if under.height > 0 and excess > 0:
                under_sum = under["weight"].sum()
                if under_sum > 0:
                    df = df.with_columns(
                        pl.when(pl.col("weight") < self.max_single_weight)
                        .then(
                            pl.col("weight")
                            + excess * pl.col("weight") / under_sum
                        )
                        .otherwise(pl.col("weight"))
                        .alias("weight"),
                    )

        # 归一化确保总和=1
        total = df["weight"].sum()
        if total > 0:
            df = df.with_columns((pl.col("weight") / total).alias("weight"))

        return df

    def _apply_industry_constraint(
        self,
        pool: pl.DataFrame,
        industry_map: pl.DataFrame,
    ) -> pl.DataFrame:
        """按行业集中度约束筛选。"""
        pool = pool.join(industry_map, on="symbol", how="left")
        pool = pool.with_columns(
            pl.col("industry").fill_null("未知"),
        )

        # 每个行业最多允许的票数
        max_per_ind = max(1, int(self.top_n * self.max_industry_weight))

        result = (
            pool.with_columns(
                pl.col("elastic_score")
                .rank("ordinal", descending=True)
                .over("industry")
                .alias("_ind_rank"),
            )
            .filter(pl.col("_ind_rank") <= max_per_ind)
            .drop("_ind_rank", "industry")
        )

        return result
