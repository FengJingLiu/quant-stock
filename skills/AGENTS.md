# Skills 技能目录

本目录包含 15 个 FinSkills 中国市场投资分析技能，涵盖从股票筛选、财务分析到组合管理的完整投资研究流程。

## 目录结构

```
skills/
├── esg-screener/                      # ESG筛选器
├── event-driven-detector/             # 事件驱动机会识别器
├── financial-statement-analyzer/      # 财务报表深度分析
├── findata-toolkit-cn/                # 金融数据工具包（数据支撑）
├── high-dividend-strategy/            # 高股息策略分析器
├── insider-trading-analyzer/          # 董监高增减持分析器
├── portfolio-health-check/            # 组合健康诊断
├── quant-factor-screener/             # 量化因子筛选器
├── risk-adjusted-return-optimizer/    # 风险调整收益优化器
├── sector-rotation-detector/          # 行业轮动信号探测器
├── sentiment-reality-gap/             # 市场情绪与基本面偏差分析
├── small-cap-growth-identifier/       # 小盘成长股发现器
├── suitability-report-generator/      # 投资适当性报告生成器
├── tech-hype-vs-fundamentals/         # 科技股炒作vs基本面分析
├── undervalued-stock-screener/        # 低估值股票筛选器
└── AGENTS.md                          # 本文件
```

## 技能分类

### 一、发现与筛选（Discovery & Screening）

用于寻找投资机会的技能。

| 技能 | 用途 | 触发关键词 |
|------|------|-----------|
| **低估值股票筛选器** (`undervalued-stock-screener`) | 扫描A股低估值、基本面强劲的公司 | "低估值股票"、"价值投资"、"A股便宜股票"、"筛选低PE/PB" |
| **董监高增减持分析器** (`insider-trading-analyzer`) | 分析管理层增持信号 | "董监高增持"、"大股东买入"、"内部人交易"、"增持信号" |
| **市场情绪与基本面偏差分析** (`sentiment-reality-gap`) | 寻找被市场错杀的股票 | "逆向投资"、"超跌反弹"、"市场错杀"、"情绪与基本面背离" |
| **小盘成长股发现器** (`small-cap-growth-identifier`) | 发现被忽视的小市值高成长公司 | "小盘成长股"、"专精特新"、"市值小但增长快" |
| **量化因子筛选器** (`quant-factor-screener`) | 多因子模型系统化选股 | "因子投资"、"Smart Beta"、"量化选股"、"多因子筛选" |
| **ESG筛选器** (`esg-screener`) | ESG视角筛选可持续投资标的 | "ESG投资"、"绿色投资"、"社会责任投资"、"ESG评分" |

### 二、深度分析（Deep Analysis）

用于评估特定机会的技能。

| 技能 | 用途 | 触发关键词 |
|------|------|-----------|
| **高股息策略分析器** (`high-dividend-strategy`) | 分析A股高股息股票的分红可持续性 | "高股息"、"红利策略"、"A股分红"、"股息率排名" |
| **科技股炒作vs基本面分析** (`tech-hype-vs-fundamentals`) | 评估科技股估值泡沫与基本面 | "科技股估值"、"科创板估值"、"AI/芯片估值"、"科技泡沫" |
| **行业轮动信号探测器** (`sector-rotation-detector`) | 宏观驱动的行业配置建议 | "行业轮动"、"超配行业"、"宏观配置"、"经济周期投资" |
| **财务报表深度分析** (`financial-statement-analyzer`) | 单公司财务深度分析 | "财务分析"、"杜邦分析"、"Z值评分"、"盈利质量" |
| **事件驱动机会识别器** (`event-driven-detector`) | 并购重组、回购等事件机会 | "并购重组"、"资产注入"、"回购增持"、"国企改革" |

### 三、组合与文档（Portfolio & Documentation）

用于构建组合和生成报告的技能。

| 技能 | 用途 | 触发关键词 |
|------|------|-----------|
| **风险调整收益优化器** (`risk-adjusted-return-optimizer`) | 构建风险调整后最优组合 | "构建投资组合"、"资产配置"、"组合优化"、"仓位管理" |
| **组合健康诊断** (`portfolio-health-check`) | 诊断现有持仓风险 | "组合诊断"、"持仓风险"、"组合审查"、"压力测试" |
| **投资适当性报告生成器** (`suitability-report-generator`) | 生成合规投资报告 | "适当性报告"、"风险披露"、"投资理由"、"合规报告" |

### 四、数据工具包（Data Toolkit）

| 技能 | 用途 |
|------|------|
| **金融数据工具包** (`findata-toolkit-cn`) | A股实时数据：行情、财务指标、董监高增减持、北向资金、宏观数据（LPR、PMI、CPI、M2） |

## 快速使用指南

### 安装数据工具包依赖（一次性）

```bash
cd skills/findata-toolkit-cn
pip install -r requirements.txt
```

### 使用示例

**股票筛选类**：
```
"帮我筛选A股低估值股票" → 使用 undervalued-stock-screener
"分析最近哪些公司董事长在大量增持" → 使用 insider-trading-analyzer
"帮我找几只被市场错杀的A股" → 使用 sentiment-reality-gap
"推荐几只市值小但增长快的专精特新公司" → 使用 small-cap-growth-identifier
"用多因子模型帮我筛选A股" → 使用 quant-factor-screener
"帮我找ESG评分最高的沪深300成分股" → 使用 esg-screener
```

**深度分析类**：
```
"A股有哪些高股息但分红可持续的标的？" → 使用 high-dividend-strategy
"当前宏观环境下应该超配哪些行业？" → 使用 sector-rotation-detector
"深度分析一下贵州茅台的财务报表" → 使用 financial-statement-analyzer
"最近有哪些A股并购重组机会？" → 使用 event-driven-detector
"科创板哪些公司估值泡沫最严重？" → 使用 tech-hype-vs-fundamentals
```

**组合管理类**：
```
"用30万帮我构建一个稳健型投资组合" → 使用 risk-adjusted-return-optimizer
"帮我诊断一下我的持仓有什么风险" → 使用 portfolio-health-check
"为这个投资建议生成一份适当性报告" → 使用 suitability-report-generator
```

## 数据说明

所有技能的数据来源均为免费，无需API密钥：

| 来源 | 数据内容 |
|------|---------|
| **AKShare** | A股行情、财务数据、董监高交易、北向资金、宏观指标 |

## 技能文件结构

每个技能遵循统一的三层架构：

```
skill-name/
├── SKILL.md                        # 主文件：触发条件、工作流程、核心指导
└── references/
    ├── xxx-methodology.md          # 详细方法论：公式、评分标准、行业基准
    └── output-template.md          # 报告模板：结构化输出格式
```

## 重要声明

> **免责声明**：本技能集合仅供信息和教育目的使用，不构成投资建议、推荐或买卖证券的要约。所有分析基于公开数据和模型假设，可能存在错误或遗漏。过往业绩不代表未来表现。投资有风险，包括可能损失本金。在做出任何投资决策前，请咨询合格的投资顾问。

## 许可证

所有技能均采用 Apache License 2.0 许可证。
Copyright 2025 FinoGeeks Technology Ltd
