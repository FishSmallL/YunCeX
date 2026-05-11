# Kernel → Skill 转化与匹配系统

## 概述

Kaggle 下载的 kernel notebook（`.ipynb`）是非结构化的代码+Markdown+输出的混合体。本系统将其转化为结构化的、可按关键词检索的"可复用技巧库"（Skill Library），供主 Agent 在新竞赛中快速查阅。

---

## 一、整体数据流

```
用户指定关键词 "machine learning"
       │
       ▼
  [0. 匹配]  match_keyword()
       ├── 命中 → 直接返回已有 skill 上下文，跳过后续步骤
       └── 未命中 ↓
  [0.1 查找]  _find_kernel_dirs()
       ├── 本地有 kernel → 跳到 [1]
       └── 本地没有 ↓
  [0.2 下载]  _download_kernels_for_keyword()
       自动调用 Kaggle CLI：搜竞赛 → 排行榜 → 下载 kernel
       │
       ▼
  [1. 解析]  notebook_parser.py
       提取 cell 列表（code / markdown / raw）
       │
       ▼
  [2. 压缩]  notebook_to_text()
       展平为 LLM 友好的紧凑文本（≤8000 chars）
       │
       ▼
  [3. 提取]  skill_extractor.py + LLM
       按6类聚焦提取可复用技巧，输出结构化 JSON
       │
       ▼
  [4. 存储]  skill_library.py
       渲染为 SKILL.md，存入 skill_library/<keyword>/
       │
       ▼
  [5. 匹配]  match_keyword() + search_skills()
       精确+模糊匹配 → 取 top-k 条注入主 Agent
```

---

## 二、Kernel → Skill 转化详解

### 2.1 解析阶段（`notebook_parser.py`）

Kaggle 下载的 `.ipynb` 有两种格式：

- **XML 格式**（Kaggle CLI 产出）：`<cell id="..."><cell_type>markdown</cell_type>内容</cell>`
- **nbformat JSON**（标准 Jupyter）：`{"cells": [{"cell_type": "code", "source": [...]}]}`

`parse_notebook()` 自动检测并解析，输出统一的 cell 列表：

```python
[
  {"cell_type": "markdown", "source": "# Titanic — Clean Feature...", "outputs": []},
  {"cell_type": "code", "source": "import numpy as np\n...", "outputs": []},
  ...
]
```

`extract_code_comments()` 从代码中提取 `#` 注释，保留开发者的思路说明。

### 2.2 压缩阶段（`notebook_to_text()`）

LLM 有上下文窗口限制（8000 chars），notebook 原始内容通常远超此值。压缩策略：

| cell 类型 | 处理方式 |
|-----------|---------|
| markdown | 截断到 500 字符，保留标题和关键说明 |
| code | 截断到 800 字符，前置 `#` 注释作为上下文 |
| raw | 截断到 300 字符 |

输出格式：

```
## MARKDOWN
# Titanic — Clean Feature Engineering + CV Ensemble

## CODE
# Key comments: 创建 Title 特征; 聚合稀有称呼
def add_features(df):
    df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    title = title.replace({"Mlle": "Miss", ...})
    ...
```

### 2.3 提取阶段（`skill_extractor.py`）（核心）

这是整个系统的关键。LLM 在精心设计的 prompt 引导下，从压缩文本中提取可复用技巧。

#### Prompt 设计

```
你是一个竞赛数据科学专家。分析以下 Kaggle kernel notebook，提取可复用的技巧。

提取规则（只关注6类）：
1. 数据预处理 - 缺失值处理、异常值检测、数据清洗、类型转换
2. 特征工程   - 特征创建、编码、选择、缩放、组合
3. 模型架构   - 模型选择、层级设计、超参数配置、损失函数选择
4. 集成策略   - 模型融合、Stacking、Blending、投票、加权
5. 调试技巧   - 验证策略、问题排查方法、性能分析
6. 提交避坑   - 提交格式注意事项、常见错误避免、路径处理

输出格式（JSON 数组）：
[{
  "name": "英文短名-用连字符",
  "description": "一句话描述（≤200字）",
  "category": "特征工程",
  "technique": "技巧详细说明，包括为什么有效",
  "code_pattern": "关键代码片段（精简）",
  "when_to_use": "什么场景下应该使用这个技巧",
  "notes": "注意事项或常见陷阱"
}]

重要原则：
- 每类最多提取3个最重要的技巧
- 每个 description ≤ 200字
- code_pattern 只保留核心逻辑
- 只提取可迁移的通用技巧，不要竞赛特定信息（如列名、路径）
- 没有值得提取内容的类别直接跳过
- 关注 notebook 中实际实现的技巧，而非泛泛而谈
```

#### Prompt 设计思路

| 设计要素 | 作用 |
|---------|------|
| 6类明确划分 | 引导 LLM 按结构扫描，避免遗漏或偏科 |
| JSON 强约束输出 | 确保结果可程序化解析，不产生自由文本 |
| description ≤200字 | 防止 LLM 展开写长文，保持技巧库精简 |
| "可迁移"约束 | 最关键的过滤条件——剔除"把 titanic 的 Age 填为中位数"这类不可复用信息 |
| 每类最多3条 | 防止某个类别过度提取，保持分布均衡 |
| code_pattern 精简 | 只保留核心代码骨架，便于后续直接用 |

### 2.4 存储阶段（`skill_library.py`）

LLM 返回的 JSON 被 `save_skill()` 渲染为 SKILL.md 格式（兼容 hello_agents 的 SkillLoader）：

```markdown
---
name: target-encoding-with-smoothing
description: 使用平滑目标编码处理高基数类别特征，减少过拟合风险
keyword: machine_learning
category: 特征工程
source_kernel: titanic-cv-v2
source_competition: titanic
created: 2026-05-10
---

# 平滑目标编码

## 技巧说明
使用 k-fold target encoding + Laplace smoothing...

## 适用场景
- 类别数 > 100 的分类特征
- 目标变量是二分类/回归

## 代码模式
```python
def target_encode(train, test, col, target, folds=5, smoothing=10):
    ...
```

## 注意事项
- 必须使用交叉验证防止 data leakage
```

#### 目录结构

```
skill_library/
├── machine_learning/
│   ├── skill_001_target-encoding-smoothing.md
│   ├── skill_002_null-importance-feature-selection.md
│   └── ...
├── deep_learning/
│   └── ...
├── nlp/
│   └── ...
└── computer_vision/
    └── ...
```

每个关键词对应一个子目录，技能文件按编号+名称命名。

---

## 三、匹配算法详解

匹配系统解决的核心问题：**给定一个赛题关键词（如 "machine learning"），如何在本地 Skill 库中找到最相关的已有技巧？**

分为两层匹配：**粗匹配**（定位关键词目录）和 **精细匹配**（从目录中选 top-k 条）。

### 3.1 粗匹配：定位关键词目录（`match_keyword()`）

粗匹配负责将用户查询映射到 `skill_library/` 下的某个关键词目录。

#### 第一层：精确匹配

```
输入: "machine learning"
  → sanitize: 转小写 → 空格转下划线 → 去特殊字符 → "machine_learning"
  → 查找: skill_library/ 下是否存在目录 "machine_learning"？
  → 命中 ✓ → 返回 "machine_learning"
```

sanitize 规则（`_sanitize_dirname()`）：

| 步骤 | 示例 |
|------|------|
| 转小写 | `"Machine Learning"` → `"machine learning"` |
| 去特殊字符 | `"nlp/cv"` → `"nlpcv"` |
| 空格转下划线 | `"machine learning"` → `"machine_learning"` |
| 截断80字符 | 超长名称只取前80字符 |

#### 第二层：模糊匹配（精确匹配失败时触发）

当精确匹配未命中时，用 **Jaccard 相似度 + Levenshtein 编辑距离** 的加权组合对所有目录名打分：

```
综合得分 = 0.5 × Jaccard(token重叠率) + 0.5 × Levenshtein(字符串相似度)
```

**Jaccard 相似度**（`_tokenize()` + 集合运算）：

```
查询: "machine-learning" → tokens = {"machine", "learning"}
目录: "machine_learning" → tokens = {"machine", "learning"}

Jaccard = |交集| / |并集| = 2 / 2 = 1.0
```

**Levenshtein 编辑距离**（`_levenshtein_similarity()`）：

```
查询: "machine_learn" (13 chars)
目录: "machine_learning" (16 chars)

最少编辑次数: 插入 'i', 'n', 'g' = 3次
相似度 = 1 - 3/16 = 0.8125
```

**综合得分**：

```
score = 0.5 × 1.0 + 0.5 × 0.8125 = 0.906
threshold = 0.6 → 0.906 ≥ 0.6 → 匹配成功 ✓
```

#### 匹配示例

| 查询 | 目录 | 精确匹配 | 模糊得分 | 结果 |
|------|------|---------|---------|------|
| `"machine learning"` | `machine_learning` | ✓ | — | 命中 |
| `"machine-learning"` | `machine_learning` | ✗ | 0.906 | 命中 |
| `"ml"` | `machine_learning` | ✗ | 0.17 | **未命中**（差距太大） |
| `"deep learn"` | `deep_learning` | ✗ | 0.58 | **未命中**（低于阈值） |
| `"nlp"` | `natural_language_processing` | ✗ | 0.18 | **未命中** |

> 阈值 0.6 的设计意图：保守策略，宁可未命中后走 Kaggle 拉取流程，也不要在不相关的技巧目录里乱匹配。

#### 未命中时怎么办？

粗匹配返回 `None` → 触发 `KernelSkillAgent` 子 agent。

**KernelSkillAgent 处理逻辑**：

1. `match_keyword()` 查 skill_library → 命中则直接返回已有 skill 上下文
2. `_find_kernel_dirs()` 查本地是否有已下载的 kernel
3. **若都没有** → `_download_kernels_for_keyword()` 自动调用 Kaggle 流水线：
   - `batch_search_competitions()` 搜索竞赛
   - `get_leaderboard()` 获取排行榜
   - `download_competition_kernels()` 下载 kernel
4. 下载完成后 → 解析 notebook → LLM 提取 skill → 存入 skill_library
5. 下次相同关键词就能直接匹配到

### 3.2 精细匹配：选取 top-k 条技巧（`search_skills()`）

粗匹配定位到目录后，精细匹配从该目录中选取最相关的 k 条技巧。

**当前策略**：按创建时间倒序，取最新 k 条。

```
skill_library/machine_learning/
  skill_001_xxx.md  ← 最早
  skill_002_xxx.md
  skill_003_xxx.md
  ...
  skill_015_xxx.md  ← 最新

search_skills("machine_learning", library_dir, top_k=5)
  → 返回 skill_015, 014, 013, 012, 011（最新5条）
```

**设计原因**：更新的 kernel 通常代表更先进的技巧（新竞赛、新方法），按时间排序自然偏向高质量技能。

#### 可扩展方向（后续迭代）

当前精细匹配仅按时间排序，后续可增强：

- **类别加权**：根据当前赛题类型（回归/分类/CV/NLP），优先返回匹配类别的技能
- **关键词匹配**：将赛题描述与 skill frontmatter 中的 description/technique 做文本相似度排序
- **语义匹配**：用 embedding 模型对 skill 做向量化，与赛题描述做余弦相似度排序
- **战绩加权**：记录每条 skill 在历史竞赛中的使用效果（提分/不提分），高分优先

### 3.3 整体匹配流程图

```
新赛题 → LLM 提取关键词 "tabular ML"
              │
              ▼
    match_keyword("tabular ML")
              │
    ┌─────────┴──────────┐
    │  精确匹配            │
    │  sanitize → "tabular_ml" │
    │  skill_library/ 有此目录？ │
    └─────────┬──────────┘
         ┌────┴────┐
        是         否
         │          │
         ▼          ▼
    命中 ✓    模糊匹配（Jaccard + Levenshtein）
         │          │
         │    ┌─────┴─────┐
         │   ≥0.6      <0.6
         │    │          │
         │    ▼          ▼
         │  命中 ✓    未命中 ✗
         │    │          │
         └────┴──────────┘
              │          │
              ▼          ▼
	    search_skills()   _find_kernel_dirs()
	    取 top-k 条       本地有 kernel 吗？
	              │          │
	              │    ┌─────┴─────┐
	              │   有          没有
	              │    │            │
	              │    ▼            ▼
	              │  直接提取    [自动下载]
	              │    │       batch_search_competitions()
	              │    │       get_leaderboard()
	              │    │       download_competition_kernels()
	              │    │            │
	              │    └─────┬──────┘
	              │          ▼
	              │   提取 skill → 入库

              └────┬─────┘
                   ▼
          build_skill_context()
          注入主 Agent prompt
```

---

## 四、关键设计决策

| 决策 | 原因 |
|------|------|
| 用 LLM 而非正则提取技巧 | 技巧隐含在代码逻辑和注释中，需要语义理解；正则只能匹配表面模式 |
| 6类聚焦而非开放式提取 | 约束 LLM 注意力，避免产出"这个 kernel 用了 pandas"之类无用信息 |
| JSON 强格式输出 | 保证后续程序化解析入库，不需要人工审核 |
| "可迁移"约束 | 核心过滤——把 kernel 中的通用方法从竞赛特定细节中剥离 |
| 保守的模糊匹配阈值（0.6） | 宁可未命中走拉取流程，也不匹配到无关技巧误导 Agent |
| 最新优先的精细匹配 | 新 kernel 方法更先进，节省实现复杂相关性排序的成本 |

---

## 五、文件索引

| 文件 | 职责 |
|------|------|
| `kaggle_knowledge/kernel_processor/notebook_parser.py` | 解析 .ipynb，提取 cell 和注释，压缩为文本 |
| `kaggle_knowledge/kernel_processor/skill_extractor.py` | LLM prompt 设计 + 响应解析 |
| `kaggle_knowledge/kernel_processor/skill_library.py` | 存储、粗匹配、精细匹配、上下文构建 |
| `hello_agents/agents/kernel_skill_agent.py` | Agent 封装：匹配→查找→[自动下载]→解析→提取→入库，串联完整 pipeline |
| `kaggle_knowledge/skill_library/` | 技巧库存储目录 |
| `hello_agents/agents/factory.py` | Agent 工厂注册（`"kernel_skill"` 类型） |

---

## 六、Kaggle 数据采集配置

Kernel 的数量和质量直接决定 Skill 库的丰富程度。采集数量通过 `kaggle_knowledge/config.json` 和命令行参数控制。

### 6.1 配置文件（`config.json`）

```json
{
  "keywords": ["machine learning", "data science"],
  "output_dir": "output",
  "competitions_per_keyword": 5,
  "top_leaderboard_users": 5,
  "kernels_per_user": 5,
  "save_csv": {
    "competitions": true,
    "leaderboard": true,
    "kernels_list": true
  },
  "min_leaderboard_score": 0.6,
  "sort_by_teams": "numberOfTeams"
}
```

### 6.2 三个核心数量参数

这三个参数形成漏斗模型，决定最终下载的 kernel 总量：

```
每个关键词
  → 搜索 N 个竞赛         (competitions_per_keyword)
    → 每个竞赛取 M 名用户  (top_leaderboard_users)
      → 每个用户下载 K 个 kernel (kernels_per_user)

理论最大 kernel 数 = N × M × K
```

| 参数 | 默认值 | 作用 | 调大效果 | 调小效果 |
|------|--------|------|---------|---------|
| `competitions_per_keyword` | 5 | 每个关键词搜索几个竞赛 | 覆盖更多竞赛类型，Skill 更全面 | 只关注最热门的竞赛 |
| `top_leaderboard_users` | 5 | 每个竞赛排行榜取前几名用户 | 采集更多高分方案 | 只看最顶尖的几个选手 |
| `kernels_per_user` | 5 | 每个用户下载几个 kernel | 同一用户的不同方案都收录 | 每人只取代表作 |

#### 如何调整

**编辑 `config.json`**：

```json
{
  "competitions_per_keyword": 10,
  "top_leaderboard_users": 3,
  "kernels_per_user": 3
}
```

含义：每个关键词搜 10 个竞赛，每个竞赛取前 3 名用户，每人下载 3 个 kernel。理论最大 = 10 × 3 × 3 = 90 个 kernel/关键词。

#### 按场景推荐

| 场景 | competitions | users | kernels | 说明 |
|------|-------------|-------|---------|------|
| 快速探索 | 3 | 3 | 2 | ~18 kernel/关键词，够用 |
| 日常使用 | 5 | 5 | 5 | ~125 kernel/关键词，均衡 |
| 深入研究 | 10 | 10 | 5 | ~500 kernel/关键词，覆盖面广 |

### 6.3 命令行覆盖关键词

关键词可以通过命令行参数动态指定，不需要修改配置文件：

```bash
# 使用 config.json 中的关键词
python main.py

# 用命令行覆盖，搜索新关键词
python main.py "NLP" "Computer Vision" "time series"

# 单个关键词也可以
python main.py "graph neural network"
```

命令行参数优先级高于 `config.json` 中的 `keywords`，但其他参数（竞赛数、用户数等）始终从配置文件读取。

### 6.4 其他采集控制参数

| 参数 | 默认值 | 作用 |
|------|--------|------|
| `min_leaderboard_score` | 0.6 | 最低分数阈值，过滤掉低分用户（设为 `null` 关闭） |
| `sort_by_teams` | `"numberOfTeams"` | 竞赛排序依据：按参赛队伍数排序，确保优先下载热门竞赛 |
| `save_csv.*` | `true` | 是否保存中间 CSV 结果（竞赛列表、排行榜、kernel 列表） |
| `output_dir` | `"output"` | 数据输出根目录，kernel 下载到 `<output_dir>/<keyword>/kernels/` |

### 6.5 数据量与 Skill 质量的关系

```
采集量 ↑  →  Skill 库覆盖面 ↑  →  匹配命中率 ↑
         →  LLM 提取成本 ↑    →  噪声可能增加
         →  下载时间 ↑

采集量 ↓  →  Skill 库精简       →  匹配命中率 ↓
         →  成本低
```

建议：**初次使用时用较大参数（10/10/5）跑一轮建立基础库，后续维护用默认参数（5/5/5）增量更新**。

### 6.6 执行流程总览

```bash
cd kaggle_knowledge

# 1. 修改 config.json 中的 competitions_per_keyword / top_leaderboard_users / kernels_per_user

# 2. 运行采集（可命令行覆盖关键词）
python main.py "machine learning" "deep learning"

# 3. 查看结果
ls output/machine_learning/kernels/
ls skill_library/machine_learning/   # Skill 库（需运行 KernelSkillAgent 提取后才有）

# 4. 用 KernelSkillAgent 提取 Skill（在 Python 中调用或通过主 Agent 触发）
```
