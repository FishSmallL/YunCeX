# Kaggle Skill 提取系统 — 技术文档

> 记录 2026-05-11 ~ 2026-05-12 期间对 `kaggle_knowledge` 和 `hello_agents` 体系的重要修改。

---

## 目录

1. [架构概览](#1-架构概览)
2. [TaskTool 支持 kernel_skill 子 Agent](#2-tasktool-支持-kernel_skill-子-agent)
3. [KernelSkillAgent 异步流式改造](#3-kernelskillagent-异步流式改造)
4. [Kaggle 竞赛分页搜索](#4-kaggle-竞赛分页搜索)
5. [Skill 提取质量改进](#5-skill-提取质量改进)
6. [Skill 库结构变更](#6-skill-库结构变更)
7. [Bug 修复清单](#7-bug-修复清单)
8. [配置文件变更](#8-配置文件变更)
9. [演示脚本 main_kaggle.py](#9-演示脚本-main_kagglepy)
10. [关键调用链路](#10-关键调用链路)

---

## 1. 架构概览

```
main.py (ReActAgent 主控)
  │
  ├─ 调用 Task 工具 → TaskTool.run()
  │     │
  │     ├─ agent_type="kernel_skill" → _run_kernel_skill_streaming()
  │     │     ├─ KernelSkillAgent.arun_stream()  →  流式事件
  │     │     └─ 收集 AGENT_FINISH 事件 → ToolResponse 返回主Agent
  │     │
  │     └─ 其他 agent_type → run_as_subagent() (同步)
  │
  └─ KernelSkillAgent 内部流程:
        1. 解析输入 keyword
        2. 匹配 Skill 库 (match_keyword, 两阶段匹配)
           ├─ 匹配到且有足够技能 → 直接返回
           └─ 不足或未匹配 → 继续
        3. 检查本地 kernel 数 (对比 min_kernels_per_keyword)
           └─ 不足 → Kaggle 下载
        4. 两阶段 LLM 提取:
           Phase 1: 亮点扫描 (流式)
           Phase 2: 深度提取 (流式)
        5. 两阶段去重:
           Stage 1: 精确去重 (category + name 匹配)
           Stage 2: 语义去重 (Jaccard + LLM 确认)
        6. 清理旧文件 → 保存新 Skill
```

---

## 2. TaskTool 支持 kernel_skill 子 Agent

### 修改文件

- `hello_agents/tools/builtin/task_tool.py`

### 改动详情

**2.1 agent_type 参数扩展**

```python
# task_tool.py:65 — 新增 kernel_skill 选项
ToolParameter(
    name="agent_type",
    type="string",
    description="子代理类型：react（推理行动）、reflection（反思）、"
                "plan（规划）、simple（简单对话）、kernel_skill（Kaggle技能提取）",
)
```

**2.2 流式执行路径**

`TaskTool.run()` 中检测 `agent_type == "kernel_skill"` 后走专门的流式路径：

```python
# task_tool.py:118
if agent_type == "kernel_skill" and hasattr(subagent, 'arun_stream'):
    return self._run_kernel_skill_streaming(
        subagent, task, agent_type, start_time
    )
```

**2.3 `_run_kernel_skill_streaming()` 方法**

通过 `asyncio.run()` 桥接同步/异步：

```python
def _run_kernel_skill_streaming(self, subagent, task, agent_type, start_time):
    import asyncio
    import time
    from ...core.streaming import StreamEventType

    async def _stream():
        async for event in subagent.arun_stream(task):
            if event.type == StreamEventType.LLM_CHUNK:
                chunk = event.data.get("chunk", "")
                if chunk:
                    print(chunk, end="", flush=True)  # 实时打印
            elif event.type == StreamEventType.AGENT_FINISH:
                final_result = event.data.get("result", "")
            # ... 处理其他事件类型

    asyncio.run(_stream())
    # 构建 ToolResponse 返回给主 Agent
```

关键点：`_run_kernel_skill_streaming` 内对所有 chunk 类型都打印（不跳过 thinking），确保 LLM 提取过程的流式输出可见。

---

## 3. KernelSkillAgent 异步流式改造

### 修改文件

- `hello_agents/agents/kernel_skill_agent.py`

### 3.1 新增 `arun_stream()` 方法

替代同步 `run()`，使用 `StreamEvent` 体系实时报告进度。分 7 个阶段:

```
阶段 1: AGENT_START
阶段 2: 解析输入 → LLM_CHUNK
阶段 3: 匹配 Skill 库 → STEP_START → LLM_CHUNK → STEP_FINISH
  3a: 匹配成功 → AGENT_FINISH (含 build_skill_context 结果)
阶段 4: 下载 Kaggle kernel → LLM_CHUNK (竞赛/排行榜/下载进度)
阶段 5: 提取技能 → Phase1 流式扫描 → Phase2 流式深度提取
阶段 6: 去重 + 保存
阶段 7: AGENT_FINISH (含完整摘要)
```

### 3.2 两阶段流式提取

**Phase 1 亮点扫描**（`astream_invoke`, max_tokens=2048）:
```python
yield self._chunk("    Phase1: 正在扫描 notebook 亮点...\n")
async for chunk in self.llm.astream_invoke(
    messages=[{"role": "user", "content": scan_prompt}],
    max_tokens=2048,
):
    full_scan += chunk
    yield self._chunk(chunk, chunk_type="thinking")
```

**Phase 2 深度提取**（`astream_invoke`, max_tokens=8192）:
```python
async for chunk in self.llm.astream_invoke(
    messages=[{"role": "user", "content": deep_prompt}],
    max_tokens=8192,
):
    full_response += chunk
    yield self._chunk(chunk, chunk_type="thinking")
```

### 3.3 `_chunk()` 工具方法

```python
@staticmethod
def _chunk(text: str, chunk_type: str = "text") -> StreamEvent:
    return StreamEvent.create(
        StreamEventType.LLM_CHUNK, "",
        chunk=text, chunk_type=chunk_type,
    )
```

### 3.4 Skill 数量检查

在 `run()` 和 `arun_stream()` 中，匹配 Skill 库成功后会比较 `skill_count` 和 `min_skills_per_keyword`（配置文件）:

```python
min_skills = self._get_min_skills()
skill_count = len(search_skills(matched, self.skill_library_dir, top_k=999))
if skill_count >= min_skills:
    return existing_skills  # 够用，直接返回
else:
    # 不足，继续下载+提取补充
```

### 3.5 最少 Kernel 数检查

在 `run()` 和 `arun_stream()` 中，下载前检查本地 kernel 数:

```python
min_kernels = self._get_min_kernels()
if len(kernel_dirs) < min_kernels:
    # 触发 Kaggle 下载补充
    async for event in self._download_kernels_stream(keyword):
        ...
```

### 3.6 旧 Skill 文件清理

保存前先删除该关键词目录下所有旧 `skill_*.md`:

```python
kw_dir = Path(self.skill_library_dir) / self._sanitize_keyword(keyword)
if kw_dir.is_dir():
    for f in kw_dir.glob("skill_*.md"):
        try:
            f.unlink()
        except Exception:
            pass
```

### 3.7 递归查找 Kernel 目录

`_find_kernel_dirs()` 改为递归搜索，用 `rglob("kernel-metadata.json")` 定位:

```python
def _find_kernel_dirs_recursive(root: Path) -> List[str]:
    results = []
    for item in root.rglob("kernel-metadata.json"):
        kernel_dir = str(item.parent)
        if kernel_dir not in results:
            results.append(kernel_dir)
    return results
```

---

## 4. Kaggle 竞赛分页搜索

### 修改文件

- `kaggle_knowledge/search_competitions.py`

### 4.1 命令格式变更

```
旧: kaggle competitions list --search nlp --sort-by numberOfTeams --csv --page-size 30
新: kaggle competitions list --search nlp --sort-by numberOfTeams --csv -p 1
```

### 4.2 新增函数

**`search_competitions_paginated()`** — 分页搜索主函数:

```python
def search_competitions_paginated(
    keyword: str,
    max_competitions: int = 5,
    save_csv_flag: bool = True,
    output_dir: str = "output",
    min_team_count: int = 0,
    max_pages: int = 5,
) -> Optional[List[Dict]]:
```

**分页逻辑**:
1. 从 `competitions_{keyword}.csv` 加载已处理竞赛的 `ref` 集合（去重用）
2. 从 page 1 开始，每页调用 `search_competitions(keyword, page=page)`
3. 过滤已处理的竞赛（`ref` 去重）
4. 过滤 `teamCount < min_team_count` 的竞赛
5. 若整页所有竞赛 `teamCount < min_team_count` → 停止翻页
6. 新竞赛追加保存到 CSV（`_append_competitions_csv`）
7. 累计足够数量或达到 `max_pages` → 停止

**`_load_existing_competition_refs()`** — 读取 CSV 返回 ref 集合:

```python
def _load_existing_competition_refs(keyword: str, output_dir: str) -> Set[str]:
    # 读取 competitions_{keyword}.csv，提取所有 ref 值
```

**`_append_competitions_csv()`** — 追加模式保存:

```python
def _append_competitions_csv(competitions, keyword, output_dir):
    # 用 "a" 模式追加，首次写入时带 header
```

### 4.3 更新 `batch_search_competitions()`

改为调用 `search_competitions_paginated()` 统一分页逻辑。

---

## 5. Skill 提取质量改进

### 修改文件

- `kaggle_knowledge/kernel_processor/skill_extractor.py`
- `kaggle_knowledge/kernel_processor/notebook_parser.py`

### 5.1 Notebook 处理能力提升

```python
# notebook_parser.py:95
def notebook_to_text(cells: List[Dict], max_len: int = 30000) -> str:
    # 从 8000 → 30000 字符
```

### 5.2 分类体系重构（5 类 → 8 类）

```python
CATEGORIES = [
    "数据处理",      # 缺失值/异常值/清洗/格式转换/数据加载
    "特征与表示",    # 特征工程(ML)、Embedding(DL)、Tokenizer、时序分解
    "模型设计",      # 网络结构、Backbone、层/激活/注意力/归一化
    "训练策略",      # LR调度、优化器、正则化、早停、梯度管理、混合精度
    "验证与评估",    # CV分折、防泄漏、稳定性评估、OOF、对抗验证
    "数据增强",      # CV增广、NLP增广、表格合成、mixup/cutmix
    "损失与指标",    # 自定义损失、多任务损失、Label Smoothing、Focal Loss
    "集成与后处理",  # 模型融合、Stacking/Blending、TTA、校准、阈值优化
]
```

### 5.3 新增 Prompt

**`EXTRACTION_PROMPT`** — 单 pass 提取（保留兼容）:
- 8 类提取类别
- 每类最多 5 条
- 新增字段: `competition_type`, `estimated_impact`, `use_case`

**`SCAN_PROMPT`** — Phase 1 亮点扫描:
```
快速浏览 notebook，识别 2-5 个最有价值的技术亮点。
输出: [{"highlight": "...", "category": "...", "impact": "高"}]
```

**`DEEP_EXTRACT_PROMPT`** — Phase 2 深度提取:
```
针对以下亮点深度提取结构化 skill JSON。
输入: 亮点列表 + 完整 notebook 文本
```

### 5.4 两阶段提取函数

```python
def extract_skills_2pass(
    notebook_text, kernel_name, competition, keyword, llm
) -> List[Dict]:
    # Phase 1: 亮点扫描 (max_tokens=2048)
    # Phase 2: 深度提取 (max_tokens=8192)
    # 失败回退到单 pass
```

### 5.5 两阶段去重函数

```python
def dedup_skills(skills: List[Dict], llm=None) -> List[Dict]:
```

**阶段 1 — 精确去重**:
- 按 `(category, sanitized_name)` 分组
- 同组内按 `_skill_score()` 择优
- 合并 use_case/notes/source_kernel

**阶段 2 — 语义去重** (`_semantic_dedup`):
- 同 category 内两两计算 Jaccard 分词重叠率（阈值 0.45）
- 候选对批量 LLM 确认: "技巧A 和 技巧B 是否本质上是同一技巧？yes/no"
- 确认相同 → 合并到评分高的版本

**`_skill_score()` 评分标准**:
```python
code_score  = min(len(code_pattern) // 200, 3)
tech_score  = min(len(technique) // 150, 3)
desc_score  = 1 if len(description) > 50 else 0
impact_bonus = 2 if estimated_impact == "高" else (1 if "中" else 0)
```

### 5.6 新增字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `estimated_impact` | `"高"/"中"/"低"` | LLM 主观判断的技巧影响力，用于排序和去重择优 |
| `competition_type` | 字符串 | 竞赛类型：图像分类/文本分类/表格二分类/时序预测 等 |
| `use_case` | 字符串 | 合并旧字段 `problems_solved` + `when_to_use`，格式: "解决: ... \| 场景: ..." |

---

## 6. Skill 库结构变更

### 修改文件

- `kaggle_knowledge/kernel_processor/skill_library.py`

### 6.1 SKILL.md 模板变更

**旧模板**:
```yaml
---
name: xxx
description: xxx
problems_solved: xxx
keyword: xxx
category: xxx
source_kernel: xxx
source_competition: xxx
created: 2026-05-12
---
## 可解决的问题
## 技巧说明
## 适用场景
## 代码模式
## 注意事项
```

**新模板**:
```yaml
---
name: xxx
description: xxx
use_case: xxx
keyword: xxx
category: xxx
competition_type: xxx
estimated_impact: xxx
source_kernel: xxx
source_competition: xxx
created: 2026-05-12
---
## 用途与场景        ← 合并 "可解决的问题" + "适用场景"
## 技巧说明
## 代码模式
## 注意事项
```

### 6.2 向后兼容

`search_skills()` 读取旧文件时自动转换:
```python
uc = frontmatter.get("use_case", "")
if not uc:
    ps = frontmatter.get("problems_solved", "")
    wtu = frontmatter.get("when_to_use", "")
    uc = f"解决: {ps} | 场景: {wtu}" if ps and wtu else (ps or wtu)
```

### 6.3 `build_skill_context()` 输出格式

```python
## 历史竞赛技巧库 (关键词: machine_learning)

### 技巧1: elo-dynamic-rating-feature [中] (表格分类)
分类: 特征与表示
描述: 为序列化对抗性事件构建动态实时能力评分特征...
用途: 解决: 选手实时实力估计,序列比赛预测 | 场景: 序列对抗性事件...
{body 截断到 2000 字符，优先在代码块结束处截断}
```

**新增 display 字段**: `description` 独立显示, `[影响力]` 标签, `(竞赛类型)` 标签。

### 6.4 `search_skills()` 排序变更

按 `estimated_impact` 排序: 高 → 中 → 低 → 无，同影响力按文件序（时间倒序）。

### 6.5 body 截断策略

从 1000 → 2000 字符，优先在 `` ``` `` 代码块结束标记后截断，保证代码完整:

```python
if len(body) > 2000:
    code_end = body.rfind("```", 0, 2000)
    if code_end > 1500:
        body = body[:code_end + 4] + "\n...(已截断)"
    else:
        body = body[:2000] + "\n...(已截断)"
```

---

## 7. Bug 修复清单

| # | 文件 | Bug | 修复 |
|---|------|-----|------|
| 1 | `task_tool.py:238` | `name 'time' is not defined` | `_run_kernel_skill_streaming` 内加 `import time` |
| 2 | `download_kernels.py:160` | `WinError 267` 目录名含 `:` | kernel_name sanitize: `re.sub(r'[<>:"/\\\|?*]', '_', ...)` |
| 3 | `kernel_skill_agent.py` | `_find_kernel_dirs` 只找一层深 | 改为 `rglob("kernel-metadata.json")` 递归搜索 |
| 4 | `kernel_skill_agent.py` | Skill 库匹配在 `if not kernel_dirs` 内 | 提到外面，始终先查 Skill 库 |
| 5 | `main_kaggle.py` | LLM_CHUNK 双重输出导致乱码 | 去掉事件循环中的 `print(chunk)`，ReActAgent 已自己打印 |
| 6 | `skill_extractor.py` | Phase 1 `llm.invoke()` 阻塞无输出 | 改为 `astream_invoke()` 流式输出 |
| 7 | `search_competitions.py` | 不同关键词搜索到同一竞赛时重复下载 | `_load_existing_competition_refs` 从单文件改为扫描全部 `competitions_*.csv`，跨关键词去重 |
| 8 | `kernel_skill_agent.py` | Phase 1 结束到 Phase 2 开始之间屏幕空白 | Phase 2 开始前 yield `"Phase2: 正在深度提取技巧..."` |

---

## 8. 配置文件变更

### `kaggle_knowledge/config.json`

```json
{
  "min_skills_per_keyword": 10,
  "min_kernels_per_keyword": 3,
  "search_max_pages": 5,
  "competitions_per_keyword": 20,
  "top_leaderboard_users": 10,
  "kernels_per_user": 2,
  "min_team_count": 100,
  "min_leaderboard_score": 0.6,
  "top_skills": 10
}
```

| 字段 | 默认值 | 用途 |
|------|--------|------|
| `min_skills_per_keyword` | 10 | 每个关键词最少技能数，不足触发补充提取 |
| `min_kernels_per_keyword` | 3 | 每个关键词最少 kernel 数，不足触发 Kaggle 搜索下载 |
| `search_max_pages` | 5 | 竞赛搜索最多翻页数 |
| `top_skills` | 10 | `build_skill_context` 返回的 top-k 技能数 |

---

## 9. 演示脚本 main_kaggle.py

### 用法

```bash
# 全链路模拟 (ReActAgent → Task → KernelSkillAgent)
python main_kaggle.py --keyword "deep_learning"

# 纯流式演示 (直接调 arun_stream)
python main_kaggle.py --direct --keyword "deep_learning"

# 匹配已有 skill 库（不消耗 tokens）
python main_kaggle.py --direct --keyword "machine_learning"
```

### 架构

- `--direct` 模式: 直接创建 `KernelSkillAgent` → `arun_stream()` → 事件循环打印
- 默认模式: 创建 `ReActAgent` + Task 工具 → 流式推理 → 触发 `KernelSkillAgent` 子 Agent
- 事件循环复用 `main.py` 的模式，但 llm_chunk 处理改为 `pass`（避免双重输出乱码）

---

## 10. 关键调用链路

### 主 Agent 触发 Skill 提取

```
main.py ReActAgent
  → system_prompt: "当需要了解某领域的竞赛技巧时，使用 Task 工具..."
  → LLM 输出: Action: Task[task="machine_learning", agent_type="kernel_skill"]
  → TaskTool.run()
    → agent_factory("kernel_skill")
      → default_subagent_factory → create_agent("kernel_skill")
        → KernelSkillAgent(name="subagent-kernel_skill", llm=..., ...)
    → _run_kernel_skill_streaming(subagent, task, ...)
      → asyncio.run(_stream())
        → async for event in subagent.arun_stream("machine_learning"):
            AGENT_START → STEP_START → LLM_CHUNK → ... → AGENT_FINISH
      → return ToolResponse.success(text=final_result)
  → ReActAgent 收到工具结果 → 总结技巧 → Finish
```

### 数据流: Notebook → Skill

```
Kernel .ipynb
  → parse_notebook()  → cells[{cell_type, source}]
  → notebook_to_text(max_len=30000)  → 压缩文本
  → Phase 1: SCAN_PROMPT + astream_invoke(max_tokens=2048)
      → _extract_json_array()  → [{"highlight": ..., "category": ..., "impact": ...}]
  → Phase 2: DEEP_EXTRACT_PROMPT + astream_invoke(max_tokens=8192)
      → _extract_json_array()  → [{name, description, category, ...}]
      → _parse_llm_response()  → filled skill dicts
  → dedup_skills(skills, llm)
      → 精确去重 (category + sanitized_name)
      → 语义去重 (Jaccard + LLM confirmation)
  → clean old files → save_skill() × N  → skill_library/<keyword>/skill_NNN_name.md
```

### 数据流: Skill 库 → Agent Context

```
search_skills(keyword, library_dir, top_k)
  → 扫描 skill_*.md → 解析 YAML frontmatter + body
  → 向后兼容: problems_solved+when_to_use → use_case
  → 按 estimated_impact 排序 (高→中→低)
  → 取 top_k 条
  → build_skill_context()
      → 格式化: name [impact] (type) / category / description / use_case / body[:2000]
      → 返回字符串 → 注入 Agent prompt 或 ToolResponse
```

### 工具注册链

```
ReActAgent.__init__(config=Config())  # subagent_enabled=True (default)
  → Agent.__init__()
    → if subagent_enabled and tool_registry:
        → _register_task_tool()
          → TaskTool(agent_factory=..., tool_registry=..., config=...)
          → self.tool_registry.register_tool(task_tool)
    → if skills_enabled: register SkillTool
    → if todowrite_enabled: register TodoWriteTool
    → if devlog_enabled: register DevLogTool

executor.add_tool(write_file)  # 用户工具
executor.add_tool(read_file)
...
```

---

## 11. DeepSeek v4-pro 推理过程可见性修复

### 问题

`DeepSeekAdapter.astream_invoke()` 和 `OpenAIAdapter.astream_invoke()` 中，`reasoning_content` 被**收集但不 yield**。Deepseek v4-pro 推理阶段只产生 `reasoning_content` 流，无 `content` 流。用户看到 Phase 2 卡死 5-15 秒，实际上是 LLM 正在推理。

### 修复

**文件**: `hello_agents/core/llm_adapters.py`

4 个流式方法全部改为推理内容**优先 yield**：

```python
# 改前
if getattr(delta, "content", None):
    yield delta.content          # 只 yield 正式内容
rc = getattr(delta, "reasoning_content", None)
if rc:
    reasoning_content = ...      # 收集但不 yield → 用户看不到

# 改后
rc = getattr(delta, "reasoning_content", None)
if rc:
    reasoning_content = (reasoning_content or "") + rc
    yield rc                     # 推理过程实时输出
if getattr(delta, "content", None):
    yield delta.content
```

涉及方法：
- `DeepSeekAdapter.stream_invoke()` — 同步流式
- `DeepSeekAdapter.astream_invoke()` — 异步流式
- `OpenAIAdapter.stream_invoke()` — 同步流式（thinking model 路径）
- `OpenAIAdapter.astream_invoke()` — 异步流式（thinking model 路径）

---

## 12. 跨关键词竞赛查重

### 问题

`_load_existing_competition_refs` 只读 `competitions_{当前keyword}.csv`，同一竞赛被不同关键词搜索时会重复下载。

### 修复

**文件**: `kaggle_knowledge/search_competitions.py`

```python
# 改前: 只读当前关键词的 CSV
safe_keyword = sanitize(keyword)
filepath = f"competitions_{safe_keyword}.csv"

# 改后: 扫描 output/ 下所有 competitions_*.csv，按 ref 字段去重
for filename in os.listdir(output_dir):
    if filename.startswith("competitions_") and filename.endswith(".csv"):
        # 读取每个 CSV，收集所有 ref
```

`search_competitions_paginated()` 中对跳过的竞赛打印 `跳过 N 个已处理的竞赛`，未跳过则不打印。

---

## 13. Skill 保存位置变更为竞赛 slug

### 问题

Skills 按用户搜索关键词分组保存（如 `skill_library/Hull-Tactical-Market-Prediction/`），导致同一竞赛的技能分散在不同目录。

### 修复

**文件**: `hello_agents/agents/kernel_skill_agent.py` — `run()` 和 `arun_stream()`

```python
# 改前
save_skill(keyword, skill, self.skill_library_dir)
# → skill_library/Hull-Tactical-Market-Prediction/skill_001_xxx.md

# 改后
comp_slug = skill.get("source_competition", keyword)
save_skill(comp_slug, skill, self.skill_library_dir)
# → skill_library/hulltacticalmarketprediction/skill_001_xxx.md
```

`comp_slug` 取自 competition `ref` 最后一个字段（如 `hull-tactical-market-prediction`），经由 `_sanitize_dirname` 去掉 `-` 变为目录名。清理旧文件也改为按 `comp_slug` 分组清理。

---

## 14. main.py vs main_kaggle.py 对比

### Kaggle 处理链路

两者当调用 kernel_skill 时走**完全相同的路径**：

```
ReActAgent → TaskTool.run() → _run_kernel_skill_streaming()
           → KernelSkillAgent.arun_stream()
           → ToolResponse → ReActAgent 收到结果
```

### 差异表

| 维度 | main.py | main_kaggle.py |
|------|---------|----------------|
| 用途 | A1模型训练 | Kaggle技能提取演示 |
| max_steps | 300 | 10 |
| 工具数 | 13 (7训练 + 6自动) | 6 (仅自动注册) |
| 事件类型检测 | `event.event_type` (bug → 返回"") | `event.type` (正确) |
| llm_chunk 处理 | 因 bug 永不匹配(安全网) | 显式 `pass` |
| kernel_skill 触发 | LLM 自主决定 | 提示词强制引导 |
| 记忆提炼 | MemoryManager.distill() | 无 |
| 工具结果截断 | 参数截断300字，结果不截断 | 结果完整展示 |

### 关键发现

main.py 的 `_event_type(event)` 访问不存在的 `event.event_type`（StreamEvent 只有 `.type`），永远返回 `""`，所有事件分支不匹配。这**碰巧**避免了 ReActAgent 直接 `print()` + 事件循环重复 `print()` 的双重输出乱码。不可"修复"此 bug，否则需要像 main_kaggle 一样将 llm_chunk 处理设为 `pass`。
