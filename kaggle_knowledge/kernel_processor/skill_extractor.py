"""基于 LLM 的 notebook 技巧提取

用设计的 prompt 让 LLM 从 Kaggle kernel notebook 中提取可复用的竞赛技巧，
覆盖 ML/DL/CV/NLP/时序等各类竞赛，支持两阶段提取和技能去重。
"""

import json
import re
from typing import List, Dict, Optional
from pathlib import Path


# 提取的 8 个关注类别（覆盖 ML/DL/CV/NLP/时序）
CATEGORIES = [
    "数据处理",
    "特征与表示",
    "模型设计",
    "训练策略",
    "验证与评估",
    "数据增强",
    "损失与指标",
    "集成与后处理",
]

# 两阶段提取：Phase 1 亮点扫描
SCAN_PROMPT = """你是一个 Kaggle 竞赛专家。快速浏览以下 notebook，识别 2-4 个最有价值的技术亮点。

## 要求
- 每个亮点一句话描述（≤100字）
- 标注所属类别（从以下8类中选择）：数据处理、特征与表示、模型设计、训练策略、验证与评估、数据增强、损失与指标、集成与后处理
- 标注预期影响力（高/中/低）：判断依据是该技术在 notebook 中的篇幅和是否是核心创新
- 只关注**实际实现**的代码技巧，跳过泛泛而谈的文字
- **如果 notebook 中没有值得提取的技术亮点（如仅有环境安装、pip install、import 等），直接返回 [] 空数组，不要输出任何解释文字**

## 输出格式
直接返回 JSON 数组：
```json
[
  {"highlight": "...", "category": "模型设计", "impact": "高"}
]
```
没有亮点时：
```json
[]
```

## notebook 内容
{notebook_text}

## 扫描结果
直接返回 JSON 数组，不要额外解释："""

# 两阶段提取：Phase 2 深度提取
DEEP_EXTRACT_PROMPT = """你是一个 Kaggle 竞赛专家。以下是从一个 kernel 中识别到的技术亮点，请针对每个亮点深度提取结构化技巧。

## 亮点列表
{highlights}

## 对应的 notebook 代码
{notebook_text}

## 提取要求
参考上述亮点，从以下 8 个类别中提取技巧（没有则跳过）：
1. **数据处理** 2. **特征与表示** 3. **模型设计** 4. **训练策略**
5. **验证与评估** 6. **数据增强** 7. **损失与指标** 8. **集成与后处理**

## 输出格式
```json
[
  {
    "name": "英文短名-用连字符",
    "description": "一句话说明该技巧解决什么问题（≤200字）",
    "category": "模型设计",
    "competition_type": "图像分类",
    "estimated_impact": "高",
    "technique": "详细说明（≤500字）。包含: 1)具体做法 2)为什么有效 3)相比常见做法的优势",
    "code_pattern": "可执行的代码片段（含import），基于 notebook 中的实际代码",
    "use_case": "解决什么问题 | 什么场景适用",
    "notes": "失效场景、调参建议、常见错误"
  }
]
```

## 提取结果
直接返回 JSON 数组，不要额外解释："""


def _parse_llm_response(
    raw: str, kernel_name: str, competition: str, keyword: str
) -> List[Dict]:
    """解析 LLM 返回的 JSON，填充完整的 skill frontmatter 字段。

    用括号平衡计数替代正则，正确处理：
    - code_pattern 中包含 triple backticks
    - LLM 在 JSON 前后添加解释文字
    - LLM 漏写 closing fence
    """
    items = _extract_json_array(raw)
    if items is None:
        print(f"  LLM 返回无法解析为 JSON（共 {len(raw)} 字）")
        if len(raw) < 500:
            print(f"  原始返回: {raw}")
        return []

    skills = []
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        name = _sanitize_name(item.get("name", f"skill-{idx+1}"))
        category = item.get("category", "未分类")
        # 合并 problems_solved + when_to_use → use_case
        use_case = item.get("use_case", "")
        if not use_case:
            ps = item.get("problems_solved", "")
            wtu = item.get("when_to_use", "")
            use_case = f"解决: {ps} | 场景: {wtu}" if ps and wtu else (ps or wtu)
        skill = {
            "name": name,
            "description": item.get("description", "")[:200],
            "use_case": use_case,
            "keyword": keyword,
            "category": category,
            "competition_type": item.get("competition_type", ""),
            "estimated_impact": item.get("estimated_impact", ""),
            "source_kernel": kernel_name,
            "source_competition": competition,
            "technique": item.get("technique", item.get("description", "")),
            "code_pattern": item.get("code_pattern", ""),
            "notes": item.get("notes", ""),
        }
        skills.append(skill)

    return skills


def _try_parse_json(text: str, start: int):
    """从指定位置尝试解析 JSON，返回收集到的对象列表，失败返回 None。

    支持: 完整数组 [{...}], 紧凑数组 [{...},{...}], 连续独立对象 {...}{...}
    """
    depth = 0
    in_string = False
    escape = False
    bracket_map = {'[': ']', '{': '}'}
    open_stack = []
    collected = []

    for i in range(start, len(text)):
        ch = text[i]

        if escape:
            escape = False
            continue
        if ch == '\\':
            escape = True
            continue
        if ch == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue

        if ch in ('[', '{'):
            depth += 1
            open_stack.append(bracket_map[ch])
        elif ch in (']', '}'):
            if open_stack:
                expected = open_stack.pop()
                if ch != expected:
                    continue
            depth -= 1
            if depth == 0:
                json_str = text[start:i + 1]
                try:
                    result = json.loads(json_str)
                    if isinstance(result, list):
                        return result
                    elif isinstance(result, dict):
                        collected.append(result)
                        # 继续向后找下一个对象
                        next_start = -1
                        for m in re.finditer(r'\{\s*"name"', text[i + 1:]):
                            next_start = i + 1 + m.start()
                            break
                        if next_start == -1:
                            return collected if collected else None
                        start = next_start
                        depth = 0
                        open_stack = []
                except json.JSONDecodeError:
                    # 继续向后找下一个可能的起始位置
                    next_start = -1
                    for m in re.finditer(r'\{\s*"name"', text[i + 1:]):
                        next_start = i + 1 + m.start()
                        break
                    if next_start == -1:
                        return collected if collected else None
                    start = next_start
                    depth = 0
                    open_stack = []

    return collected if collected else None


def _extract_json_array(raw: str):
    """从 LLM 原始返回中提取 JSON 数组，鲁棒处理各种格式。

    支持: 格式化JSON(换行缩进)、紧凑JSON、连续独立对象、markdown围栏
    """
    text = raw.strip()

    # 去除 markdown 代码围栏
    if text.startswith("```"):
        text = re.sub(r'^```\w*\s*\n?', '', text)
        text = re.sub(r'\n?```\s*$', '', text)

    # 找到 JSON 数组起始位置
    # 策略：搜索所有可能的 '[' 起始位置，尝试解析，取最长有效结果
    candidates = []

    # 找所有可能是 JSON 数组起始的 '[' 位置
    # 排除 markdown 中的 [类别]（前面是 '- ' 或 '* '）
    for m in re.finditer(r'\[', text):
        pos = m.start()
        # 跳过 markdown 列表标签: "- [xxx]" 或 "* [xxx]"
        prefix = text[max(0, pos - 2):pos]
        if prefix in ('- ', '* '):
            continue
        # 跳过内联代码中的 "[" (前面是字母/数字/下划线)
        if pos > 0 and text[pos - 1].isalnum():
            continue
        candidates.append(pos)

    # 对每个候选位置尝试解析，取第一个成功或最长的结果
    best_result = None
    for start in candidates:
        result = _try_parse_json(text, start)
        if result and (best_result is None or len(result) > len(best_result)):
            best_result = result

    if best_result:
        return best_result

    # 回退：尝试找单个 JSON 对象
    for m in re.finditer(r'\{\s*"name"', text):
        result = _try_parse_json(text, m.start())
        if result:
            return result

    return None


def _sanitize_name(name: str) -> str:
    """将技能名转换为文件系统安全的 slug"""
    name = name.lower().strip()
    name = re.sub(r'[^\w\s-]', '', name)
    name = re.sub(r'[\s_]+', '-', name)
    name = re.sub(r'-+', '-', name)
    return name[:60]


# ── 两阶段提取 ──

def extract_skills_2pass(
    notebook_text: str,
    kernel_name: str,
    competition: str,
    keyword: str,
    llm,
) -> List[Dict]:
    """两阶段提取：先扫描亮点 → 再深度提取每个亮点

    Args:
        notebook_text: 展平后的 notebook 文本
        kernel_name: 来源 kernel 名称
        competition: 竞赛 slug
        keyword: 分类关键词
        llm: HelloAgentsLLM 实例

    Returns:
        结构化 skill dict 列表
    """
    # Phase 1: 亮点扫描
    scan_prompt = SCAN_PROMPT.replace("{notebook_text}", notebook_text)
    highlights = []
    try:
        response = llm.invoke(
            messages=[{"role": "user", "content": scan_prompt}],
            max_tokens=2048,
        )
        raw = response.content.strip()
        items = _extract_json_array(raw)
        if items:
            highlights = [
                h for h in items
                if isinstance(h, dict) and h.get("highlight")
            ]
    except Exception as e:
        print(f"  Phase 1 扫描失败: {e}")
        return []

    if not highlights:
        print(f"  Phase 1 未发现技术亮点，跳过提取")
        return []

    print(f"  Phase 1: 识别到 {len(highlights)} 个亮点")

    # Phase 2: 深度提取每一个亮点
    hl_text = "\n".join(
        f"- [{h.get('category','?')}] {h.get('highlight','')} (impact: {h.get('impact','?')})"
        for h in highlights
    )
    deep_prompt = DEEP_EXTRACT_PROMPT.replace("{highlights}", hl_text)
    deep_prompt = deep_prompt.replace("{notebook_text}", notebook_text)

    try:
        response = llm.invoke(
            messages=[{"role": "user", "content": deep_prompt}],
            max_tokens=8192,
        )
        raw = response.content.strip()
    except Exception as e:
        print(f"  Phase 2 提取失败: {e}")
        return []

    skills = _parse_llm_response(raw, kernel_name, competition, keyword)
    return skills


# ── 去重合并 ──

def dedup_skills(skills: List[Dict], llm=None) -> List[Dict]:
    """两阶段去重合并：
    1. 精确去重：同 category + 同 name → 保留最佳版本（无 LLM 成本）
    2. 语义去重：同 category 下 Jaccard 粗筛 + LLM 确认 → 合并相似技能
    """
    if len(skills) <= 1:
        return skills

    # 阶段 1: 精确去重（规则匹配）
    groups: Dict[tuple, List[Dict]] = {}
    for s in skills:
        key = (s.get("category", ""), _sanitize_name(s.get("name", "")))
        groups.setdefault(key, []).append(s)

    result = []
    for group in groups.values():
        if len(group) == 1:
            result.append(group[0])
        else:
            best = max(group, key=_skill_score)
            for other in group:
                if other is best:
                    continue
                best["use_case"] = _merge_str(
                    best.get("use_case", ""),
                    other.get("use_case", ""),
                    sep=" | ",
                )
                if other.get("notes"):
                    best["notes"] += (
                        f"\n(来自 {other['source_kernel']}): {other['notes']}"
                    )
                if other["source_kernel"] not in best.get("source_kernel", ""):
                    best["source_kernel"] += f", {other['source_kernel']}"
            result.append(best)

    # 阶段 2: 语义去重（LLM 确认）
    if llm is not None:
        result = _semantic_dedup(result, llm)

    return result


def _skill_score(s: Dict) -> int:
    """对一条 skill 打分，用于去重时择优"""
    code_score = min(len(s.get("code_pattern", "")) // 200, 3)
    tech_score = min(len(s.get("technique", "")) // 150, 3)
    desc_score = 1 if len(s.get("description", "")) > 50 else 0
    impact = s.get("estimated_impact", "")
    impact_bonus = 2 if impact == "高" else (1 if impact == "中" else 0)
    return code_score + tech_score + desc_score + impact_bonus


def _merge_str(a: str, b: str, sep: str = ",") -> str:
    """合并两个字符串，按分隔符去重"""
    if not a:
        return b
    if not b:
        return a
    parts_a = set(p.strip() for p in a.split(sep) if p.strip())
    parts_b = set(p.strip() for p in b.split(sep) if p.strip())
    merged = parts_a | parts_b
    return sep.join(sorted(merged))


# ── 语义去重 ──

def _jaccard_similarity(text_a: str, text_b: str) -> float:
    """计算两段文本的 Jaccard 分词重叠率"""
    def _tokenize(text: str):
        text = text.lower().strip()
        # 保留中文字符、英文单词、数字
        tokens = []
        # 中文单字分词
        chinese = re.findall(r'[一-鿿]+', text)
        for seg in chinese:
            tokens.extend(seg)  # 逐字
        # 英文/数字分词
        eng = re.findall(r'[a-z0-9]+', text)
        tokens.extend(eng)
        return set(tokens)

    tokens_a = _tokenize(text_a)
    tokens_b = _tokenize(text_b)
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union) if union else 0.0


def _semantic_dedup(skills: List[Dict], llm) -> List[Dict]:
    """语义去重：同 category 下，用 Jaccard 粗筛 + LLM 确认是否同一技巧。

    对 Jaccard 相似度 > 0.45 的候选对，批量询问 LLM 判断是否本质相同。
    LLM 确认相同 → 合并，保留评分高的版本。
    """
    if len(skills) <= 1:
        return skills

    # 按 category 分组
    by_cat: Dict[str, List[Dict]] = {}
    for s in skills:
        cat = s.get("category", "未分类")
        by_cat.setdefault(cat, []).append(s)

    # 对每组内，生成候选对
    candidates: List[tuple] = []  # [(idx_a, idx_b, skill_a, skill_b, jaccard)]
    for cat, group in by_cat.items():
        if len(group) <= 1:
            continue
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                text_a = (
                    group[i].get("description", "") + " "
                    + group[i].get("use_case", "") + " "
                    + group[i].get("technique", "")[:200]
                )
                text_b = (
                    group[j].get("description", "") + " "
                    + group[j].get("use_case", "") + " "
                    + group[j].get("technique", "")[:200]
                )
                sim = _jaccard_similarity(text_a, text_b)
                if sim > 0.45:
                    candidates.append((i, j, group[i], group[j], sim))

    if not candidates:
        return skills

    # 批量 LLM 确认
    if len(candidates) == 1:
        _, _, sa, sb, _ = candidates[0]
        prompt = (
            f"技巧A: {sa.get('name','')}: {sa.get('description','')}\n"
            f"技巧B: {sb.get('name','')}: {sb.get('description','')}\n"
            "它们是否本质上是同一技巧？仅回复 yes 或 no。"
        )
    else:
        lines = ["以下是一些技巧对，判断每对是否本质上是同一技巧（仅回复编号+yes/no）："]
        for idx, (_, _, sa, sb, _) in enumerate(candidates, 1):
            lines.append(
                f"{idx}. A:{sa.get('name','')} B:{sb.get('name','')}"
            )
        prompt = "\n".join(lines)

    try:
        response = llm.invoke(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=min(len(candidates) * 15, 256),
        )
        answers = response.content.strip().lower()
    except Exception:
        return skills

    # 解析 LLM 回复
    merge_pairs: set = set()  # 记录要合并的 (idx_a, idx_b)
    if len(candidates) == 1:
        if "yes" in answers:
            merge_pairs.add((0, 1))
    else:
        for idx, (_, _, _, _, _) in enumerate(candidates, 1):
            # 找 "1.yes" 或 "1: yes" 等模式
            pattern = rf'{idx}[.\s:)]*\s*(yes|no)'
            match = re.search(pattern, answers)
            if match and match.group(1) == "yes":
                merge_pairs.add((idx - 1, idx - 1))

    if not merge_pairs:
        return skills

    # 执行合并
    # 构建一个 mapping: original_index → merged_to_index（-1 表示被合并掉了）
    n = len(skills)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx == ry:
            return
        # 保留评分高的
        score_x = _skill_score(skills[rx])
        score_y = _skill_score(skills[ry])
        if score_x >= score_y:
            parent[ry] = rx
        else:
            parent[rx] = ry

    for (i, j, sa, sb, _) in candidates:
        # 找到它们在全局 skills 列表中的实际索引
        idx_a = next(idx for idx, s in enumerate(skills) if s is sa)
        idx_b = next(idx for idx, s in enumerate(skills) if s is sb)
        union(idx_a, idx_b)

    # 对每个 root，合并其组内所有其他技能
    roots: Dict[int, List[int]] = {}
    for i in range(n):
        r = find(i)
        roots.setdefault(r, []).append(i)

    result = []
    removed_indices = set()
    for root_idx, group_indices in roots.items():
        if len(group_indices) == 1:
            result.append(skills[root_idx])
        else:
            best_idx = root_idx
            best = skills[best_idx]
            for other_idx in group_indices:
                if other_idx == best_idx:
                    continue
                removed_indices.add(other_idx)
                other = skills[other_idx]
                best["use_case"] = _merge_str(
                    best.get("use_case", ""),
                    other.get("use_case", ""),
                    sep=" | ",
                )
                if other.get("notes"):
                    best["notes"] += (
                        f"\n(来自 {other['source_kernel']}): {other['notes']}"
                    )
                if other["source_kernel"] not in best.get("source_kernel", ""):
                    best["source_kernel"] += f", {other['source_kernel']}"
            result.append(best)

    if removed_indices:
        print(f"  语义去重: 合并了 {len(removed_indices)} 条重复技能")

    return result


# ── 批量提取（保留兼容） ──

