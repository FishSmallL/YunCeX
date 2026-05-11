"""Skill 库管理 — 保存、搜索、匹配。

Skill 以 SKILL.md 文件存储，按关键词目录组织：
  skill_library/<keyword>/skill_NNN_name.md

匹配算法：先精确匹配，失败后模糊匹配（Jaccard + Levenshtein）。
"""

import re
import os
import json
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime


def save_skill(keyword: str, skill_data: Dict, library_dir: str) -> str:
    """将一条 skill 保存为 SKILL.md 文件到对应关键词目录

    Args:
        keyword: 分类关键词（用作子目录名）
        skill_data: 包含 name, description, category, technique,
                    code_pattern, when_to_use, notes, source_kernel, source_competition
        library_dir: Skill 库根目录

    Returns:
        保存的文件路径
    """
    kw_safe = _sanitize_dirname(keyword)
    kw_dir = Path(library_dir) / kw_safe
    kw_dir.mkdir(parents=True, exist_ok=True)

    # 按编号自动命名
    existing = list(kw_dir.glob("skill_*.md"))
    next_num = len(existing) + 1
    name_slug = skill_data.get("name", "unnamed")
    filename = f"skill_{next_num:03d}_{name_slug}.md"
    filepath = kw_dir / filename

    created = datetime.now().strftime("%Y-%m-%d")
    content = _render_skill_md(skill_data, created)

    filepath.write_text(content, encoding="utf-8")
    return str(filepath)


def _render_skill_md(skill_data: Dict, created: str) -> str:
    """将 skill dict 渲染为 SKILL.md 格式（YAML frontmatter + Markdown body）"""
    name = skill_data.get("name", "unnamed")
    description = skill_data.get("description", "")
    problems_solved = skill_data.get("problems_solved", "")
    keyword = skill_data.get("keyword", "")
    category = skill_data.get("category", "")
    source_kernel = skill_data.get("source_kernel", "")
    source_competition = skill_data.get("source_competition", "")
    technique = skill_data.get("technique", "")
    code_pattern = skill_data.get("code_pattern", "")
    when_to_use = skill_data.get("when_to_use", "")
    notes = skill_data.get("notes", "")

    # 转义 YAML 中的特殊字符
    def yaml_str(s):
        if any(c in s for c in ['"', "'", ":", "#", "{", "}", "[", "]", ",", "&", "*", "?", "|", "-", "<", ">", "=", "!", "%", "@", "`"]):
            return f'"{s.replace(chr(34), chr(92)+chr(34))}"'
        return s if s else '""'

    frontmatter = f"""---
name: {name}
description: {yaml_str(description)}
problems_solved: {yaml_str(problems_solved)}
keyword: {keyword}
category: {category}
source_kernel: {source_kernel}
source_competition: {source_competition}
created: {created}
---

# {name.replace('-', ' ').title()}

## 可解决的问题
{problems_solved if problems_solved else '通用机器学习问题'}

## 技巧说明
{technique}

## 适用场景
{when_to_use if when_to_use else '通用场景'}

## 代码模式
```python
{code_pattern if code_pattern else '# 无具体代码'}
```

## 注意事项
{notes if notes else '无特殊注意事项'}
"""
    return frontmatter


def match_keyword(
    query: str, library_dir: str, threshold: float = 0.6
) -> Optional[str]:
    """将查询关键词匹配到 skill 库中的分类目录。

    两阶段匹配：
    1. 精确：sanitize 查询字符串，检查目录是否存在
    2. 模糊：计算 Jaccard + Levenshtein 综合得分，返回高于阈值的最佳匹配

    Args:
        query: 查询关键词
        library_dir: Skill 库根目录
        threshold: 模糊匹配的最低得分阈值（0.0-1.0）

    Returns:
        匹配到的目录名，未匹配则返回 None
    """
    lib = Path(library_dir)
    if not lib.is_dir():
        return None

    dirs = [d.name for d in lib.iterdir() if d.is_dir()]
    if not dirs:
        return None

    # 第一阶段：精确匹配
    result = _exact_match(query, dirs)
    if result:
        return result

    # 第二阶段：模糊匹配
    return _fuzzy_match(query, dirs, threshold)


def _exact_match(query: str, dirs: List[str]) -> Optional[str]:
    sanitized = _sanitize_dirname(query)
    if sanitized in dirs:
        return sanitized
    # 也尝试原始字符串（未经 sanitize）
    if query.strip() in dirs:
        return query.strip()
    return None


def _fuzzy_match(
    query: str, dirs: List[str], threshold: float
) -> Optional[str]:
    best_score = 0.0
    best_dir = None
    query_tokens = set(_tokenize(query))
    query_str = _sanitize_dirname(query)

    for d in dirs:
        d_tokens = set(_tokenize(d))
        # Jaccard 相似度（token 重叠率）
        intersection = query_tokens & d_tokens
        union = query_tokens | d_tokens
        jaccard = len(intersection) / len(union) if union else 0.0

        # 归一化 Levenshtein 相似度（字符串编辑距离）
        lev = _levenshtein_similarity(query_str, d)

        # 综合得分
        score = 0.5 * jaccard + 0.5 * lev
        if score > best_score:
            best_score = score
            best_dir = d

    if best_score >= threshold and best_dir:
        return best_dir
    return None


def _tokenize(text: str) -> List[str]:
    """对文本做分词，用于 Jaccard 比较"""
    text = text.lower().strip()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return [t for t in text.split() if len(t) > 1]


def _levenshtein_similarity(a: str, b: str) -> float:
    """归一化 Levenshtein 相似度（1.0 = 完全相同）"""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0

    # 只比较前 50 个字符以提高效率
    a, b = a[:50], b[:50]

    # 动态规划
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        dp[i][0] = i
    for j in range(len(b) + 1):
        dp[0][j] = j

    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )

    max_len = max(len(a), len(b))
    return 1.0 - dp[len(a)][len(b)] / max_len


def search_skills(
    keyword: str, library_dir: str, top_k: int = 5
) -> List[Dict]:
    """获取某个关键词分类下的 top-k 条 skill

    Args:
        keyword: 已匹配的分类目录名
        library_dir: Skill 库根目录
        top_k: 最多返回的技能数

    Returns:
        含 name、description、body 的 skill dict 列表
    """
    kw_dir = Path(library_dir) / keyword
    if not kw_dir.is_dir():
        return []

    skills = []
    for md_file in sorted(kw_dir.glob("skill_*.md")):
        try:
            content = md_file.read_text(encoding="utf-8")
            frontmatter, body = _parse_skill_md(content)
            if frontmatter:
                skills.append(
                    {
                        "name": frontmatter.get("name", md_file.stem),
                        "description": frontmatter.get("description", ""),
                        "category": frontmatter.get("category", ""),
                        "body": body,
                        "file": str(md_file),
                    }
                )
        except Exception:
            continue

    # 按创建时间倒序（文件名排序 ≈ 创建顺序），返回最新的 k 条
    return skills[-top_k:][::-1] if len(skills) > top_k else skills[::-1]


def _parse_skill_md(content: str):
    """解析 SKILL.md 内容为 (frontmatter_dict, body_str)"""
    import yaml

    match = re.match(r'^---\s*\n(.*?)\n---\s*\n(.*)$', content, re.DOTALL)
    if not match:
        return None, content

    yaml_str, body = match.groups()
    try:
        frontmatter = yaml.safe_load(yaml_str) or {}
    except Exception:
        return None, body

    return frontmatter, body.strip()


def build_skill_context(
    keyword: str, library_dir: str, top_k: int = 5
) -> str:
    """将 top-k 条 skill 构建为紧凑的上下文字符串，用于注入 Agent prompt

    Args:
        keyword: 已匹配的分类目录名
        library_dir: Skill 库根目录
        top_k: 最多包含的技能数

    Returns:
        格式化字符串，可注入 Agent system prompt 或上下文
    """
    skills = search_skills(keyword, library_dir, top_k)
    if not skills:
        return ""

    lines = [f"## 历史竞赛技巧库 (关键词: {keyword})"]
    for i, s in enumerate(skills, 1):
        lines.append(f"\n### 技巧{i}: {s['name']}")
        lines.append(f"分类: {s.get('category', 'N/A')}")
        lines.append(f"描述: {s['description']}")
        # 保留 body 但截断到 1000 字符
        body = s["body"]
        if len(body) > 1000:
            body = body[:1000] + "\n...(已截断)"
        lines.append(body)

    return "\n".join(lines)


def list_keywords(library_dir: str) -> List[str]:
    """列出 skill 库中所有可用的关键词分类"""
    lib = Path(library_dir)
    if not lib.is_dir():
        return []
    return sorted(
        [d.name for d in lib.iterdir() if d.is_dir()]
    )


def get_library_stats(library_dir: str) -> Dict:
    """获取 skill 库的统计信息"""
    lib = Path(library_dir)
    if not lib.is_dir():
        return {"keywords": 0, "total_skills": 0, "categories": {}}

    stats = {"keywords": 0, "total_skills": 0, "categories": {}}
    for kw_dir in lib.iterdir():
        if not kw_dir.is_dir():
            continue
        skill_files = list(kw_dir.glob("skill_*.md"))
        stats["keywords"] += 1
        stats["total_skills"] += len(skill_files)

        # 按类别统计
        for sf in skill_files:
            try:
                content = sf.read_text(encoding="utf-8")
                fm, _ = _parse_skill_md(content)
                if fm:
                    cat = fm.get("category", "未分类")
                    stats["categories"][cat] = (
                        stats["categories"].get(cat, 0) + 1
                    )
            except Exception:
                pass

    return stats


def _sanitize_dirname(name: str) -> str:
    """将字符串转换为安全的目录名"""
    name = name.lower().strip()
    name = re.sub(r'[^a-z0-9_\s]', '', name)
    name = re.sub(r'\s+', '_', name)
    return name[:80]
