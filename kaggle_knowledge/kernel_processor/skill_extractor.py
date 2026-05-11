"""基于 LLM 的 notebook 技巧提取

用精心设计的 prompt 让 LLM 从 Kaggle kernel notebook 中提取可复用的 ML 技巧，
聚焦 5 个类别。
"""

import json
import re
from typing import List, Dict, Optional
from pathlib import Path


# 提取的 5 个关注类别
CATEGORIES = [
    "数据预处理",
    "特征工程",
    "模型架构",
    "集成策略",
    "技巧",
]

EXTRACTION_PROMPT = """你是一个竞赛数据科学专家。分析以下 Kaggle kernel notebook，提取可复用的技巧。

## 提取规则
扫描 notebook 内容，从以下6个类别中提取技巧（没有则跳过）：
1. **数据预处理** - 缺失值处理、异常值检测、数据清洗、类型转换等
2. **特征工程** - 特征创建、编码、选择、缩放、组合等
3. **模型架构** - 模型选择、层级设计、超参数配置、损失函数选择等
4. **集成策略** - 模型融合、Stacking、Blending、投票、加权等
5. **调试技巧** - 验证策略、问题排查方法、性能分析等

## 输出格式
返回 JSON 数组，每个元素一个技巧：
```json
[
  {
    "name": "英文短名-用连字符",
    "description": "一句话说明该技巧解决什么问题（≤300字）。不要循环定义，要指出适用的问题类型。",
    "problems_solved": "该技巧可解决的具体问题列表，逗号分隔。如：高基数类别特征编码、缺失值模式保留、类别不平衡分类、时间序列特征构建、多模型方差缩减",
    "category": "特征工程",
    "technique": "详细说明（≤500字）。必须包含：1)具体怎么做 2)为什么这样做有效 3)相比常见做法的优势",
    "code_pattern": "可执行的代码片段，必须包含 import 语句，禁止使用 # 占位注释（如 # 这里做XX），必须是真实有用的代码",
    "when_to_use": "什么场景下应该使用（指明数据条件、问题类型、模型类型）",
    "notes": "注意事项：什么情况下可能失效、调参建议、常见错误"
  }
]
```

## 重要原则
- 每类最多提取3个最重要的技巧
- description ≤300字，必须明确指出能解决什么类型的问题
- technique ≤500字，必须解释原理和为什么有效，不能只罗列步骤
- code_pattern 必须是可执行的完整片段（含 import），严禁伪代码或占位注释
- problems_solved 列出该技巧适用的具体问题类型，帮助后续检索匹配
- 只提取**可迁移**的通用技巧，不要竞赛特定信息（如特定列名、路径、业务含义）
- 如果某个类别没有值得提取的内容，直接跳过
- 关注 notebook 中**实际实现**的技巧，而不是泛泛而谈

## Notebook 内容
{notebook_text}

## 提取结果
直接返回 JSON 数组，不要额外解释："""


def extract_skills_from_notebook(
    notebook_text: str,
    kernel_name: str,
    competition: str,
    keyword: str,
    llm,
) -> List[Dict]:
    """用 LLM 从 notebook 文本中提取结构化技巧

    Args:
        notebook_text: 展平后的 notebook 文本
        kernel_name: 来源 kernel 名称
        competition: 竞赛 slug
        keyword: 分类关键词
        llm: HelloAgentsLLM 实例

    Returns:
        含 frontmatter 字段的 skill dict 列表
    """
    prompt = EXTRACTION_PROMPT.replace("{notebook_text}", notebook_text)

    try:
        response = llm.invoke(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=8192,
        )
        raw = response.content.strip()
    except Exception as e:
        print(f"  LLM 提取失败: {e}")
        return []

    skills = _parse_llm_response(raw, kernel_name, competition, keyword)
    return skills


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
        skill = {
            "name": name,
            "description": item.get("description", "")[:300],
            "problems_solved": item.get("problems_solved", ""),
            "keyword": keyword,
            "category": category,
            "source_kernel": kernel_name,
            "source_competition": competition,
            "technique": item.get("technique", item.get("description", "")),
            "code_pattern": item.get("code_pattern", ""),
            "when_to_use": item.get("when_to_use", ""),
            "notes": item.get("notes", ""),
        }
        skills.append(skill)

    return skills


def _extract_json_array(raw: str):
    """用括号平衡计数从 LLM 原始返回中提取 JSON 数组。

    正确处理 JSON 内容中包含 triple backticks、转义引号、
    嵌套括号等复杂情况。
    """
    text = raw.strip()

    # 去除 markdown 代码围栏
    if text.startswith("```"):
        text = re.sub(r'^```\w*\s*\n?', '', text)
        text = re.sub(r'\n?```\s*$', '', text)

    # 找到 JSON 数组起始位置
    start = text.find('[')
    if start == -1:
        # 尝试找 JSON 对象（单条 skill 返回）
        start = text.find('{')
        if start == -1:
            return None

    # 括号平衡状态机
    depth = 0
    in_string = False
    escape = False
    bracket_map = {'[': ']', '{': '}'}
    open_stack = []  # 记录开括号类型

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
                    continue  # 括号类型不匹配，继续
            depth -= 1
            if depth == 0:
                json_str = text[start:i + 1]
                try:
                    result = json.loads(json_str)
                    if isinstance(result, list):
                        return result
                    elif isinstance(result, dict):
                        return [result]
                except json.JSONDecodeError as e:
                    print(f"  [DEBUG] JSON 解析失败: {e}")
                    print(f"  [DEBUG] 提取内容 ({len(json_str)} 字): {json_str[:300]}")
                    return None

    print(f"  [DEBUG] JSON 未闭合（depth={depth}），LLM 返回可能被截断（共 {len(text)} 字）")
    return None


def _sanitize_name(name: str) -> str:
    """将技能名转换为文件系统安全的 slug"""
    name = name.lower().strip()
    name = re.sub(r'[^\w\s-]', '', name)
    name = re.sub(r'[\s_]+', '-', name)
    name = re.sub(r'-+', '-', name)
    return name[:60]


def extract_skills_from_notebooks_batch(
    kernel_dirs: List[str],
    keyword: str,
    llm,
) -> List[Dict]:
    """批量处理多个 kernel 目录，提取所有技巧

    Args:
        kernel_dirs: kernel 目录路径列表（每个目录包含 .ipynb + metadata）
        keyword: 分类关键词
        llm: HelloAgentsLLM 实例

    Returns:
        聚合后的所有提取技巧列表
    """
    from .notebook_parser import (
        parse_notebook,
        notebook_to_text,
        get_kernel_competition,
    )

    all_skills = []
    for kdir in kernel_dirs:
        kpath = Path(kdir)
        if not kpath.is_dir():
            continue

        # 查找 .ipynb 文件
        ipynb_files = list(kpath.glob("*.ipynb"))
        if not ipynb_files:
            print(f"  {kdir} 中未找到 .ipynb，跳过")
            continue

        ipynb_path = str(ipynb_files[0])
        kernel_name = kpath.name

        # 从 metadata 获取竞赛信息
        meta_files = list(kpath.glob("kernel-metadata.json"))
        competition = (
            get_kernel_competition(str(meta_files[0]))
            if meta_files
            else "unknown"
        )

        print(f"  处理中: {kernel_name} (竞赛: {competition})")

        try:
            cells = parse_notebook(ipynb_path)
            text = notebook_to_text(cells, max_len=8000)
            skills = extract_skills_from_notebook(
                text, kernel_name, competition, keyword, llm
            )
            print(f"    提取到 {len(skills)} 条技巧")
            all_skills.extend(skills)
        except Exception as e:
            print(f"    处理 {kernel_name} 时出错: {e}")
            continue

    return all_skills
