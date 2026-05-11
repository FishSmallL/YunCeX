"""解析 .ipynb 文件，提取结构化 cell 列表。

支持两种格式：
- 标准 nbformat v4 JSON（{"cells": [{"cell_type": "...", "source": [...]}]}）
- Kaggle XML 风格（<cell id="..."><cell_type>code</cell_type>...</cell>）
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Optional


def parse_notebook(ipynb_path: str) -> List[Dict]:
    """解析 .ipynb 文件，返回 cell 列表。

    每个 cell: {"cell_type": "code"|"markdown"|"raw", "source": str, "outputs": []}
    """
    path = Path(ipynb_path)
    if not path.exists():
        raise FileNotFoundError(f"Notebook 文件不存在: {ipynb_path}")

    content = path.read_text(encoding="utf-8").strip()

    if content.startswith("{"):
        return _parse_nbformat(content)
    elif content.startswith("<cell"):
        return _parse_xml_format(content)
    else:
        raise ValueError(f"无法识别的 notebook 格式: {ipynb_path}")


def _parse_nbformat(content: str) -> List[Dict]:
    """解析标准 Jupyter nbformat JSON 格式"""
    nb = json.loads(content)
    cells = []
    for c in nb.get("cells", []):
        source = "".join(c.get("source", []))
        cell_type = c.get("cell_type", "code")
        outputs = []
        if cell_type == "code":
            for o in c.get("outputs", []):
                text = "".join(o.get("text", []))
                if text:
                    outputs.append(text)
        cells.append({"cell_type": cell_type, "source": source.strip(), "outputs": outputs})
    return cells


def _parse_xml_format(content: str) -> List[Dict]:
    """解析 Kaggle XML 风格的 .ipynb 格式

    格式示例：
      <cell id="...">代码内容</cell id="...">
      <cell id="..."><cell_type>markdown</cell_type>Markdown 内容</cell id="...">
    """
    cells = []
    # 匹配每个 cell 块：<cell id="...">...</cell id="...">
    # 关闭标签包含相同的 id
    pattern = re.compile(
        r'<cell\s+id="([^"]*)">(.*?)</cell\s+id="\1">', re.DOTALL
    )
    for match in pattern.finditer(content):
        inner = match.group(2)

        # 检查是否嵌入了 cell_type 标签
        ct_match = re.match(r'<cell_type>(\w+)</cell_type>', inner)
        if ct_match:
            cell_type = ct_match.group(1)
            source = inner[ct_match.end():].strip()
        else:
            cell_type = "code"
            source = inner.strip()

        cells.append({"cell_type": cell_type, "source": source, "outputs": []})

    return cells


def extract_code_comments(code: str) -> List[str]:
    """从 Python 代码中提取单行注释（# 开头）和行内注释"""
    comments = []
    for line in code.split("\n"):
        stripped = line.strip()
        if stripped.startswith("#"):
            comments.append(stripped.lstrip("# ").strip())
        elif "#" in stripped:
            # 行内注释：提取 # 之后的部分
            code_part, _, comment = stripped.partition("#")
            if code_part.strip() and comment.strip():
                comments.append(comment.strip())
    return comments


def notebook_to_text(cells: List[Dict], max_len: int = 8000) -> str:
    """将 notebook cell 列表展平为紧凑文本，用于 LLM 输入。

    代码 cell 按 cell 截断；Markdown cell 保留较短的版本。
    """
    parts = []
    total = 0
    for cell in cells:
        ct = cell["cell_type"]
        source = cell["source"]
        if not source:
            continue

        if ct == "markdown":
            label = "## MARKDOWN"
            text = source[:500]
            if len(source) > 500:
                text += "..."
        elif ct == "code":
            label = "## CODE"
            text = source[:800]
            if len(source) > 800:
                text += "\n# ... (truncated)"
            # 预先提取注释作为上下文
            comments = extract_code_comments(source)
            if comments:
                text = "# Key comments: " + "; ".join(comments[:5]) + "\n" + text
        else:
            label = "## RAW"
            text = source[:300]

        block = f"{label}\n{text}\n"
        if total + len(block) > max_len:
            parts.append(f"## ... (剩余 {len(cells) - len(parts)} 个 cell 已截断)")
            break
        parts.append(block)
        total += len(block)

    return "\n".join(parts)


def get_kernel_competition(metadata_path: str) -> Optional[str]:
    """从 kernel-metadata.json 中提取竞赛名称"""
    try:
        meta = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        sources = meta.get("competition_sources", [])
        return sources[0] if sources else None
    except Exception:
        return None
