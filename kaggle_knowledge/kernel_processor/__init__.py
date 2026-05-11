"""Kernel Processor - 从 Kaggle notebook 中提取可复用的机器学习技巧"""

from .notebook_parser import (
    parse_notebook,
    extract_code_comments,
    notebook_to_text,
    get_kernel_competition,
)
from .skill_extractor import (
    extract_skills_from_notebook,
    extract_skills_from_notebooks_batch,
)
from .skill_library import (
    save_skill,
    match_keyword,
    search_skills,
    build_skill_context,
    list_keywords,
    get_library_stats,
)

__all__ = [
    # 解析
    "parse_notebook",
    "extract_code_comments",
    "notebook_to_text",
    "get_kernel_competition",
    # 提取
    "extract_skills_from_notebook",
    "extract_skills_from_notebooks_batch",
    # 库管理
    "save_skill",
    "match_keyword",
    "search_skills",
    "build_skill_context",
    "list_keywords",
    "get_library_stats",
]
