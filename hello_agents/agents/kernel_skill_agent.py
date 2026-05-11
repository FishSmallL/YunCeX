"""Kernel Skill Agent - 从 Kaggle kernel notebooks 中提取可复用技巧。

作为子 agent 运行，输入 keyword，自动：
1. 匹配本地 Skill 库
2. 查找已下载的 kernel 目录
3. 若都没有 → 自动调用 Kaggle 工具下载 kernel
4. 解析 .ipynb notebook 内容
5. 用 LLM 提取结构化技巧
6. 保存到 skill_library/<keyword>/
"""

import json
import re
from pathlib import Path
from typing import Optional, List, Dict

from ..core.agent import Agent
from ..core.llm import HelloAgentsLLM
from ..core.config import Config


class KernelSkillAgent(Agent):
    """处理 Kaggle kernel 并提取可复用技巧的子 agent。

    输入格式 (input_text, JSON 或纯文本):
        JSON: {"keyword": "machine_learning",
               "kernel_dirs": ["path/to/kernel1", ...]}
        纯文本: "machine learning"
               (将自动在 kaggle_knowledge/output/ 下查找)

    返回:
        提取结果摘要 (str)
    """

    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        tool_registry: Optional["ToolRegistry"] = None,
        skill_library_dir: Optional[str] = None,
        kernels_base_dir: Optional[str] = None,
    ):
        super().__init__(
            name=name,
            llm=llm,
            system_prompt=system_prompt,
            config=config,
            tool_registry=tool_registry,
        )
        # Default paths relative to project
        project_root = Path(__file__).parent.parent.parent
        kaggle_dir = project_root / "kaggle_knowledge"

        self.skill_library_dir = skill_library_dir or str(
            kaggle_dir / "skill_library"
        )
        self.kernels_base_dir = kernels_base_dir or str(
            kaggle_dir / "output"
        )

    def run(self, input_text: str, **kwargs) -> str:
        """运行 kernel skill 提取流程。

        Args:
            input_text: JSON 或纯文本，包含 keyword 和可选的 kernel_dirs

        Returns:
            提取结果摘要
        """
        from kaggle_knowledge.kernel_processor import (
            extract_skills_from_notebooks_batch,
            save_skill,
            match_keyword,
            list_keywords,
            get_library_stats,
        )

        # Parse input
        keyword, kernel_dirs = self._parse_input(input_text)

        print(f"\n{'='*50}")
        print(f"KernelSkillAgent: 开始处理关键词 '{keyword}'")
        print(f"Skill 库目录: {self.skill_library_dir}")
        print(f"Kernel 目录数: {len(kernel_dirs)}")
        print(f"{'='*50}")

        if not kernel_dirs:
            # Check if we already have skills for this keyword
            matched = match_keyword(keyword, self.skill_library_dir)
            if matched:
                # Build and return existing skill context
                from kaggle_knowledge.kernel_processor.skill_library import (
                    search_skills, build_skill_context,
                )
                top_k = self._get_top_skills()
                ctx = build_skill_context(matched, self.skill_library_dir, top_k=top_k)
                return (
                    f"关键词 '{keyword}' 已匹配到已有 skill 库目录 '{matched}'。\n\n"
                    f"{ctx}"
                )

            # No local kernels, no matching skills → auto-download
            print(f"\n未找到关键词 '{keyword}' 的本地 kernel 或 skill，自动触发 Kaggle 下载...")
            self._download_kernels_for_keyword(keyword)
            kernel_dirs = self._find_kernel_dirs(keyword)

            if not kernel_dirs:
                return (
                    f"已尝试从 Kaggle 下载关键词 '{keyword}' 的 kernel，但仍未找到。"
                    f"请检查关键词是否正确，或手动运行 kaggle_knowledge/main.py。"
                )

        # Process kernels and extract skills
        all_skills = extract_skills_from_notebooks_batch(
            kernel_dirs, keyword, self.llm
        )

        if not all_skills:
            return f"未能从 {len(kernel_dirs)} 个 kernel 中提取到任何技巧。"

        # Save skills to library
        saved_paths = []
        for skill in all_skills:
            path = save_skill(keyword, skill, self.skill_library_dir)
            saved_paths.append(path)

        # Build summary
        categories = {}
        for s in all_skills:
            cat = s.get("category", "未分类")
            categories[cat] = categories.get(cat, 0) + 1

        summary_lines = [
            f"\n✅ 技能提取完成",
            f"关键词: {keyword}",
            f"处理 kernel 数: {len(kernel_dirs)}",
            f"提取技巧总数: {len(all_skills)}",
            f"保存位置: {self.skill_library_dir}/{self._sanitize_keyword(keyword)}/",
            f"\n分类统计:",
        ]
        for cat, count in sorted(categories.items()):
            summary_lines.append(f"  - {cat}: {count} 条")
        summary_lines.append(f"\n已保存文件:")
        for p in saved_paths[:10]:
            summary_lines.append(f"  - {Path(p).name}")
        if len(saved_paths) > 10:
            summary_lines.append(f"  ... 共 {len(saved_paths)} 个文件")

        # Library-wide stats
        stats = get_library_stats(self.skill_library_dir)
        summary_lines.append(
            f"\nSkill 库总览: {stats['total_skills']} 条技巧, "
            f"{stats['keywords']} 个关键词目录"
        )

        return "\n".join(summary_lines)

    def _download_kernels_for_keyword(self, keyword: str):
        """调用 Kaggle 工具下载指定关键词的竞赛 kernel。

        复用 kaggle_knowledge 的完整流水线：
        搜索竞赛 → 获取排行榜 → 下载 kernel
        """
        import sys
        project_root = Path(__file__).parent.parent.parent
        kg_dir = project_root / "kaggle_knowledge"
        config_path = kg_dir / "config.json"

        # kaggle_knowledge 内部模块使用 "from utils import ..." 的相对导入，
        # 需要将 kaggle_knowledge/ 加入 sys.path
        kg_path = str(kg_dir)
        if kg_path not in sys.path:
            sys.path.insert(0, kg_path)

        from kaggle_knowledge.utils import load_config, extract_competition_slug
        from kaggle_knowledge.search_competitions import batch_search_competitions
        from kaggle_knowledge.get_leaderboard import get_leaderboard, extract_usernames
        from kaggle_knowledge.download_kernels import download_competition_kernels

        config = load_config(str(config_path))
        output_dir = config.get("output_dir", "output")
        output_dir = str(kg_dir / output_dir)  # relative → absolute
        competitions_per_keyword = config.get("competitions_per_keyword", 5)
        top_users = config.get("top_leaderboard_users", 5)
        min_score = config.get("min_leaderboard_score")
        kernels_per_user = config.get("kernels_per_user", 5)
        min_team_count = config.get("min_team_count", 0)
        save_csv = config.get("save_csv", {})

        print(f"\n  [下载] 搜索关键词 '{keyword}' 的竞赛...")
        competitions_dict = batch_search_competitions(
            [keyword],
            max_competitions=competitions_per_keyword,
            save_csv_flag=save_csv.get("competitions", True),
            output_dir=output_dir,
            min_team_count=min_team_count,
        )

        competitions = competitions_dict.get(keyword, [])
        if not competitions:
            print(f"  [下载] 未找到与 '{keyword}' 相关的竞赛")
            return

        print(f"  [下载] 找到 {len(competitions)} 个竞赛")

        for idx, comp in enumerate(competitions, 1):
            ref = comp.get("ref", "")
            slug = extract_competition_slug(ref)
            name = comp.get("title", slug)

            print(f"  [下载] [{idx}/{len(competitions)}] {name} ({slug})")

            leaderboard = get_leaderboard(
                slug,
                top_users=top_users,
                min_score=min_score,
                save_csv_flag=save_csv.get("leaderboard", True),
                output_dir=output_dir,
            )
            if not leaderboard:
                continue

            usernames = extract_usernames(leaderboard)
            print(f"         排行榜前 {len(usernames)} 名用户: {usernames[:3]}...")

            download_competition_kernels(
                slug, name, usernames,
                keyword=keyword,
                max_kernels_per_user=kernels_per_user,
                base_output_dir=output_dir,
                save_csv_flag=save_csv.get("kernels_list", True),
            )

        print(f"  [下载] 关键词 '{keyword}' 下载完成\n")

    def _parse_input(self, input_text: str):
        """Parse input_text into keyword and kernel_dirs list.

        Supports:
        - JSON: {"keyword": "...", "kernel_dirs": [...]}
        - Plain text: treated as keyword, auto-discover kernel dirs
        """
        input_text = input_text.strip()

        # Try JSON
        try:
            data = json.loads(input_text)
            if isinstance(data, dict):
                keyword = data.get("keyword", "")
                kernel_dirs = data.get("kernel_dirs", [])
                if keyword and kernel_dirs:
                    return keyword, kernel_dirs
        except (json.JSONDecodeError, TypeError):
            pass

        # Plain text: treat as keyword
        keyword = input_text
        kernel_dirs = self._find_kernel_dirs(keyword)
        return keyword, kernel_dirs

    def _find_kernel_dirs(self, keyword: str) -> List[str]:
        """Auto-discover kernel directories under the output base dir for a keyword.

        Searches under kaggle_knowledge/output/<keyword>/kernels/
        Tries: exact keyword → sanitized version → substring match
        """
        kw_safe = self._sanitize_keyword(keyword)
        base = Path(self.kernels_base_dir)

        def _subdirs(path):
            if path.is_dir():
                return [str(d) for d in path.iterdir() if d.is_dir()]
            return []

        # Try exact keyword as-is (handles "machine learning" with space)
        result = _subdirs(base / keyword / "kernels")
        if result:
            return result

        # Try sanitized version ("machine_learning")
        result = _subdirs(base / kw_safe / "kernels")
        if result:
            return result

        # Try to find partial match
        if base.is_dir():
            for d in base.iterdir():
                if d.is_dir() and (keyword in d.name or kw_safe in d.name.lower()):
                    result = _subdirs(d / "kernels")
                    if result:
                        return result

        return []

    def _get_top_skills(self) -> int:
        """从 config.json 读取 top_skills 配置，默认 5"""
        try:
            from kaggle_knowledge.utils import load_config
            kg_dir = Path(__file__).parent.parent.parent / "kaggle_knowledge"
            config = load_config(str(kg_dir / "config.json"))
            return int(config.get("top_skills", 5))
        except Exception:
            return 5

    @staticmethod
    def _sanitize_keyword(keyword: str) -> str:
        name = keyword.lower().strip()
        name = re.sub(r'[^a-z0-9_\s]', '', name)
        name = re.sub(r'\s+', '_', name)
        return name[:80]
