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
from typing import Optional, List, Dict, AsyncGenerator

from ..core.agent import Agent
from ..core.llm import HelloAgentsLLM
from ..core.config import Config
from ..core.streaming import StreamEvent, StreamEventType


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
        from kaggle_knowledge.kernel_processor import save_skill

        # Parse input (supports comma-separated multi-keyword)
        kw_pairs = self._parse_input(input_text)
        print(f"\n{'='*50}")
        print(f"KernelSkillAgent: 共 {len(kw_pairs)} 个关键词")
        print(f"Skill 库目录: {self.skill_library_dir}\n{'='*50}")

        all_skills = []
        cached_contexts: list[str] = []
        for kw_idx, (keyword, kernel_dirs) in enumerate(kw_pairs, 1):
            if len(kw_pairs) > 1:
                print(f"\n── [{kw_idx}/{len(kw_pairs)}]: {keyword} (kernels: {len(kernel_dirs)}) ──")
            skills, ctx = self._run_one_keyword(keyword, kernel_dirs)
            if ctx:
                cached_contexts.append(ctx)
            all_skills.extend(skills)

        # 去重（精确 + 语义）
        if len(all_skills) > 1:
            all_skills = dedup_skills(all_skills, llm=self.llm)

        if not all_skills:
            if cached_contexts:
                return "\n\n".join(cached_contexts)
            return f"未能从 {len(kernel_dirs)} 个 kernel 中提取到任何技巧。"

        # ── 按关键词分组重新保存（覆盖即时保存的预去重版）──
        skills_by_kw: dict[str, list] = {}
        for s in all_skills:
            kw = s.get("keyword", kw_pairs[0][0])
            skills_by_kw.setdefault(kw, []).append(s)
        for kw, kw_skills in skills_by_kw.items():
            kw_dir = Path(self.skill_library_dir) / self._sanitize_keyword(kw)
            for f in kw_dir.glob("skill_*.md"):
                try: f.unlink()
                except Exception: pass
            for skill in kw_skills:
                save_skill(kw, skill, self.skill_library_dir)

        # ── 只返回更新后的 Skill 库紧凑摘要 ──
        all_contexts = list(cached_contexts)
        seen_kw = set()
        for kw, _ in kw_pairs:
            if kw not in seen_kw:
                ctx = self._build_compact_skill_summary(kw)
                if ctx:
                    all_contexts.append(ctx)
                    seen_kw.add(kw)
        return "\n\n".join(all_contexts) if all_contexts else "未能从 kernel 中提取到任何技巧。"

    async def arun_stream(
        self, input_text: str, **kwargs
    ) -> AsyncGenerator[StreamEvent, None]:
        """流式执行 kernel skill 提取流程，实时 yield 进度事件。

        与 run() 功能相同，但通过 StreamEvent 实时报告进度，
        提升子 Agent 调用时的用户体验。

        Args:
            input_text: JSON 或纯文本，包含 keyword 和可选的 kernel_dirs

        Yields:
            StreamEvent: 进度事件
        """
        total_steps = 0

        # ── 阶段 1: 启动 ──
        yield StreamEvent.create(
            StreamEventType.AGENT_START,
            self.name,
            input_text=input_text,
        )

        # ── 阶段 2: 解析输入 ──
        kw_pairs = self._parse_input(input_text)
        yield self._chunk(
            f"KernelSkillAgent 共 {len(kw_pairs)} 个关键词: "
            + ", ".join(kw for kw, _ in kw_pairs) + "\n"
            f"  Skill 库: {self.skill_library_dir}\n"
        )

        all_skills = []
        cached_contexts: list[str] = []  # 收集所有 skip 关键词的库内 context
        for kw_idx, (keyword, kernel_dirs) in enumerate(kw_pairs, 1):
            async for event in self._process_one_keyword_stream(
                keyword, kernel_dirs, kw_idx, len(kw_pairs)
            ):
                if event.type == StreamEventType.AGENT_FINISH:
                    if event.data.get("skip") and len(kw_pairs) == 1:
                        # single keyword with enough skills → 流式只输出摘要，完整上下文放 result
                        if matched_context := event.data.get("context"):
                            yield self._chunk(f"从 Skill 库返回已有技巧\n")
                            yield StreamEvent.create(
                                StreamEventType.AGENT_FINISH, self.name,
                                result=matched_context, total_steps=total_steps, max_steps_reached=False,
                            )
                        return
                    if event.data.get("skip"):
                        if ctx := event.data.get("context"):
                            cached_contexts.append(ctx)
                        continue
                    all_skills.extend(event.data.get("skills", []))
                    total_steps = event.data.get("total_steps", total_steps)
                else:
                    yield event

        # ── 阶段 6: 去重 + 保存 ──
        if not all_skills:
            if cached_contexts:
                result = "\n\n".join(cached_contexts)
                # 流式只输出简短摘要，完整上下文由 AGENT_FINISH.result 传递给工具结果
                kw_list = ", ".join(kw for kw, _ in kw_pairs)
                yield self._chunk(f"从 Skill 库返回 {len(cached_contexts)} 个关键词({kw_list})的已有技巧\n")
                yield StreamEvent.create(
                    StreamEventType.AGENT_FINISH, self.name,
                    result=result, total_steps=total_steps, max_steps_reached=False,
                )
            else:
                result = "未能从 kernel 中提取到任何技巧。"
                yield StreamEvent.create(StreamEventType.AGENT_FINISH, self.name, result=result, total_steps=total_steps)
            return

        # 去重合并
        if len(all_skills) > 1:
            from kaggle_knowledge.kernel_processor.skill_extractor import dedup_skills as _dedup
            before_dedup = len(all_skills)
            all_skills = _dedup(all_skills, llm=self.llm)
            if before_dedup > len(all_skills):
                yield self._chunk(f"去重合并: {before_dedup} -> {len(all_skills)} 条\n")

        # ── 按关键词分组重新保存（覆盖即时保存的预去重版）──
        from kaggle_knowledge.kernel_processor import save_skill as _sv
        skills_by_kw: dict[str, list] = {}
        for s in all_skills:
            kw = s.get("keyword", kw_pairs[0][0])
            skills_by_kw.setdefault(kw, []).append(s)
        total_steps += 1
        yield StreamEvent.create(StreamEventType.STEP_START, self.name, step=total_steps, max_steps=str(len(all_skills)))
        for kw, kw_skills in skills_by_kw.items():
            kw_dir = Path(self.skill_library_dir) / self._sanitize_keyword(kw)
            for f in kw_dir.glob("skill_*.md"):
                try: f.unlink()
                except Exception: pass
            for skill in kw_skills:
                _sv(kw, skill, self.skill_library_dir)
        yield self._chunk(f"已保存 {len(all_skills)} 条技能到 {len(skills_by_kw)} 个关键词目录\n")
        yield StreamEvent.create(StreamEventType.STEP_FINISH, self.name, step=total_steps)

        # ── 只返回更新后的 Skill 库紧凑摘要（不输出统计总结）──
        all_contexts = list(cached_contexts)
        seen_kw = set()
        for kw, _ in kw_pairs:
            if kw not in seen_kw:
                ctx = self._build_compact_skill_summary(kw)
                if ctx:
                    all_contexts.append(ctx)
                    seen_kw.add(kw)
        result = "\n\n".join(all_contexts) if all_contexts else "未能从 kernel 中提取到任何技巧。"

        yield StreamEvent.create(StreamEventType.AGENT_FINISH, self.name, result=result, total_steps=total_steps, max_steps_reached=False)
        return

        # DEAD CODE BELOW (unreachable — removed by return above)
        total_steps += 1
        max_hint = self._get_top_skills()
        yield StreamEvent.create(
            StreamEventType.STEP_START,
            self.name,
            step=total_steps,
            max_steps=f"{max_hint}+",
        )
        yield self._chunk(f"正在匹配 Skill 库 (top_k={max_hint})...\n")

        matched = match_keyword(keyword, self.skill_library_dir)
        if matched:
            from kaggle_knowledge.kernel_processor.skill_library import (
                search_skills, build_skill_context,
            )
            min_skills = self._get_min_skills()
            skill_count = len(search_skills(
                matched, self.skill_library_dir, top_k=999
            ))
            if skill_count >= min_skills:
                ctx = build_skill_context(
                    matched, self.skill_library_dir, top_k=max_hint
                )
                yield self._chunk(
                    f"已匹配到目录 '{matched}' ({skill_count} 条已有技能，满足阈值 {min_skills})\n"
                )
                yield StreamEvent.create(
                    StreamEventType.STEP_FINISH,
                    self.name,
                    step=total_steps,
                )
                result = (
                    f"关键词 '{keyword}' 已匹配到已有 skill 库目录 '{matched}'。\n\n"
                    f"{ctx}"
                )
                yield StreamEvent.create(
                    StreamEventType.AGENT_FINISH,
                    self.name,
                    result=result,
                    total_steps=total_steps,
                    max_steps_reached=False,
                )
                return
            else:
                yield self._chunk(
                    f"已匹配到目录 '{matched}' 但仅有 {skill_count} 条技能"
                    f"（阈值 {min_skills}），将继续提取补充...\n"
                )
                # 不足 → 继续走提取路径（下面）
        else:
            yield self._chunk("未匹配到已有 skill\n")

        yield StreamEvent.create(
            StreamEventType.STEP_FINISH,
            self.name,
            step=total_steps,
        )

        # ── 阶段 4: 本地 kernel 不足时自动下载补充 ──
        min_kernels = self._get_min_kernels()
        if len(kernel_dirs) < min_kernels:
            if kernel_dirs:
                yield self._chunk(
                    f"本地 kernel 数 {len(kernel_dirs)} < {min_kernels}，将下载补充...\n"
                )
            total_steps += 1
            yield StreamEvent.create(
                StreamEventType.STEP_START,
                self.name,
                step=total_steps,
                max_steps="...",
            )

            kernel_dirs_found = []
            async for event in self._download_kernels_stream(keyword):
                if event.type == StreamEventType.LLM_CHUNK:
                    yield event
                elif event.type == StreamEventType.AGENT_FINISH:
                    kernel_dirs_found = event.data.get("kernel_dirs", [])
                    yield self._chunk(
                        f"下载完成，找到 {len(kernel_dirs_found)} 个 kernel 目录\n"
                    )

            kernel_dirs = kernel_dirs_found or self._find_kernel_dirs(keyword)

            yield StreamEvent.create(
                StreamEventType.STEP_FINISH,
                self.name,
                step=total_steps,
            )

            if not kernel_dirs:
                result = (
                    f"已尝试从 Kaggle 下载关键词 '{keyword}' 的 kernel，但仍未找到。"
                    f"请检查关键词是否正确，或手动运行 kaggle_knowledge/main.py。"
                )
                yield StreamEvent.create(
                    StreamEventType.AGENT_FINISH,
                    self.name,
                    result=result,
                    total_steps=total_steps,
                    max_steps_reached=False,
                )
                return

        # ── 阶段 5: 提取技能（逐个 kernel，带进度） ──
        total_steps += 1
        yield StreamEvent.create(
            StreamEventType.STEP_START,
            self.name,
            step=total_steps,
            max_steps=str(len(kernel_dirs)),
        )

        from kaggle_knowledge.kernel_processor import (
            save_skill,
            get_library_stats,
        )
        from kaggle_knowledge.kernel_processor.notebook_parser import (
            parse_notebook,
            notebook_to_text,
            get_kernel_competition,
        )
        from kaggle_knowledge.kernel_processor.skill_extractor import (
            SCAN_PROMPT,
            DEEP_EXTRACT_PROMPT,
            EXTRACTION_PROMPT,
            _parse_llm_response,
            _extract_json_array,
            dedup_skills,
        )

        all_skills = []
        for i, kdir in enumerate(kernel_dirs, 1):
            kpath = Path(kdir)
            if not kpath.is_dir():
                continue

            ipynb_files = list(kpath.glob("*.ipynb"))
            if not ipynb_files:
                yield self._chunk(f"  [{i}/{len(kernel_dirs)}] {kpath.name} — 无 .ipynb，跳过\n")
                continue

            meta_files = list(kpath.glob("kernel-metadata.json"))
            competition = (
                get_kernel_competition(str(meta_files[0]))
                if meta_files else "unknown"
            )

            yield self._chunk(
                f"  [{i}/{len(kernel_dirs)}] {kpath.name} (竞赛: {competition}) 解析中... [{keyword}]\n"
            )

            try:
                cells = parse_notebook(str(ipynb_files[0]))
                text = notebook_to_text(cells, max_len=30000)
            except Exception as e:
                yield self._chunk(f"    解析 notebook 失败: {e}\n")
                continue

            # ── 两阶段提取 ──
            # Phase 1: 亮点扫描（流式调用，实时显示 LLM 输出）
            scan_prompt = SCAN_PROMPT.replace("{notebook_text}", text)
            yield self._chunk("    Phase1: 正在扫描 notebook 亮点...\n")
            highlights = []
            full_scan = ""
            try:
                async for chunk in self.llm.astream_invoke(
                    messages=[{"role": "user", "content": scan_prompt}],
                    max_tokens=2048,
                ):
                    full_scan += chunk
                    yield self._chunk(chunk, chunk_type="thinking")
                yield self._chunk("\n")
                items = _extract_json_array(full_scan.strip()) or []
                highlights = [
                    h for h in items
                    if isinstance(h, dict) and h.get("highlight")
                ]
                if highlights:
                    yield self._chunk(
                        f"    Phase1: 识别到 {len(highlights)} 个亮点\n"
                    )
            except Exception:
                pass

            # Phase 2: 深度提取（流式 LLM 调用）
            if highlights:
                hl_text = "\n".join(
                    f"- [{h.get('category','?')}] {h.get('highlight','')}"
                    for h in highlights
                )
                deep_prompt = DEEP_EXTRACT_PROMPT.replace("{highlights}", hl_text)
                deep_prompt = deep_prompt.replace("{notebook_text}", text)
            else:
                yield self._chunk(f"    未识别到亮点，回退到单 pass 提取\n")
                deep_prompt = EXTRACTION_PROMPT.replace("{notebook_text}", text)

            yield self._chunk("    Phase2: 正在深度提取技巧...\n")
            full_response = ""
            try:
                async for chunk in self.llm.astream_invoke(
                    messages=[{"role": "user", "content": deep_prompt}],
                    max_tokens=8192,
                ):
                    full_response += chunk
                    yield self._chunk(chunk, chunk_type="thinking")
                yield self._chunk("\n")
            except Exception as e:
                yield self._chunk(f"\n    LLM 调用失败: {e}\n")
                continue

            # 解析
            skills = _parse_llm_response(
                full_response, kpath.name, competition, keyword
            )
            if skills:
                yield self._chunk(
                    f"    提取到 {len(skills)} 条技巧: "
                    + ", ".join(s.get("category", "?") for s in skills)
                    + "\n"
                )
            else:
                yield self._chunk(f"    未提取到技巧\n")
            all_skills.extend(skills)

        yield StreamEvent.create(
            StreamEventType.STEP_FINISH,
            self.name,
            step=total_steps,
        )

        # 去重合并（同 category + 同 name 视为同一技巧，保留最佳版本）
        if len(all_skills) > 1:
            before_dedup = len(all_skills)
            all_skills = dedup_skills(all_skills, llm=self.llm)
            removed = before_dedup - len(all_skills)
            if removed > 0:
                yield self._chunk(f"去重合并: {before_dedup} → {len(all_skills)} 条 (-{removed})\n")

        if not all_skills:
            result = f"未能从 {len(kernel_dirs)} 个 kernel 中提取到任何技巧。"
            yield StreamEvent.create(
                StreamEventType.AGENT_FINISH,
                self.name,
                result=result,
                total_steps=total_steps,
                max_steps_reached=False,
            )
            return

        # ── 阶段 6: 保存技能（按竞赛 slug 分组）──
        # 清理涉及到的竞赛目录的旧 skill 文件
        comp_slugs = {s.get("source_competition", keyword) for s in all_skills}
        for slug in comp_slugs:
            comp_dir = Path(self.skill_library_dir) / self._sanitize_keyword(slug)
            if comp_dir.is_dir():
                old_files = list(comp_dir.glob("skill_*.md"))
                for f in old_files:
                    try:
                        f.unlink()
                    except Exception:
                        pass
                if old_files:
                    yield self._chunk(
                        f"  已清理 [{slug}] {len(old_files)} 个旧 skill 文件\n"
                    )

        total_steps += 1
        yield StreamEvent.create(
            StreamEventType.STEP_START,
            self.name,
            step=total_steps,
            max_steps=str(len(all_skills)),
        )

        saved_paths = []
        for i, skill in enumerate(all_skills, 1):
            comp_slug = skill.get("source_competition", keyword)
            path = save_skill(keyword, skill, self.skill_library_dir)
            saved_paths.append(path)
            if i % 5 == 0 or i == len(all_skills):
                yield self._chunk(f"  已保存 {i}/{len(all_skills)} 条技能\n")

        yield StreamEvent.create(
            StreamEventType.STEP_FINISH,
            self.name,
            step=total_steps,
        )

        # ── 阶段 7: 构建摘要，完成 ──
        categories = {}
        for s in all_skills:
            cat = s.get("category", "未分类")
            categories[cat] = categories.get(cat, 0) + 1

        comp_dirs = ", ".join(sorted(comp_slugs))
        summary_lines = [
            f"\n技能提取完成",
            f"关键词: {', '.join(kw for kw, _ in kw_pairs)}",
            f"处理 kernel 数: {sum(len(kd) for _, kd in kw_pairs)}",
            f"提取技巧总数: {len(all_skills)}",
            f"保存位置: {self.skill_library_dir}/{self._sanitize_keyword(kw_pairs[0][0])}/",
            f"\n分类统计:",
        ]
        for cat, count in sorted(categories.items()):
            summary_lines.append(f"  - {cat}: {count} 条")
        summary_lines.append(f"\n已保存文件:")
        for p in saved_paths[:10]:
            summary_lines.append(f"  - {Path(p).name}")
        if len(saved_paths) > 10:
            summary_lines.append(f"  ... 共 {len(saved_paths)} 个文件")

        stats = get_library_stats(self.skill_library_dir)
        summary_lines.append(
            f"\nSkill 库总览: {stats['total_skills']} 条技巧, "
            f"{stats['keywords']} 个关键词目录"
        )

        result = "\n".join(summary_lines)
        yield self._chunk(result + "\n")
        yield StreamEvent.create(
            StreamEventType.AGENT_FINISH,
            self.name,
            result=result,
            total_steps=total_steps,
            max_steps_reached=False,
        )

    async def _download_kernels_stream(
        self, keyword: str
    ) -> AsyncGenerator[StreamEvent, None]:
        """流式版 Kaggle kernel 下载，在每个子步骤 yield 进度事件。

        复用 _download_kernels_for_keyword 的完整流水线：
        搜索竞赛 → 获取排行榜 → 下载 kernel
        """
        import sys
        project_root = Path(__file__).parent.parent.parent
        kg_dir = project_root / "kaggle_knowledge"
        config_path = kg_dir / "config.json"

        kg_path = str(kg_dir)
        if kg_path not in sys.path:
            sys.path.insert(0, kg_path)

        from kaggle_knowledge.utils import load_config, extract_competition_slug
        from kaggle_knowledge.search_competitions import search_competitions_paginated
        from kaggle_knowledge.get_leaderboard import get_leaderboard, extract_usernames
        from kaggle_knowledge.download_kernels import download_competition_kernels

        config = load_config(str(config_path))
        output_dir = config.get("output_dir", "output")
        output_dir = str(kg_dir / output_dir)
        competitions_per_keyword = config.get("competitions_per_keyword", 5)
        top_users = config.get("top_leaderboard_users", 5)
        min_score = config.get("min_leaderboard_score")
        kernels_per_user = config.get("kernels_per_user", 5)
        min_team_count = config.get("min_team_count", 0)
        search_max_pages = config.get("search_max_pages", 5)
        save_csv = config.get("save_csv", {})

        yield self._chunk(f"搜索关键词 '{keyword}' 的 Kaggle 竞赛...\n")
        competitions = search_competitions_paginated(
            keyword,
            max_competitions=competitions_per_keyword,
            save_csv_flag=save_csv.get("competitions", True),
            output_dir=output_dir,
            min_team_count=min_team_count,
            max_pages=search_max_pages,
        )
        if not competitions:
            yield self._chunk(f"未找到与 '{keyword}' 相关的竞赛\n")
            yield StreamEvent.create(
                StreamEventType.AGENT_FINISH,
                self.name,
                kernel_dirs=[],
            )
            return

        yield self._chunk(f"找到 {len(competitions)} 个竞赛\n")

        for idx, comp in enumerate(competitions, 1):
            ref = comp.get("ref", "")
            slug = extract_competition_slug(ref)
            name = comp.get("title", slug)

            yield self._chunk(
                f"  [{idx}/{len(competitions)}] {name} ({slug}) — 获取排行榜...\n"
            )

            leaderboard = get_leaderboard(
                slug,
                top_users=top_users,
                min_score=min_score,
                save_csv_flag=save_csv.get("leaderboard", True),
                output_dir=output_dir,
            )
            if not leaderboard:
                yield self._chunk(f"    排行榜为空，跳过\n")
                continue

            usernames = extract_usernames(leaderboard)
            user_ranks = {e.get("teamName",""): i for i, e in enumerate(leaderboard, 1)}
            yield self._chunk(
                f"    排行榜前 {len(usernames)} 名: {', '.join(usernames[:3])}... — 下载 kernel\n"
            )

            download_competition_kernels(
                slug, name, usernames,
                keyword=keyword,
                max_kernels_per_user=kernels_per_user,
                base_output_dir=output_dir,
                save_csv_flag=save_csv.get("kernels_list", True),
                user_ranks=user_ranks,
            )

        yield self._chunk(f"关键词 '{keyword}' 全部下载完成\n")

        # Find kernel dirs after download
        kernel_dirs = self._find_kernel_dirs(keyword)
        yield StreamEvent.create(
            StreamEventType.AGENT_FINISH,
            self.name,
            kernel_dirs=kernel_dirs,
        )

    async def _process_one_keyword_stream(
        self, keyword: str, kernel_dirs: list, kw_idx: int, total_kw: int
    ) -> AsyncGenerator[StreamEvent, None]:
        """单关键词异步流式 pipeline: match→download→extract，yield 事件。
        AGENT_FINISH 事件 data 包含 'skills' 列表和可选 'skip' 标记。
        """
        from kaggle_knowledge.kernel_processor import match_keyword
        matched = match_keyword(keyword, self.skill_library_dir)
        if matched:
            from kaggle_knowledge.kernel_processor.skill_library import search_skills
            skill_count = len(search_skills(matched, self.skill_library_dir, top_k=999))
            min_skills = self._get_min_skills()
            if skill_count >= min_skills:
                ctx = self._build_compact_skill_summary(keyword)
                yield self._chunk(f"已匹配到 '{matched}' ({skill_count} 条)\n")
                yield StreamEvent.create(StreamEventType.AGENT_FINISH, self.name, skills=[], skip=True, context=ctx)
                return
            else:
                yield self._chunk(f"匹配到 '{matched}' 仅 {skill_count} 条（阈值 {min_skills}），继续...\n")
        else:
            yield self._chunk("未匹配到已有 skill\n")

        min_kernels = self._get_min_kernels()
        if len(kernel_dirs) < min_kernels:
            yield self._chunk(f"本地 kernel {len(kernel_dirs)} < {min_kernels}，下载补充...\n")
            kernel_dirs_found = []
            async for event in self._download_kernels_stream(keyword):
                if event.type == StreamEventType.LLM_CHUNK: yield event
                elif event.type == StreamEventType.AGENT_FINISH: kernel_dirs_found = event.data.get("kernel_dirs", [])
            kernel_dirs = kernel_dirs_found or self._find_kernel_dirs(keyword)
            if not kernel_dirs:
                yield self._chunk("下载后仍未找到 kernel\n")
                yield StreamEvent.create(StreamEventType.AGENT_FINISH, self.name, skills=[])
                return

        from kaggle_knowledge.kernel_processor.notebook_parser import parse_notebook, notebook_to_text, get_kernel_competition
        from kaggle_knowledge.kernel_processor.skill_extractor import SCAN_PROMPT, DEEP_EXTRACT_PROMPT, _parse_llm_response, _extract_json_array
        skills = []
        for i, kdir in enumerate(kernel_dirs, 1):
            kpath = Path(kdir)
            if not kpath.is_dir(): continue
            ipynb_files = list(kpath.glob("*.ipynb"))
            if not ipynb_files:
                yield self._chunk(f"  [{i}/{len(kernel_dirs)}] {kpath.name} — 无 .ipynb\n"); continue
            meta_files = list(kpath.glob("kernel-metadata.json"))
            competition = get_kernel_competition(str(meta_files[0])) if meta_files else "unknown"
            yield self._chunk(f"  [{i}/{len(kernel_dirs)}] {kpath.name} ({competition}) 解析中... [{keyword}]\n")
            try:
                cells = parse_notebook(str(ipynb_files[0]))
                text = notebook_to_text(cells, max_len=30000)
            except Exception as e:
                yield self._chunk(f"    解析失败: {e}\n"); continue
            scan_prompt = SCAN_PROMPT.replace("{notebook_text}", text)
            yield self._chunk("    Phase1: 正在扫描 notebook 亮点...\n")
            highlights, full_scan = [], ""
            try:
                async for chunk in self.llm.astream_invoke(messages=[{"role":"user","content":scan_prompt}], max_tokens=2048):
                    full_scan += chunk; yield self._chunk(chunk, chunk_type="thinking")
                yield self._chunk("\n")
                items = _extract_json_array(full_scan.strip()) or []
                highlights = [h for h in items if isinstance(h,dict) and h.get("highlight")]
                if highlights: yield self._chunk(f"    Phase1: 识别到 {len(highlights)} 个亮点\n")
            except Exception: pass
            if highlights:
                hl_text = "\n".join(f"- [{h.get('category','?')}] {h.get('highlight','')}" for h in highlights)
                deep_prompt = DEEP_EXTRACT_PROMPT.replace("{highlights}",hl_text).replace("{notebook_text}",text)
                yield self._chunk("    Phase2: 正在深度提取技巧...\n")
                full_response = ""
                try:
                    async for chunk in self.llm.astream_invoke(messages=[{"role":"user","content":deep_prompt}], max_tokens=8192):
                        full_response += chunk; yield self._chunk(chunk, chunk_type="thinking")
                    yield self._chunk("\n")
                except Exception as e:
                    yield self._chunk(f"    LLM 调用失败: {e}\n"); continue
                parsed = _parse_llm_response(full_response, kpath.name, competition, keyword)
            else:
                yield self._chunk("    未发现技术亮点，跳过\n")
                parsed = []
            if parsed:
                yield self._chunk(f"    提取到 {len(parsed)} 条: "+", ".join(s.get("category","?") for s in parsed)+"\n")
                # 立即保存，避免中途失败全部丢失
                from kaggle_knowledge.kernel_processor import save_skill as _sv2
                for sk in parsed:
                    _sv2(keyword, sk, self.skill_library_dir)
            else: yield self._chunk("    未提取到技巧\n")
            skills.extend(parsed)
        yield StreamEvent.create(StreamEventType.AGENT_FINISH, self.name, skills=skills)

    def _run_one_keyword(self, keyword: str, kernel_dirs: list) -> tuple:
        """Per-keyword sync pipeline: match->download->extract.
        Returns (skills_list, context_or_none).
        When library has enough skills, returns ([], context_string).
        """
        from kaggle_knowledge.kernel_processor import match_keyword
        matched = match_keyword(keyword, self.skill_library_dir)
        if matched:
            from kaggle_knowledge.kernel_processor.skill_library import search_skills
            skill_count = len(search_skills(matched, self.skill_library_dir, top_k=999))
            if skill_count >= self._get_min_skills():
                ctx = self._build_compact_skill_summary(keyword)
                print(f"  已匹配到 '{matched}' ({skill_count} 条)，跳过")
                return [], ctx
            else:
                print(f"  匹配到 '{matched}' 仅 {skill_count} 条（阈值 {self._get_min_skills()}），继续...")
        else:
            print(f"  未匹配到已有 skill")
        min_kernels = self._get_min_kernels()
        if len(kernel_dirs) < min_kernels:
            print(f"  本地 kernel {len(kernel_dirs)} < {min_kernels}，下载补充...")
            self._download_kernels_for_keyword(keyword)
            kernel_dirs = self._find_kernel_dirs(keyword)
            if not kernel_dirs:
                print(f"  下载后仍未找到 kernel"); return [], None
        from kaggle_knowledge.kernel_processor.skill_extractor import extract_skills_2pass
        from kaggle_knowledge.kernel_processor.notebook_parser import parse_notebook, notebook_to_text, get_kernel_competition
        skills = []
        for kdir in kernel_dirs:
            kpath = Path(kdir)
            ipynb_files = list(kpath.glob("*.ipynb"))
            if not ipynb_files: continue
            meta_files = list(kpath.glob("kernel-metadata.json"))
            competition = get_kernel_competition(str(meta_files[0])) if meta_files else "unknown"
            print(f"  处理: {kpath.name} ({competition})")
            try:
                cells = parse_notebook(str(ipynb_files[0])); text = notebook_to_text(cells, max_len=30000)
                result = extract_skills_2pass(text, kpath.name, competition, keyword, self.llm)
                if result:
                    from kaggle_knowledge.kernel_processor import save_skill as _sv3
                    for sk in result:
                        _sv3(keyword, sk, self.skill_library_dir)
                skills.extend(result)
                print(f"    提取到 {len(result)} 条")
            except Exception as e:
                print(f"    出错: {e}")
        return skills, None
    @staticmethod
    def _chunk(text: str, chunk_type: str = "text") -> StreamEvent:
        """创建文本进度事件"""
        return StreamEvent.create(
            StreamEventType.LLM_CHUNK,
            "",
            chunk=text,
            chunk_type=chunk_type,
        )

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
        from kaggle_knowledge.search_competitions import search_competitions_paginated
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
        search_max_pages = config.get("search_max_pages", 5)
        save_csv = config.get("save_csv", {})

        print(f"\n  [下载] 搜索关键词 '{keyword}' 的竞赛...")
        competitions = search_competitions_paginated(
            keyword,
            max_competitions=competitions_per_keyword,
            save_csv_flag=save_csv.get("competitions", True),
            output_dir=output_dir,
            min_team_count=min_team_count,
            max_pages=search_max_pages,
        )

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
            user_ranks = {e.get("teamName",""): i for i, e in enumerate(leaderboard, 1)}
            print(f"         排行榜前 {len(usernames)} 名用户: {usernames[:3]}...")

            download_competition_kernels(
                slug, name, usernames,
                keyword=keyword,
                max_kernels_per_user=kernels_per_user,
                base_output_dir=output_dir,
                save_csv_flag=save_csv.get("kernels_list", True),
                user_ranks=user_ranks,
            )

        print(f"  [下载] 关键词 '{keyword}' 下载完成\n")

    def _parse_input(self, input_text: str) -> list:
        """Parse into list of (keyword, kernel_dirs). Supports comma-separated."""
        input_text = input_text.strip()
        try:
            data = json.loads(input_text)
            if isinstance(data, dict):
                kw, kd = data.get("keyword",""), data.get("kernel_dirs",[])
                if kw and kd: return [(kw, kd)]
        except (json.JSONDecodeError, TypeError): pass
        keywords = [k.strip() for k in input_text.split(",") if k.strip()]
        if not keywords: keywords = [input_text]
        return [(kw, self._find_kernel_dirs(kw)) for kw in keywords]

    def _find_kernel_dirs(self, keyword: str) -> List[str]:
        """递归搜索 keyword 目录下所有 kernel 目录。

        kernel 目录的判断标准：包含 kernel-metadata.json 文件。
        支持多层嵌套结构: output/<keyword>/<competition>/kernels/<kernel_dir>

        Tries: exact keyword → sanitized version → substring match
        """
        kw_safe = self._sanitize_keyword(keyword)
        base = Path(self.kernels_base_dir)

        def _find_keyword_root():
            """找到 keyword 对应的根目录"""
            # Try exact keyword as-is
            p = base / keyword
            if p.is_dir():
                return p
            # Try sanitized version
            p = base / kw_safe
            if p.is_dir():
                return p
            # Try substring match
            if base.is_dir():
                for d in base.iterdir():
                    if d.is_dir() and (keyword in d.name or kw_safe in d.name.lower()):
                        return d
            return None

        def _find_kernel_dirs_recursive(root: Path) -> List[str]:
            """递归查找所有包含 kernel-metadata.json 的目录"""
            results = []
            try:
                for item in root.rglob("kernel-metadata.json"):
                    kernel_dir = str(item.parent)
                    if kernel_dir not in results:
                        results.append(kernel_dir)
            except Exception:
                pass
            return results

        root = _find_keyword_root()
        if root is None:
            return []

        return _find_kernel_dirs_recursive(root)

    def _get_top_skills(self) -> int:
        """从 config.json 读取 top_skills 配置，默认 5"""
        try:
            from kaggle_knowledge.utils import load_config
            kg_dir = Path(__file__).parent.parent.parent / "kaggle_knowledge"
            config = load_config(str(kg_dir / "config.json"))
            return int(config.get("top_skills", 5))
        except Exception:
            return 5

    def _get_min_skills(self) -> int:
        """从 config.json 读取 min_skills_per_keyword 配置，默认 10"""
        try:
            from kaggle_knowledge.utils import load_config
            kg_dir = Path(__file__).parent.parent.parent / "kaggle_knowledge"
            config = load_config(str(kg_dir / "config.json"))
            return int(config.get("min_skills_per_keyword", 10))
        except Exception:
            return 10

    def _get_min_kernels(self) -> int:
        """从 config.json 读取 min_kernels_per_keyword 配置，默认 3"""
        try:
            from kaggle_knowledge.utils import load_config
            kg_dir = Path(__file__).parent.parent.parent / "kaggle_knowledge"
            config = load_config(str(kg_dir / "config.json"))
            return int(config.get("min_kernels_per_keyword", 3))
        except Exception:
            return 3

    def _build_compact_skill_summary(self, keyword: str, top_k: int = None) -> str:
        """构建紧凑技能摘要（只有名称+影响力+一句话描述，不含完整 technique/code）。
        用于返回给 Agent 的结果，避免过长内容被 ReActAgent 层层回音。
        """
        from kaggle_knowledge.kernel_processor import match_keyword
        from kaggle_knowledge.kernel_processor.skill_library import search_skills
        if top_k is None:
            top_k = self._get_top_skills()
        matched = match_keyword(keyword, self.skill_library_dir)
        if not matched:
            return ""
        skills = search_skills(matched, self.skill_library_dir, top_k=top_k)
        if not skills:
            return ""
        lines = [f"## {keyword} 方向 — {len(skills)} 条技巧"]
        for i, s in enumerate(skills, 1):
            impact = s.get('estimated_impact', '')
            impact_tag = f" [{impact}]" if impact else ""
            cat = s.get('category', '')
            desc = s.get('description', '')
            lines.append(f"{i}. **{s['name']}**{impact_tag} ({cat}) — {desc}")
        return "\n".join(lines)

    @staticmethod
    def _sanitize_keyword(keyword: str) -> str:
        name = keyword.lower().strip()
        name = re.sub(r'[^a-z0-9_\s]', '', name)
        name = re.sub(r'\s+', '_', name)
        return name[:80]
