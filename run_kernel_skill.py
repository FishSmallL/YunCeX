"""
测试 KernelSkillAgent — 从已下载的 Kaggle kernel 中提取可复用技巧。

使用方法:
  1. 确保 .env 文件中配置了 LLM_API_KEY
  2. conda activate cameltrack
  3. python run_kernel_skill.py

可选命令行参数:
  python run_kernel_skill.py                        # 处理所有已下载的 kernel
  python run_kernel_skill.py "machine learning"     # 指定关键词
  python run_kernel_skill.py --dry-run              # 只预览将要处理的 kernel，不实际调用 LLM
"""

import sys
import os
import time
from pathlib import Path

# 确保项目根目录在 Python path 中
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

if not os.getenv("LLM_API_KEY"):
    print("错误: 未配置 LLM_API_KEY，请在项目根目录创建 .env 文件，参考 .env example")
    print("  LLM_API_KEY=your_key_here")
    print("  LLM_MODEL_ID=deepseek-v4-pro")
    print("  LLM_BASE_URL=https://api.deepseek.com")
    sys.exit(1)

from hello_agents.core.llm import HelloAgentsLLM
from hello_agents.core.config import Config
from hello_agents.agents.kernel_skill_agent import KernelSkillAgent

from kaggle_knowledge.kernel_processor import (
    parse_notebook,
    notebook_to_text,
    get_kernel_competition,
    extract_skills_from_notebooks_batch,
    save_skill,
    match_keyword,
    search_skills,
    build_skill_context,
    list_keywords,
    get_library_stats,
)

# ── 路径配置 ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
KERNELS_BASE = PROJECT_ROOT / "kaggle_knowledge" / "output"
SKILL_LIBRARY = PROJECT_ROOT / "kaggle_knowledge" / "skill_library"


def _now() -> float:
    return time.perf_counter()


def _fmt_seconds(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.1f} ms"
    if seconds < 60:
        return f"{seconds:.2f} s"
    minutes = int(seconds // 60)
    remainder = seconds - minutes * 60
    return f"{minutes}m {remainder:.1f}s"


def find_all_kernel_dirs(keyword: str) -> list:
    """在 output/<keyword>/kernels/ 下查找所有 kernel 目录

    依次尝试: 原始关键词 → sanitize → 子串匹配
    """
    def _subdirs(path):
        return sorted([str(d) for d in path.iterdir() if d.is_dir()]) if path.is_dir() else []

    kw_safe = keyword.lower().strip().replace(" ", "_")

    # 1. 原始关键词（保留空格，如 "machine learning"）
    result = _subdirs(KERNELS_BASE / keyword / "kernels")
    if result:
        return result

    # 2. sanitized 版本（如 "machine_learning"）
    result = _subdirs(KERNELS_BASE / kw_safe / "kernels")
    if result:
        return result

    # 3. 子串匹配
    for d in KERNELS_BASE.iterdir():
        if d.is_dir() and (keyword in d.name or kw_safe in d.name.lower()):
            result = _subdirs(d / "kernels")
            if result:
                return result

    return []


def preview_kernels(keyword: str, kernel_dirs: list):
    """预览将要处理的 kernel 列表"""
    t_preview = _now()
    print(f"\n关键词: {keyword}")
    print(f"Kernel 目录数: {len(kernel_dirs)}")
    print(f"Skill 库路径: {SKILL_LIBRARY}")
    print()
    for i, kdir in enumerate(kernel_dirs, 1):
        kpath = Path(kdir)
        ipynb_files = list(kpath.glob("*.ipynb"))
        meta_files = list(kpath.glob("kernel-metadata.json"))
        ipynb_name = ipynb_files[0].name if ipynb_files else "(未找到)"
        competition = get_kernel_competition(str(meta_files[0])) if meta_files else "unknown"
        print(f"  [{i}] {kpath.name}")
        print(f"      文件: {ipynb_name}")
        print(f"      竞赛: {competition}")

        # 预览解析
        if ipynb_files:
            t_parse = _now()
            cells = parse_notebook(str(ipynb_files[0]))
            text = notebook_to_text(cells, max_len=300)
            parse_cost = _now() - t_parse
            print(f"      cells: {len(cells)} (code={sum(1 for c in cells if c['cell_type']=='code')}, "
                  f"md={sum(1 for c in cells if c['cell_type']=='markdown')})")
            print(f"      压缩预览 ({len(text)} chars):")
            print(f"      {text[:200]}...")
            print(f"      解析耗时: {_fmt_seconds(parse_cost)}")
        print()
    print(f"预览总耗时: {_fmt_seconds(_now() - t_preview)}")


def run_extraction(keyword: str, kernel_dirs: list):
    """实际运行 LLM 提取流程"""
    t_total = _now()
    print(f"\n{'='*60}")
    print(f"  开始提取: 关键词='{keyword}', {len(kernel_dirs)} 个 kernel")
    print(f"{'='*60}")

    llm = HelloAgentsLLM()
    print(f"LLM 模型: {llm.model}")

    # 初始化 Skill 库目录
    SKILL_LIBRARY.mkdir(parents=True, exist_ok=True)

    # 提取前统计
    print(f"\n提取前 Skill 库状态:")
    t_stats = _now()
    stats_before = get_library_stats(str(SKILL_LIBRARY))
    print(f"  统计耗时: {_fmt_seconds(_now() - t_stats)}")
    print(f"  关键词目录: {stats_before['keywords']} 个")
    print(f"  技能总数: {stats_before['total_skills']} 条")

    # 逐个 kernel 处理并实时展示
    all_skills = []
    for i, kdir in enumerate(kernel_dirs, 1):
        kpath = Path(kdir)
        ipynb_files = list(kpath.glob("*.ipynb"))
        if not ipynb_files:
            continue
        meta_files = list(kpath.glob("kernel-metadata.json"))
        competition = get_kernel_competition(str(meta_files[0])) if meta_files else "unknown"

        print(f"\n── [{i}/{len(kernel_dirs)}] {kpath.name} ──")

        t_parse = _now()
        cells = parse_notebook(str(ipynb_files[0]))
        text = notebook_to_text(cells, max_len=8000)
        print(f"  解析: {len(cells)} cells → {len(text)} chars 压缩文本")
        print(f"  解析耗时: {_fmt_seconds(_now() - t_parse)}")

        from kaggle_knowledge.kernel_processor.skill_extractor import extract_skills_from_notebook
        t_extract = _now()
        skills = extract_skills_from_notebook(
            text, kpath.name, competition, keyword, llm
        )
        print(f"  提取: {len(skills)} 条技巧")
        print(f"  提取耗时: {_fmt_seconds(_now() - t_extract)}")
        for s in skills:
            cat = s.get("category", "?")
            desc = s.get("description", "?")[:80]
            print(f"    [{cat}] {desc}")

        # 即时保存
        t_save = _now()
        for s in skills:
            save_skill(keyword, s, str(SKILL_LIBRARY))
        all_skills.extend(skills)
        print(f"  保存耗时: {_fmt_seconds(_now() - t_save)}")

    # 提取后统计
    print(f"\n{'='*60}")
    print(f"  提取完成")
    print(f"{'='*60}")
    t_stats_after = _now()
    stats_after = get_library_stats(str(SKILL_LIBRARY))
    print(f"  统计耗时: {_fmt_seconds(_now() - t_stats_after)}")
    print(f"\nSkill 库状态变化:")
    print(f"  关键词目录: {stats_before['keywords']} → {stats_after['keywords']} 个")
    print(f"  技能总数: {stats_before['total_skills']} → {stats_after['total_skills']} 条")
    print(f"  本次新增: {len(all_skills)} 条")
    print(f"\n分类分布:")
    for cat, count in sorted(stats_after.get("categories", {}).items()):
        print(f"  {cat}: {count} 条")

    # 展示提取结果
    print(f"\n── 提取的技能预览 ──")
    for i, s in enumerate(all_skills, 1):
        print(f"\n  [{i}] {s.get('name', '?')}")
        print(f"      分类: {s.get('category', '?')}")
        print(f"      描述: {s.get('description', '?')}")
        tech = s.get('technique', '')
        if len(tech) > 150:
            tech = tech[:150] + "..."
        print(f"      技巧: {tech}")

    # 验证：搜索刚入库的技能
    print(f"\n── 验证匹配 ──")
    matched = match_keyword(keyword, str(SKILL_LIBRARY))
    if matched:
        print(f"匹配到目录: {matched}")
        t_search = _now()
        skills_found = search_skills(matched, str(SKILL_LIBRARY), top_k=3)
        print(f"搜索耗时: {_fmt_seconds(_now() - t_search)}")
        print(f"取 top-3 条:")
        for s in skills_found:
            print(f"  - {s['name']}: {s['description'][:60]}")
        print(f"\n可注入 Agent 的上下文 (前300字):")
        t_ctx = _now()
        ctx = build_skill_context(matched, str(SKILL_LIBRARY), top_k=2)
        print(f"上下文构建耗时: {_fmt_seconds(_now() - t_ctx)}")
        print(ctx[:300])
    else:
        print("未匹配到（这不应该发生）")
    print(f"\n总耗时: {_fmt_seconds(_now() - t_total)}")


def main():
    t_main = _now()
    dry_run = "--dry-run" in sys.argv
    keyword = None

    for arg in sys.argv[1:]:
        if not arg.startswith("--"):
            keyword = arg
            break

    if keyword is None:
        if KERNELS_BASE.is_dir():
            for d in sorted(KERNELS_BASE.iterdir()):
                if d.is_dir() and (d / "kernels").is_dir():
                    keyword = d.name
                    break
        if keyword is None:
            keyword = "machine learning"

    print(f"关键词: {keyword}")

    if dry_run:
        print("=" * 60)
        print("  DRY RUN — 仅预览，不调用 LLM")
        print("=" * 60)
        t_find = _now()
        kernel_dirs = find_all_kernel_dirs(keyword)
        print(f"查找 kernel 耗时: {_fmt_seconds(_now() - t_find)}")
        if kernel_dirs:
            preview_kernels(keyword, kernel_dirs)
        else:
            print(f"本地没有 '{keyword}' 的 kernel，正式运行时将自动从 Kaggle 下载。")
            print(f"(使用 kaggle_knowledge/config.json 配置下载参数)")
        print(f"\nDRY RUN 总耗时: {_fmt_seconds(_now() - t_main)}")
        return

    # 正式运行：使用 KernelSkillAgent（自动匹配 skill 库 / 下载 / 提取）
    print(f"\n{'='*60}")
    print(f"  启动 KernelSkillAgent（含自动下载）")
    print(f"{'='*60}")

    llm = HelloAgentsLLM()
    print(f"LLM 模型: {llm.model}")

    config = Config()
    agent = KernelSkillAgent(
        name="skill-extractor",
        llm=llm,
        config=config,
        skill_library_dir=str(SKILL_LIBRARY),
        kernels_base_dir=str(KERNELS_BASE),
    )

    # 先展示本地已有的 kernel
    t_find = _now()
    kernel_dirs = find_all_kernel_dirs(keyword)
    print(f"查找 kernel 耗时: {_fmt_seconds(_now() - t_find)}")
    if kernel_dirs:
        preview_kernels(keyword, kernel_dirs)
        input("\n按 Enter 开始 LLM 提取（会消耗 API tokens，若无本地 kernel 会自动下载）...")
    else:
        print(f"\n本地没有 '{keyword}' 的 kernel，将自动从 Kaggle 下载后再提取。")
        input("按 Enter 开始...")

    t_agent = _now()
    result = agent.run(keyword)
    print(f"Agent 运行耗时: {_fmt_seconds(_now() - t_agent)}")
    print(result)
    print(f"\n脚本总耗时: {_fmt_seconds(_now() - t_main)}")


if __name__ == "__main__":
    main()
