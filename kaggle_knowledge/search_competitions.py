"""
搜索Kaggle竞赛模块
根据关键词搜索竞赛，并按参与人数从多到少排序
支持分页搜索、已处理竞赛去重、参赛人数过滤
"""

from typing import List, Dict, Optional, Set
import os
import csv
from pathlib import Path
from utils import (
    run_kaggle_command, parse_csv_data, save_csv,
    print_section, print_result
)


def _load_existing_competition_refs(keyword: str, output_dir: str) -> Set[str]:
    """从所有 competitions_*.csv 加载已处理竞赛的 ref 集合（跨关键词去重）

    Args:
        keyword: 搜索关键词（仅用于日志，不影响加载逻辑）
        output_dir: 输出目录

    Returns:
        已处理竞赛的 ref 集合，用于去重
    """
    refs = set()
    if not os.path.isdir(output_dir):
        return refs

    for filename in os.listdir(output_dir):
        if not filename.startswith("competitions_") or not filename.endswith(".csv"):
            continue
        filepath = os.path.join(output_dir, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    ref = row.get("ref", "").strip()
                    if ref:
                        refs.add(ref)
        except Exception:
            continue
    return refs


def _append_competitions_csv(competitions: List[Dict], keyword: str, output_dir: str):
    """将新竞赛追加到 CSV 文件（不覆盖已有数据）

    Args:
        competitions: 新竞赛列表
        keyword: 搜索关键词
        output_dir: 输出目录
    """
    if not competitions:
        return

    safe_keyword = "".join(c if c.isalnum() or c in " -_" else "_" for c in keyword)
    filename = f"competitions_{safe_keyword}.csv"
    filepath = os.path.join(output_dir, filename)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    file_exists = os.path.exists(filepath)
    try:
        keys = list(competitions[0].keys())
        with open(filepath, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            if not file_exists:
                writer.writeheader()
            writer.writerows(competitions)
    except Exception as e:
        print(f"  ⚠️ 追加 CSV 失败: {e}")


def search_competitions(
    keyword: str,
    max_competitions: int = 5,
    save_csv_flag: bool = True,
    output_dir: str = "output",
    min_team_count: int = 0,
    page: int = 1,
) -> Optional[List[Dict]]:
    """
    搜索指定关键词的竞赛（单页）

    Args:
        keyword: 搜索关键词
        max_competitions: 最多获取多少个竞赛
        save_csv_flag: 是否保存为CSV
        output_dir: 输出目录
        min_team_count: 最低参赛队伍数，低于此值的竞赛被跳过（0=不过滤）
        page: 页码（从1开始）

    Returns:
        竞赛列表，每个元素是包含竞赛信息的字典
    """
    print_section(f"搜索竞赛: '{keyword}' (第{page}页)")

    # 构建kaggle命令，使用 -p 参数翻页
    cmd = [
        "kaggle", "competitions", "list",
        "--search", keyword,
        "--sort-by", "numberOfTeams",
        "--csv",
        "-p", str(page)
    ]

    success, stdout, stderr = run_kaggle_command(cmd)

    if not success:
        print(f"❌ 搜索竞赛失败: {stderr}")
        return None

    # 解析CSV数据
    competitions = parse_csv_data(stdout)

    if not competitions:
        print(f"⚠️ 关键词 '{keyword}' 第{page}页没有找到竞赛")
        return None

    # 过滤参赛人数不足的竞赛
    if min_team_count > 0:
        before = len(competitions)
        competitions = [
            c for c in competitions
            if int(c.get("teamCount", 0)) >= min_team_count
        ]
        skipped = before - len(competitions)
        if skipped > 0:
            print(f"  (已过滤 {skipped} 个参赛人数不足 {min_team_count} 的竞赛)")

    # 取前N个竞赛
    competitions = competitions[:max_competitions]

    print(f"✅ 找到竞赛: {len(competitions)} 个")

    # 打印结果
    print_result(competitions, limit=5)

    return competitions


def search_competitions_paginated(
    keyword: str,
    max_competitions: int = 5,
    save_csv_flag: bool = True,
    output_dir: str = "output",
    min_team_count: int = 0,
    max_pages: int = 5,
) -> Optional[List[Dict]]:
    """分页搜索竞赛，跳过已处理的竞赛，直到找到足够数量或下一页不满足条件。

    Args:
        keyword: 搜索关键词
        max_competitions: 总共最多获取多少个竞赛
        save_csv_flag: 是否保存为CSV
        output_dir: 输出目录
        min_team_count: 最低参赛队伍数，低于此值的竞赛被跳过，
                       若整页所有竞赛都不满足，停止翻页
        max_pages: 最多翻多少页

    Returns:
        竞赛列表（已去重），每个元素是包含竞赛信息的字典
    """
    print_section(f"分页搜索竞赛: '{keyword}' (最多{max_pages}页)")

    # 加载已处理的竞赛 ref（去重用）
    existing_refs = _load_existing_competition_refs(keyword, output_dir)
    if existing_refs:
        print(f"  已有 {len(existing_refs)} 个已处理竞赛，将跳过")

    all_new_competitions = []

    for page in range(1, max_pages + 1):
        competitions = search_competitions(
            keyword,
            max_competitions=max_competitions,
            save_csv_flag=False,  # 在外层手动保存
            output_dir=output_dir,
            min_team_count=0,  # 先不过滤，用于判断整页是否都不满足
            page=page,
        )

        if not competitions:
            print(f"  第{page}页无结果，停止翻页")
            break

        # 检查：整页所有竞赛的 teamCount 是否都低于阈值
        if min_team_count > 0:
            all_below_threshold = all(
                int(c.get("teamCount", 0)) < min_team_count
                for c in competitions
            )
            if all_below_threshold:
                print(f"  ⚠️ 第{page}页所有竞赛参赛人数均低于 {min_team_count}，停止翻页")
                break

        # 过滤参赛人数不足的
        if min_team_count > 0:
            competitions = [
                c for c in competitions
                if int(c.get("teamCount", 0)) >= min_team_count
            ]

        # 去重：跳过已处理的竞赛
        new_comps = [c for c in competitions if c.get("ref", "") not in existing_refs]
        skipped = len(competitions) - len(new_comps)
        if skipped > 0:
            print(f"  跳过 {skipped} 个已处理的竞赛")

        if new_comps:
            all_new_competitions.extend(new_comps)
            # 标记为已处理
            for c in new_comps:
                ref = c.get("ref", "")
                if ref:
                    existing_refs.add(ref)
            # 追加保存到 CSV
            if save_csv_flag:
                _append_competitions_csv(new_comps, keyword, output_dir)

        print(f"  第{page}页新增 {len(new_comps)} 个，累计 {len(all_new_competitions)} 个")

        # 如果已经找到足够的竞赛，停止翻页
        if len(all_new_competitions) >= max_competitions:
            print(f"  已找到足够竞赛 ({len(all_new_competitions)} >= {max_competitions})，停止翻页")
            break

    if not all_new_competitions:
        print(f"⚠️ 关键词 '{keyword}' 未找到新的竞赛（所有竞赛均已处理）")
        return None

    # 截断到 max_competitions
    result = all_new_competitions[:max_competitions]
    print(f"✅ 分页搜索完成: 共找到 {len(result)} 个新竞赛")
    print_result(result, limit=5)

    return result


def batch_search_competitions(
    keywords: List[str],
    max_competitions: int = 5,
    save_csv_flag: bool = True,
    output_dir: str = "output",
    min_team_count: int = 0,
    max_pages: int = 5,
) -> Dict[str, List[Dict]]:
    """
    批量搜索多个关键词的竞赛（使用分页搜索）

    Args:
        keywords: 关键词列表
        max_competitions: 每个关键词最多获取多少个竞赛
        save_csv_flag: 是否保存为CSV
        output_dir: 输出目录
        min_team_count: 最低参赛队伍数
        max_pages: 最多翻页数

    Returns:
        字典，键为关键词，值为竞赛列表
    """
    results = {}

    print_section(f"开始批量搜索竞赛，共 {len(keywords)} 个关键词")

    for idx, keyword in enumerate(keywords, 1):
        print(f"\n[{idx}/{len(keywords)}] 处理关键词: '{keyword}'")

        competitions = search_competitions_paginated(
            keyword,
            max_competitions=max_competitions,
            save_csv_flag=save_csv_flag,
            output_dir=output_dir,
            min_team_count=min_team_count,
            max_pages=max_pages,
        )

        if competitions:
            results[keyword] = competitions
        else:
            results[keyword] = []

    return results
