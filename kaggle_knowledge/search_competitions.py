"""
搜索Kaggle竞赛模块
根据关键词搜索竞赛，并按参与人数从多到少排序
"""

from typing import List, Dict, Optional
from utils import (
    run_kaggle_command, parse_csv_data, save_csv, 
    print_section, print_result
)


def search_competitions(
    keyword: str,
    max_competitions: int = 5,
    save_csv_flag: bool = True,
    output_dir: str = "output",
    min_team_count: int = 0,
) -> Optional[List[Dict]]:
    """
    搜索指定关键词的竞赛

    Args:
        keyword: 搜索关键词
        max_competitions: 最多获取多少个竞赛
        save_csv_flag: 是否保存为CSV
        output_dir: 输出目录
        min_team_count: 最低参赛队伍数，低于此值的竞赛被跳过（0=不过滤）

    Returns:
        竞赛列表，每个元素是包含竞赛信息的字典
    """
    print_section(f"搜索竞赛: '{keyword}'")

    # 构建kaggle命令
    cmd = [
        "kaggle", "competitions", "list",
        "--search", keyword,
        "--sort-by", "numberOfTeams",  # 按参与人数排序
        "--csv",
        "--page-size", "30"  # 一次获取更多结果
    ]

    success, stdout, stderr = run_kaggle_command(cmd)

    if not success:
        print(f"❌ 搜索竞赛失败: {stderr}")
        return None

    # 解析CSV数据
    competitions = parse_csv_data(stdout)

    if not competitions:
        print(f"⚠️ 关键词 '{keyword}' 没有找到竞赛")
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

    # 保存CSV
    if save_csv_flag:
        safe_keyword = "".join(c if c.isalnum() or c in " -_" else "_" for c in keyword)
        filename = f"competitions_{safe_keyword}.csv"
        save_csv(competitions, filename, output_dir)

    return competitions


def batch_search_competitions(
    keywords: List[str],
    max_competitions: int = 5,
    save_csv_flag: bool = True,
    output_dir: str = "output",
    min_team_count: int = 0,
) -> Dict[str, List[Dict]]:
    """
    批量搜索多个关键词的竞赛

    Args:
        keywords: 关键词列表
        max_competitions: 每个关键词最多获取多少个竞赛
        save_csv_flag: 是否保存为CSV
        output_dir: 输出目录
        min_team_count: 最低参赛队伍数

    Returns:
        字典，键为关键词，值为竞赛列表
    """
    results = {}

    print_section(f"开始批量搜索竞赛，共 {len(keywords)} 个关键词")

    for idx, keyword in enumerate(keywords, 1):
        print(f"\n[{idx}/{len(keywords)}] 处理关键词: '{keyword}'")

        competitions = search_competitions(
            keyword,
            max_competitions=max_competitions,
            save_csv_flag=save_csv_flag,
            output_dir=output_dir,
            min_team_count=min_team_count,
        )

        if competitions:
            results[keyword] = competitions
        else:
            results[keyword] = []

    return results
