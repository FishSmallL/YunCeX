"""
获取竞赛排行榜模块
获取竞赛的排行榜，并提取前N名用户
"""

from typing import List, Dict, Optional
from utils import (
    run_kaggle_command, parse_csv_data, save_csv,
    print_section, print_result
)


def get_leaderboard(
    competition_slug: str,
    top_users: int = 5,
    min_score: Optional[float] = None,
    save_csv_flag: bool = True,
    output_dir: str = "output"
) -> Optional[List[Dict]]:
    """
    获取指定竞赛的排行榜
    
    Args:
        competition_slug: 竞赛slug（竞赛ID）
        top_users: 获取前多少名用户
        save_csv_flag: 是否保存为CSV
        output_dir: 输出目录
        
    Returns:
        排行榜用户列表，每个元素包含用户信息（teamName等字段）
    """
    print_section(f"获取排行榜: {competition_slug}")
    
    # 构建kaggle命令
    cmd = [
        "kaggle", "competitions", "leaderboard",
        competition_slug,
        "--show",  # 显示排行榜
        "--csv",   # CSV格式输出
        "--page-size", "50"  # 一次获取更多用户
    ]
    
    success, stdout, stderr = run_kaggle_command(cmd)
    
    if not success:
        print(f"❌ 获取排行榜失败: {stderr}")
        return None
    
    # 解析CSV数据
    leaderboard = parse_csv_data(stdout)
    
    if not leaderboard:
        print(f"⚠️ 竞赛 '{competition_slug}' 的排行榜为空")
        return None
    
    # 如果指定了最小分数阈值，先过滤掉低分条目
    if min_score is not None:
        filtered = []
        skipped = 0
        for entry in leaderboard:
            score_raw = entry.get("score")
            try:
                score_val = float(score_raw) if score_raw is not None else None
            except Exception:
                score_val = None

            if score_val is None:
                # 无法解析分数，跳过并打印提示
                skipped += 1
                continue

            if score_val >= min_score:
                filtered.append(entry)
            else:
                skipped += 1

        print(f"  ℹ️ 已过滤低分（<{min_score}）的条目: {skipped} 条，保留: {len(filtered)} 条")
        leaderboard = filtered

    # 取前N个用户
    leaderboard = leaderboard[:top_users]
    
    print(f"✅ 获取排行榜用户: {len(leaderboard)} 人")
    
    # 打印结果
    print_result(leaderboard, limit=5)
    
    # 保存CSV
    if save_csv_flag:
        safe_comp = "".join(c if c.isalnum() or c in " -_" else "_" for c in competition_slug)
        filename = f"leaderboard_{safe_comp}.csv"
        save_csv(leaderboard, filename, output_dir)
    
    return leaderboard


def extract_usernames(leaderboard: List[Dict]) -> List[str]:
    """
    从排行榜提取用户名
    
    Args:
        leaderboard: 排行榜数据
        
    Returns:
        用户名列表
    """
    usernames = []

    # 排行榜中用户名字段是 teamName
    for idx, entry in enumerate(leaderboard, 1):
        tn = entry.get("teamName")
        if not tn:
            print(f"  ⚠️ 跳过第 {idx} 条：缺少 teamName 字段或值为空")
            continue
        if not isinstance(tn, str):
            print(f"  ⚠️ 跳过第 {idx} 条：teamName 不是字符串，值={tn}")
            continue

        username = tn.strip()
        if username:
            usernames.append(username)
        else:
            print(f"  ⚠️ 跳过第 {idx} 条：teamName 为空字符串")

    return usernames
