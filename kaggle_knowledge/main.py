"""
Kaggle竞赛数据采集工具 - 主程序入口

功能流程:
1. 根据关键词搜索竞赛（前5个，按参与人数排序）
2. 获取每个竞赛的排行榜（前5名用户）
3. 下载每个用户在该竞赛上的kernel（前5个）

使用方法:
  python main.py                                    # 使用配置文件中的关键词
  python main.py "NLP" "Computer Vision"           # 使用命令行参数（需要用引号）
  python main.py "machine learning" "deep learning" # 支持多词关键词
"""

import sys
import os
from typing import List, Dict, Optional
from pathlib import Path

from utils import load_config, print_section, extract_competition_slug
from search_competitions import batch_search_competitions
from get_leaderboard import get_leaderboard, extract_usernames
from download_kernels import download_competition_kernels


def parse_arguments() -> Optional[List[str]]:
    """
    解析命令行参数
    
    关键词必须用引号包裹，例如：
      python main.py "machine learning" "deep learning"
    
    Returns:
        关键词列表，如果没有参数则返回None
    """
    if len(sys.argv) > 1:
        # 命令行参数优先级最高
        keywords = sys.argv[1:]
        
        # 检查是否有单个词的参数（可能是没有用引号导致的分割）
        single_word_params = [kw for kw in keywords if len(kw.split()) == 1 and kw.islower()]
        if len(single_word_params) >= 2:
            print(f"⚠️ 警告：检测到可能未用引号包裹的参数: {keywords}")
            print(f"   请使用引号包裹多词关键词，例如：")
            print(f'   python main.py "machine learning" "deep learning"')
            print()
        
        print(f"📌 使用命令行参数覆盖配置文件，关键词: {keywords}")
        return keywords
    return None


def main():
    """主程序"""
    
    print("\n" + "=" * 60)
    print("  Kaggle竞赛数据采集工具")
    print("=" * 60)
    
    # 加载配置文件
    config = load_config("config.json")
    
    # 获取关键词
    keywords = parse_arguments()
    if keywords is None:
        keywords = config.get("keywords", [])
    
    if not keywords:
        print("❌ 没有输入关键词，请通过命令行参数或修改config.json")
        sys.exit(1)
    
    print(f"\n📋 关键词: {keywords}")
    print(f"📁 输出目录: {config.get('output_dir', 'output')}")
    
    # 提取配置参数
    output_dir = config.get("output_dir", "output")
    competitions_per_keyword = config.get("competitions_per_keyword", 5)
    top_leaderboard_users = config.get("top_leaderboard_users", 5)
    min_leaderboard_score = config.get("min_leaderboard_score")
    kernels_per_user = config.get("kernels_per_user", 5)
    save_csv_config = config.get("save_csv", {})
    
    # ============ 第一步：搜索竞赛 ============
    print_section("第一步：搜索竞赛")
    
    competitions_dict = batch_search_competitions(
        keywords,
        max_competitions=competitions_per_keyword,
        save_csv_flag=save_csv_config.get("competitions", True),
        output_dir=output_dir
    )
    
    # 统计竞赛总数
    total_competitions = sum(len(comps) for comps in competitions_dict.values())
    print(f"\n✅ 搜索完成，总共找到 {total_competitions} 个竞赛")
    
    if total_competitions == 0:
        print("⚠️ 没有找到任何竞赛，程序退出")
        return
    
    # ============ 第二步和第三步：获取排行榜并下载kernel ============
    
    statistics = {
        "total_competitions": 0,
        "competitions_with_leaderboard": 0,
        "total_users": 0,
        "total_kernels_downloaded": 0,
        "failed_downloads": []
    }
    
    for keyword in keywords:
        competitions = competitions_dict.get(keyword, [])
        
        if not competitions:
            continue
        
        print_section(f"处理关键词 '{keyword}' 的竞赛")
        
        for comp_idx, comp in enumerate(competitions, 1):
            competition_ref = comp.get("ref", "")
            competition_slug = extract_competition_slug(competition_ref)
            competition_name = comp.get("title", "")
            
            print(f"\n[{comp_idx}/{len(competitions)}] {competition_name} ({competition_slug})")
            
            statistics["total_competitions"] += 1
            
            # ============ 第二步：获取排行榜 ============
            leaderboard = get_leaderboard(
                competition_slug,
                top_users=top_leaderboard_users,
                min_score=min_leaderboard_score,
                save_csv_flag=save_csv_config.get("leaderboard", True),
                output_dir=output_dir
            )
            
            if not leaderboard:
                print(f"⚠️ 无法获取排行榜，跳过此竞赛")
                continue
            
            statistics["competitions_with_leaderboard"] += 1
            
            # 提取用户名
            usernames = extract_usernames(leaderboard)
            statistics["total_users"] += len(usernames)
            
            print(f"  提取用户: {usernames}")
            
            # ============ 第三步：下载kernel ============
            download_competition_kernels(
                competition_slug,
                competition_name,
                usernames,
                keyword=keyword,
                max_kernels_per_user=kernels_per_user,
                base_output_dir=output_dir,
                save_csv_flag=save_csv_config.get("kernels_list", True)
            )
    
    # ============ 最终统计 ============
    print_section("采集完成 - 统计结果")
    
    print(f"""
  📊 统计数据:
  
  关键词总数: {len(keywords)}
  搜索到竞赛: {statistics['total_competitions']} 个
  有排行榜的竞赛: {statistics['competitions_with_leaderboard']} 个
  扫描用户总数: {statistics['total_users']} 人
  
  📁 输出目录: {output_dir}
  
  ✅ 数据采集流程完成！
  
  说明:
  - 竞赛数据已保存在各关键词目录下
  - 每个竞赛为一个文件夹，包含对应用户的kernel代码
  - 如果启用了CSV保存，结果已导出到csv_results目录
    """)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ 程序被中断")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
