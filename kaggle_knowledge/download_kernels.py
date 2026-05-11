"""
下载Kernel代码模块
搜索指定用户在竞赛上的kernel，并下载它们
"""

from typing import List, Dict, Optional
import os
from pathlib import Path
from utils import (
    run_kaggle_command, parse_csv_data, save_csv,
    print_section, print_result, create_directory_structure
)


def get_user_kernels(
    competition_slug: str,
    username: str,
    max_kernels: int = 5,
    save_csv_flag: bool = True,
    output_dir: str = "output"
) -> Optional[List[Dict]]:
    """
    获取指定用户在指定竞赛上的kernel列表
    
    Args:
        competition_slug: 竞赛slug
        username: 用户名
        max_kernels: 最多获取多少个kernel
        save_csv_flag: 是否保存为CSV
        output_dir: CSV输出目录
        
    Returns:
        kernel列表，每个元素是包含kernel信息的字典
    """
    print(f"  📖 查询用户 '{username}' 在竞赛 '{competition_slug}' 上的kernel...")
    
    # 构建kaggle命令
    cmd = [
        "kaggle", "kernels", "list",
        "--competition", competition_slug,
        "--user", username,
        "--csv",
        "--page-size", "20"
    ]
    
    success, stdout, stderr = run_kaggle_command(cmd)
    
    if not success:
        print(f"    ⚠️ 获取kernel列表失败: {stderr}")
        return None
    
    # 解析CSV数据
    kernels = parse_csv_data(stdout)
    
    if not kernels:
        print(f"    ⚠️ 用户 '{username}' 在竞赛上没有kernel")
        return None
    
    # 取前N个kernel（假设已按最新排序）
    kernels = kernels[:max_kernels]
    
    print(f"    ✅ 找到 {len(kernels)} 个kernel")
    
    return kernels


def download_kernel(
    kernel_ref: str,
    output_path: str,
    download_metadata: bool = True
) -> bool:
    """
    下载单个kernel
    
    Args:
        kernel_ref: kernel引用，格式为 username/kernel-name
        output_path: 下载到的目录
        download_metadata: 是否同时下载metadata
        
    Returns:
        是否下载成功
    """
    # 创建输出目录
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    # 构建kaggle命令
    cmd = [
        "kaggle", "kernels", "pull",
        kernel_ref,
        "-p", output_path
    ]
    
    if download_metadata:
        cmd.append("-m")
    
    success, stdout, stderr = run_kaggle_command(cmd, retry_count=2)
    
    if success:
        print(f"      ✅ 下载kernel: {kernel_ref}")
        return True
    else:
        print(f"      ❌ 下载kernel失败: {kernel_ref}")
        return False


def download_competition_kernels(
    competition_slug: str,
    competition_name: str,
    usernames: List[str],
    keyword: str = "",
    max_kernels_per_user: int = 5,
    base_output_dir: str = "output",
    save_csv_flag: bool = True
) -> bool:
    """
    下载竞赛中多个用户的kernel
    
    Args:
        competition_slug: 竞赛slug
        competition_name: 竞赛名称（用于目录命名）
        usernames: 用户名列表
        keyword: 搜索关键词（用于目录结构）
        max_kernels_per_user: 每个用户最多下载多少个kernel
        base_output_dir: 基础输出目录
        save_csv_flag: 是否保存kernel列表为CSV
        
    Returns:
        是否全部下载成功
    """
    print_section(f"下载kernel: {competition_slug}")
    
    all_success = True
    
    for user_idx, username in enumerate(usernames, 1):
        print(f"\n  [{user_idx}/{len(usernames)}] 处理用户: {username}")
        
        # 获取该用户的kernel列表
        kernels = get_user_kernels(
            competition_slug,
            username,
            max_kernels=max_kernels_per_user,
            save_csv_flag=False  # 单个用户的kernel列表暂不保存
        )
        
        if not kernels:
            continue
        
        # 创建下载目录
        if keyword:
            download_path = create_directory_structure(base_output_dir, keyword, competition_name)
        else:
            download_path = create_directory_structure(base_output_dir, "default", competition_name)
        
        # 下载kernel
        for kernel_idx, kernel in enumerate(kernels, 1):
            kernel_ref = kernel.get("ref", "")
            kernel_name = kernel.get("title", "unknown")
            
            # 为每个kernel创建独立目录
            kernel_dir = os.path.join(download_path, f"{kernel_idx}_{kernel_name}")
            
            print(f"    [{kernel_idx}/{len(kernels)}] {kernel_name}")
            
            success = download_kernel(
                kernel_ref,
                kernel_dir,
                download_metadata=True
            )
            
            if not success:
                all_success = False
        
        # 保存该用户的kernel列表
        if save_csv_flag and kernels:
            csv_dir = os.path.join(base_output_dir, "csv_results")
            safe_comp = "".join(c if c.isalnum() or c in " -_" else "_" for c in competition_slug)
            filename = f"kernels_{safe_comp}_{username}.csv"
            save_csv(kernels, filename, csv_dir)
    
    return all_success
