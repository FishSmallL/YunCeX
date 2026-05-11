"""
工具函数模块，包含公用的命令执行、数据处理等功能
"""

import subprocess
import csv
import sys
import os
from io import StringIO
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json
import shlex


def load_config(config_file="config.json") -> Dict:
    """
    加载配置文件
    
    Args:
        config_file: 配置文件路径
        
    Returns:
        配置字典
    """
    if not os.path.exists(config_file):
        print(f"❌ 配置文件不存在: {config_file}")
        sys.exit(1)
    
    try:
        with open(config_file, "r", encoding="utf-8") as f:
            config = json.load(f)
        print(f"[OK] 加载配置文件成功: {config_file}")
        return config
    except json.JSONDecodeError as e:
        print(f"❌ 配置文件格式错误: {e}")
        sys.exit(1)


def run_kaggle_command(cmd: List[str], retry_count: int = 3) -> Tuple[bool, str, str]:
    """
    执行kaggle命令，支持重试
    
    Args:
        cmd: 命令列表，如 ["kaggle", "competitions", "list", "--search", "titanic"]
        retry_count: 重试次数，默认3次
        
    Returns:
        (成功标志, 标准输出, 错误输出)
    
    注意: 请确保在运行此程序前已激活正确的conda环境
    """
    # 直接用列表形式调用 subprocess，避免 shell 对含空格参数的分割
    for attempt in range(1, retry_count + 1):
        try:
            # 打印转义后的命令以便阅读
            printable = " ".join(shlex.quote(a) for a in cmd)
            print(f"  运行命令 (尝试 {attempt}/{retry_count}): {printable}")
            result = subprocess.run(
                cmd,
                shell=False,
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode == 0:
                print(f"  ✅ 命令执行成功")
                return True, result.stdout, result.stderr
            else:
                # 有时 kaggle CLI 会在输出中返回有用的CSV内容但退出码非0（例如返回分页token）
                combined_out = (result.stdout or "") + "\n" + (result.stderr or "")
                # 如果输出看起来像CSV（包含逗号和常见CSV头），我们也当作成功处理
                if any(h in combined_out for h in ["ref,", "title,", "teamId,", "teamName,", "submissionDate,"]):
                    print(f"  ⚠️ 注意：命令退出码非0，但发现CSV样式输出，视为成功")
                    return True, result.stdout, result.stderr

                # 否则打印错误信息
                error_msg = result.stderr if result.stderr else result.stdout
                print(f"  ⚠️ 命令执行失败 (尝试 {attempt}/{retry_count}): {error_msg[:300]}")

                # 如果是最后一次尝试，返回失败
                if attempt == retry_count:
                    return False, result.stdout, result.stderr

        except subprocess.TimeoutExpired:
            print(f"  ⏱️ 命令超时 (尝试 {attempt}/{retry_count})")
            if attempt == retry_count:
                return False, "", "Command timeout"
        except Exception as e:
            print(f"  ❌ 执行错误 (尝试 {attempt}/{retry_count}): {str(e)}")
            if attempt == retry_count:
                return False, "", str(e)

    return False, "", "All retries failed"


def parse_csv_data(csv_text: str) -> Optional[List[Dict]]:
    """
    解析CSV格式的数据
    
    Args:
        csv_text: CSV格式的文本
        
    Returns:
        字典列表，如果解析失败返回None
    """
    if not csv_text or len(csv_text.strip().splitlines()) == 0:
        return None

    # 跳过输出中可能存在的非CSV前置行（例如: Next Page Token = ...）
    lines = csv_text.splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        # 认为第一行包含逗号并且不是key=value样式的行为表头
        if "," in line and "=" not in line:
            header_idx = i
            break

    if header_idx is None:
        return None

    try:
        csv_file = StringIO("\n".join(lines[header_idx:]))
        reader = csv.DictReader(csv_file)
        rows = list(reader)
        return rows if rows else None
    except csv.Error as e:
        print(f"  ❌ CSV解析错误: {e}")
        return None


def save_csv(data: List[Dict], filename: str, output_dir: str = "output") -> bool:
    """
    保存数据为CSV文件
    
    Args:
        data: 字典列表
        filename: 输出文件名
        output_dir: 输出目录
        
    Returns:
        是否保存成功
    """
    if not data:
        print(f"  ⚠️ 没有数据可保存")
        return False
    
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    filepath = os.path.join(output_dir, filename)
    
    try:
        keys = list(data[0].keys())
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(data)
        print(f"  ✅ CSV已保存: {filepath} ({len(data)} 行)")
        return True
    except Exception as e:
        print(f"  ❌ 保存CSV失败: {e}")
        return False


def create_directory_structure(base_dir: str, keyword: str, competition_name: str) -> str:
    """
    创建目录结构
    
    Args:
        base_dir: 基础目录
        keyword: 关键词
        competition_name: 竞赛名称
        
    Returns:
        创建的路径
    """
    # 处理特殊字符，使其成为有效的目录名
    keyword_dir = "".join(c if c.isalnum() or c in " -_" else "_" for c in keyword)
    comp_dir = "".join(c if c.isalnum() or c in " -_" else "_" for c in competition_name)
    
    path = os.path.join(base_dir, keyword_dir, comp_dir, "kernels")
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def print_section(title: str) -> None:
    """打印分段标题"""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def extract_competition_slug(ref_field: str) -> str:
    """
    从ref字段提取竞赛slug
    
    如果ref是完整URL，提取最后的slug部分；
    如果ref已经是slug格式，直接返回
    
    Args:
        ref_field: ref字段值，可能是URL或slug
        
    Returns:
        竞赛slug
    """
    # 如果包含斜杠，说明是URL格式，取最后一个部分
    if "/" in ref_field:
        return ref_field.split("/")[-1]
    # 否则直接返回
    return ref_field


def print_result(data: List[Dict], limit: int = 10) -> None:
    """
    打印结果数据
    
    Args:
        data: 数据列表
        limit: 最多打印多少条
    """
    if not data:
        print("  ⚠️ 没有数据")
        return
    
    print(f"\n  📊 共获取 {len(data)} 条数据，显示前 {min(limit, len(data))} 条:\n")
    
    # 获取所有字段
    keys = list(data[0].keys())
    
    # 简化的表格显示
    for idx, row in enumerate(data[:limit], 1):
        print(f"  [{idx}] {row}")
    
    if len(data) > limit:
        print(f"  ... 还有 {len(data) - limit} 条数据未显示")
