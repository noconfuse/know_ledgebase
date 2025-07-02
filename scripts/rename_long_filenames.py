#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
处理已解析的outputs目录中的长文件名
将超长文件名重命名为截断后的文件名，保持与系统一致性
"""

import os
import sys
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.document_utils import truncate_filename
from config import settings

def get_outputs_directory() -> Path:
    """获取outputs目录路径"""
    return Path(settings.KNOWLEDGE_BASE_DIR) / "outputs"

def find_long_filenames(directory: Path, max_length: int = 60) -> List[Tuple[Path, str]]:
    """查找超长文件名的文件
    
    Returns:
        List[Tuple[Path, str]]: (原文件路径, 截断后的文件名) 的列表
    """
    long_files = []
    
    for root, dirs, files in os.walk(directory):
        root_path = Path(root)
        
        for filename in files:
            if len(filename) > max_length:
                original_path = root_path / filename
                truncated_name = truncate_filename(filename, max_length, preserve_extension=True)
                long_files.append((original_path, truncated_name))
                
    return long_files

def rename_files(file_list: List[Tuple[Path, str]], dry_run: bool = True, max_length: int = 60) -> Dict[str, int]:
    """重命名文件
    
    Args:
        file_list: 需要重命名的文件列表
        dry_run: 是否为试运行模式
        
    Returns:
        Dict: 统计信息
    """
    stats = {
        'total': len(file_list),
        'success': 0,
        'failed': 0,
        'skipped': 0
    }
    
    for original_path, new_filename in file_list:
        new_path = original_path.parent / new_filename
        
        # 检查目标文件是否已存在
        if new_path.exists() and new_path != original_path:
            print(f"跳过: 目标文件已存在 {new_path}")
            stats['skipped'] += 1
            continue
            
        if dry_run:
            print(f"[试运行] 将重命名: {original_path.name} -> {new_filename}")
            stats['success'] += 1
        else:
            try:
                original_path.rename(new_path)
                print(f"已重命名: {original_path.name} -> {new_filename}")
                stats['success'] += 1
            except Exception as e:
                print(f"重命名失败: {original_path} -> {new_path}, 错误: {e}")
                stats['failed'] += 1
                
    return stats

def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="处理outputs目录中的长文件名")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="试运行模式，只打印将要重命名的文件，不实际执行"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=60,
        help="文件名的最大长度，超过该长度将被截断"
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=str(get_outputs_directory()),
        help="要扫描的目录路径"
    )

    args = parser.parse_args()

    target_dir = Path(args.dir)
    if not target_dir.is_dir():
        print(f"错误: 目录不存在 {target_dir}")
        sys.exit(1)

    print(f"正在扫描目录: {target_dir}")
    print(f"文件名最大长度: {args.max_length}")

    long_files = find_long_filenames(target_dir, args.max_length)

    if not long_files:
        print("未找到需要重命名的超长文件名文件。")
        return

    print(f"\n找到 {len(long_files)} 个需要重命名的文件:")
    stats = rename_files(long_files, args.dry_run, args.max_length)

    print("\n--- 结果统计 ---")
    print(f"总计文件: {stats['total']}")
    print(f"成功处理: {stats['success']}")
    print(f"失败: {stats['failed']}")
    print(f"跳过: {stats['skipped']}")
    print("------------------")

if __name__ == "__main__":
    main()


    