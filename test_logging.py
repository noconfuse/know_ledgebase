#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试日志系统功能
"""

import asyncio
import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from config import settings
from utils.logging_config import setup_logging, get_logger, log_progress
from services.document_parser import document_parser

def test_basic_logging():
    """测试基本日志功能"""
    print("=== 测试基本日志功能 ===")
    
    # 初始化日志系统
    setup_logging()
    logger = get_logger(__name__)
    
    # 测试不同级别的日志
    logger.debug("这是一条调试信息")
    logger.info("这是一条信息日志")
    logger.warning("这是一条警告信息")
    logger.error("这是一条错误信息")
    
    print("基本日志测试完成")

def test_progress_logging():
    """测试进度日志功能"""
    print("=== 测试进度日志功能 ===")
    
    # 测试进度日志
    for i in range(0, 101, 20):
        log_progress(f"test_task_{i}", i, f"处理进度 {i}%", {
            "current_step": i // 20 + 1,
            "total_steps": 6
        })
    
    print("进度日志测试完成")

def test_file_logging():
    """测试文件日志功能"""
    print("=== 测试文件日志功能 ===")
    
    logger = get_logger(__name__)
    
    # 检查日志目录是否存在
    log_dir = Path(settings.LOG_DIR)
    print(f"日志目录: {log_dir}")
    print(f"日志目录存在: {log_dir.exists()}")
    
    if log_dir.exists():
        log_files = list(log_dir.glob("*.log"))
        print(f"现有日志文件: {[f.name for f in log_files]}")
    
    # 写入一些测试日志
    logger.info("测试文件日志写入功能")
    logger.warning("测试警告日志写入")
    logger.error("测试错误日志写入")
    
    print("文件日志测试完成")

async def test_document_parser_logging():
    """测试文档解析器的日志功能"""
    print("=== 测试文档解析器日志功能 ===")
    
    # 创建一个测试文件
    test_file = Path(settings.UPLOAD_DIR) / "test_document.txt"
    test_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(test_file, "w", encoding="utf-8") as f:
        f.write("这是一个测试文档\n用于测试日志系统功能\n包含多行内容")
    
    try:
        # 启动解析任务
        task_id = await document_parser.parse_document(
            str(test_file),
            {"save_to_file": True}
        )
        
        print(f"创建解析任务: {task_id}")
        
        # 等待任务完成
        max_wait = 30  # 最多等待30秒
        wait_time = 0
        
        while wait_time < max_wait:
            task_status = document_parser.get_task_status(task_id)
            if task_status:
                status = task_status.get("status")
                progress = task_status.get("progress", 0)
                current_stage = task_status.get("current_stage", "")
                
                print(f"任务状态: {status}, 进度: {progress}%, 当前阶段: {current_stage}")
                
                if status in ["completed", "failed"]:
                    break
            
            await asyncio.sleep(1)
            wait_time += 1
        
        # 获取最终状态和日志
        final_status = document_parser.get_task_status(task_id)
        if final_status:
            print(f"最终状态: {final_status.get('status')}")
            processing_logs = final_status.get("processing_logs", [])
            print(f"处理日志数量: {len(processing_logs)}")
            
            # 显示最后几条日志
            if processing_logs:
                print("最后几条处理日志:")
                for log in processing_logs[-5:]:
                    print(f"  [{log.get('level')}] {log.get('message')}: {log.get('details', {})}")
    
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
    
    finally:
        # 清理测试文件
        if test_file.exists():
            test_file.unlink()
    
    print("文档解析器日志测试完成")

def main():
    """主测试函数"""
    print("开始测试日志系统...")
    print(f"当前工作目录: {os.getcwd()}")
    print(f"日志级别: {settings.LOG_LEVEL}")
    print(f"日志目录: {settings.LOG_DIR}")
    print(f"启用文件日志: {settings.ENABLE_FILE_LOGGING}")
    print(f"启用Docling日志: {settings.ENABLE_DOCLING_LOGGING}")
    print()
    
    # 运行测试
    test_basic_logging()
    print()
    
    test_progress_logging()
    print()
    
    test_file_logging()
    print()
    
    # 运行异步测试
    asyncio.run(test_document_parser_logging())
    print()
    
    print("所有测试完成！")
    print("请检查日志目录中的日志文件以验证功能是否正常工作。")

if __name__ == "__main__":
    main()