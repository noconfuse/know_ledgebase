#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试向量数据库构建器的新功能
- 从knowledge/outputs/指定目录/读取解析后的文件
- 处理JSON元数据
- 使用不同文件类型的专用切片器
"""

import asyncio
import time
import logging
from pathlib import Path

from services.vector_store_builder import vector_store_builder

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_vector_builder():
    """测试向量数据库构建器"""
    
    # 测试目录 - 使用现有的解析后文件目录
    test_directory = "/home/ubuntu/workspace/knowledge/outputs/79eefb34-924f-4159-afd6-7952227e5d26"
    
    if not Path(test_directory).exists():
        logger.error(f"Test directory does not exist: {test_directory}")
        return
    
    logger.info(f"Testing vector store builder with directory: {test_directory}")
    
    # 配置参数
    config = {
        "chunk_size": 512,
        "chunk_overlap": 50,
        "extract_keywords": True,
        "extract_summary": True,
        "generate_qa": False  # 关闭问答生成以加快测试
    }
    
    try:
        # 创建向量数据库构建任务
        task_id = await vector_store_builder.build_vector_store(
            directory_path=test_directory,
            config=config
        )
        
        logger.info(f"Created vector store build task: {task_id}")
        
        # 监控任务进度
        while True:
            task_status = vector_store_builder.get_task_status(task_id)
            if not task_status:
                logger.error("Task not found")
                break
            
            logger.info(f"Task status: {task_status['status']}, Progress: {task_status['progress']}%")
            
            if task_status['status'] == 'completed':
                logger.info("Task completed successfully!")
                logger.info(f"Result: {task_status['result']}")
                
                # 显示处理的文件
                logger.info(f"Processed files: {task_status['processed_files']}")
                break
            elif task_status['status'] == 'failed':
                logger.error(f"Task failed: {task_status['error']}")
                break
            
            await asyncio.sleep(2)
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

async def test_multiple_file_types():
    """测试多种文件类型的处理"""
    
    # 测试目录 - 包含多种文件类型
    test_directory = "/home/ubuntu/workspace/know_ledgebase/projects/ship_check/documents/海上浮动设施检验规则（2025）"
    
    if not Path(test_directory).exists():
        logger.error(f"Test directory does not exist: {test_directory}")
        return
    
    logger.info(f"Testing multiple file types with directory: {test_directory}")
    
    # 配置参数
    config = {
        "chunk_size": 1024,
        "chunk_overlap": 100,
        "extract_keywords": True,
        "extract_summary": False,
        "generate_qa": False
    }
    
    try:
        # 创建向量数据库构建任务
        task_id = await vector_store_builder.build_vector_store(
            directory_path=test_directory,
            config=config
        )
        
        logger.info(f"Created vector store build task: {task_id}")
        
        # 监控任务进度
        while True:
            task_status = vector_store_builder.get_task_status(task_id)
            if not task_status:
                logger.error("Task not found")
                break
            
            logger.info(f"Task status: {task_status['status']}, Progress: {task_status['progress']}%")
            
            if task_status['status'] == 'completed':
                logger.info("Task completed successfully!")
                result = task_status['result']
                logger.info(f"Document count: {result['document_count']}")
                logger.info(f"Node count: {result['node_count']}")
                logger.info(f"Vector dimension: {result['vector_dimension']}")
                logger.info(f"Stats: {result['stats']}")
                
                # 显示处理的文件
                logger.info(f"Processed files: {task_status['processed_files']}")
                break
            elif task_status['status'] == 'failed':
                logger.error(f"Task failed: {task_status['error']}")
                break
            
            await asyncio.sleep(3)
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

async def main():
    """主测试函数"""
    logger.info("Starting vector store builder tests...")
    
    # 测试1：基本功能测试
    logger.info("\n=== Test 1: Basic functionality ===")
    await test_vector_builder()
    
    # 等待一段时间
    await asyncio.sleep(5)
    
    # 测试2：多文件类型测试
    logger.info("\n=== Test 2: Multiple file types ===")
    await test_multiple_file_types()
    
    logger.info("\nAll tests completed!")

if __name__ == "__main__":
    asyncio.run(main())