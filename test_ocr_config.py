#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试OCR语言配置功能
验证parse_document方法中的ocr_languages配置是否正确使用
"""

import asyncio
import json
import sys
import os
from pathlib import Path

# 添加项目路径
sys.path.append('/home/ubuntu/workspace/know_ledgebase')

from services.document_parser import DocumentParser
from config import settings
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_ocr_languages_config():
    """测试OCR语言配置"""
    print("=== 测试OCR语言配置功能 ===")
    
    # 使用现有的测试文档
    test_file = Path("/home/ubuntu/workspace/know_ledgebase/sample_test_document.txt")
    
    print(f"使用测试文件: {test_file}")
    print("注意：由于docling不支持txt文件，测试会失败，但我们可以验证配置是否正确传递")
    
    # 初始化文档解析器
    parser = DocumentParser()
    
    # 测试1: 使用默认配置
    print("\n--- 测试1: 使用默认配置 ---")
    try:
        task_id1 = await parser.parse_document(str(test_file))
        print(f"任务ID: {task_id1}")
        
        # 等待任务完成
        await asyncio.sleep(2)
        
        # 检查任务状态
        task1 = parser.get_task_status(task_id1)
        print(f"任务状态: {task1['status']}")
        if task1.get('processing_logs'):
            for log in task1['processing_logs'][-3:]:
                if 'converter_config' in log.get('details', {}):
                    print(f"转换器配置: {log['details']['converter_config']}")
                    
    except Exception as e:
        print(f"测试1失败: {e}")
    
    # 测试2: 使用自定义OCR语言配置
    print("\n--- 测试2: 使用自定义OCR语言配置 ---")
    custom_config = {
        "ocr_enabled": True,
        "ocr_languages": ["en"],  # 只使用英语
        "extract_tables": True,
        "extract_images": True
    }
    
    try:
        task_id2 = await parser.parse_document(str(test_file), custom_config)
        print(f"任务ID: {task_id2}")
        print(f"自定义配置: {custom_config}")
        
        # 等待任务完成
        await asyncio.sleep(2)
        
        # 检查任务状态
        task2 = parser.get_task_status(task_id2)
        print(f"任务状态: {task2['status']}")
        if task2.get('processing_logs'):
            for log in task2['processing_logs'][-3:]:
                if 'converter_config' in log.get('details', {}):
                    print(f"转换器配置: {log['details']['converter_config']}")
                    
    except Exception as e:
        print(f"测试2失败: {e}")
    
    # 测试3: 禁用OCR
    print("\n--- 测试3: 禁用OCR ---")
    no_ocr_config = {
        "ocr_enabled": False,
        "extract_tables": True,
        "extract_images": True
    }
    
    try:
        task_id3 = await parser.parse_document(str(test_file), no_ocr_config)
        print(f"任务ID: {task_id3}")
        print(f"禁用OCR配置: {no_ocr_config}")
        
        # 等待任务完成
        await asyncio.sleep(2)
        
        # 检查任务状态
        task3 = parser.get_task_status(task_id3)
        print(f"任务状态: {task3['status']}")
        if task3.get('processing_logs'):
            for log in task3['processing_logs'][-3:]:
                if 'converter_config' in log.get('details', {}):
                    print(f"转换器配置: {log['details']['converter_config']}")
                    
    except Exception as e:
        print(f"测试3失败: {e}")
    
    # 测试4: 使用多种语言
    print("\n--- 测试4: 使用多种语言配置 ---")
    multi_lang_config = {
        "ocr_enabled": True,
        "ocr_languages": ["ch_sim", "en", "ja"],  # 中文、英文、日文
        "extract_tables": True,
        "extract_images": True
    }
    
    try:
        task_id4 = await parser.parse_document(str(test_file), multi_lang_config)
        print(f"任务ID: {task_id4}")
        print(f"多语言配置: {multi_lang_config}")
        
        # 等待任务完成
        await asyncio.sleep(2)
        
        # 检查任务状态
        task4 = parser.get_task_status(task_id4)
        print(f"任务状态: {task4['status']}")
        if task4.get('processing_logs'):
            for log in task4['processing_logs'][-3:]:
                if 'converter_config' in log.get('details', {}):
                    print(f"转换器配置: {log['details']['converter_config']}")
                    
    except Exception as e:
        print(f"测试4失败: {e}")
    
    # 不需要清理，使用的是现有测试文件
    print(f"\n测试完成，使用的文件: {test_file}")
    
    print("\n=== OCR语言配置测试完成 ===")

if __name__ == "__main__":
    asyncio.run(test_ocr_languages_config())