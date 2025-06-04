#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试MinerU集成的脚本
"""

import asyncio
import json
import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from config import settings
from services.document_parser import document_parser
from utils.logging_config import setup_logging, get_logger

# 初始化日志
setup_logging()
logger = get_logger(__name__)

async def test_docling_parser():
    """测试Docling解析器"""
    print("\n=== 测试Docling解析器 ===")
    
    # 使用knowledge/uploads/目录下的PDF文件进行测试
    test_pdf = "/home/ubuntu/workspace/knowledge/uploads/消防法2021年修正版.pdf"
    
    if not os.path.exists(test_pdf):
        print(f"警告: 测试文件不存在: {test_pdf}")
        print("请创建一个测试PDF文件或修改路径")
        return None
    
    try:
        # 使用Docling解析器
        config = {
            "ocr_enabled": True,
            "save_to_file": True
        }
        
        task_id = await document_parser.parse_document(
            test_pdf,
            config,
            parser_type="docling"
        )
        
        print(f"Docling解析任务创建成功，任务ID: {task_id}")
        
        # 等待任务完成
        while True:
            status = document_parser.get_task_status(task_id)
            if not status:
                print("任务状态获取失败")
                break
                
            print(f"任务状态: {status['status']}, 进度: {status['progress']}%, 阶段: {status.get('current_stage', 'N/A')}")
            
            if status['status'] in ['completed', 'failed']:
                break
                
            await asyncio.sleep(2)
        
        if status['status'] == 'completed':
            result = document_parser.get_task_result(task_id)
            print(f"Docling解析完成，结果: {json.dumps(result, indent=2, ensure_ascii=False)[:500]}...")
            return task_id
        else:
            print(f"Docling解析失败: {status.get('error', 'Unknown error')}")
            return None
            
    except Exception as e:
        print(f"Docling解析测试失败: {e}")
        return None

async def test_mineru_parser():
    """测试MinerU解析器"""
    print("\n=== 测试MinerU解析器 ===")
    
    # 使用knowledge/uploads/目录下的PDF文件进行测试
    test_pdf = "/home/ubuntu/workspace/knowledge/uploads/消防法2021年修正版.pdf"
    
    if not os.path.exists(test_pdf):
        print(f"警告: 测试文件不存在: {test_pdf}")
        print("请创建一个测试PDF文件或修改路径")
        return None
    
    try:
        # 使用MinerU解析器
        config = {
            "save_to_file": True,
            "max_workers": 1
        }
        
        task_id = await document_parser.parse_document(
            test_pdf,
            config,
            parser_type="mineru"
        )
        
        print(f"MinerU解析任务创建成功，任务ID: {task_id}")
        
        # 等待任务完成
        while True:
            status = document_parser.get_task_status(task_id)
            if not status:
                print("任务状态获取失败")
                break
                
            print(f"任务状态: {status['status']}, 进度: {status['progress']}%, 阶段: {status.get('current_stage', 'N/A')}")
            
            if status['status'] in ['completed', 'failed']:
                break
                
            await asyncio.sleep(2)
        
        if status['status'] == 'completed':
            result = document_parser.get_task_result(task_id)
            print(f"MinerU解析完成，结果: {json.dumps(result, indent=2, ensure_ascii=False)[:500]}...")
            return task_id
        else:
            print(f"MinerU解析失败: {status.get('error', 'Unknown error')}")
            return None
            
    except Exception as e:
        print(f"MinerU解析测试失败: {e}")
        return None

async def test_parser_comparison():
    """比较两种解析器的结果"""
    print("\n=== 解析器比较测试 ===")
    
    # 测试Docling
    docling_task_id = await test_docling_parser()
    
    # 测试MinerU
    mineru_task_id = await test_mineru_parser()
    
    # 比较结果
    if docling_task_id and mineru_task_id:
        print("\n=== 结果比较 ===")
        
        docling_result = document_parser.get_task_result(docling_task_id)
        mineru_result = document_parser.get_task_result(mineru_task_id)
        
        if docling_result and mineru_result:
            print(f"Docling解析器:")
            print(f"  - 解析器类型: {docling_result.get('parser_type')}")
            print(f"  - 内容长度: {docling_result.get('content_length', 0)}")
            print(f"  - 包含表格: {docling_result.get('has_tables', False)}")
            print(f"  - 包含图片: {docling_result.get('has_images', False)}")
            
            print(f"\nMinerU解析器:")
            print(f"  - 解析器类型: {mineru_result.get('parser_type')}")
            print(f"  - 内容长度: {mineru_result.get('content_length', 0)}")
            print(f"  - 包含表格: {mineru_result.get('has_tables', False)}")
            print(f"  - 包含图片: {mineru_result.get('has_images', False)}")
            
            if 'statistics' in mineru_result:
                stats = mineru_result['statistics']
                print(f"  - 文本块数量: {stats.get('text_blocks', 0)}")
                print(f"  - 图片块数量: {stats.get('image_blocks', 0)}")
                print(f"  - 表格块数量: {stats.get('table_blocks', 0)}")
        
        # 清理任务
        document_parser.cleanup_task(docling_task_id)
        document_parser.cleanup_task(mineru_task_id)
        print("\n任务清理完成")

def test_configuration():
    """测试配置"""
    print("=== 配置测试 ===")
    print(f"默认解析器类型: {settings.DEFAULT_PARSER_TYPE}")
    print(f"MinerU输出目录: {settings.OUTPUT_DIR}")
    print(f"MinerU最大并发数: {settings.MINERU_MAX_WORKERS}")
    print(f"支持的文件格式: {settings.SUPPORTED_FORMATS}")
    print(f"最大文件大小: {settings.MAX_FILE_SIZE / (1024*1024):.1f}MB")
    
    # 检查目录是否存在
    mineru_dir = Path(settings.OUTPUT_DIR)
    if not mineru_dir.exists():
        print(f"创建MinerU输出目录: {mineru_dir}")
        mineru_dir.mkdir(parents=True, exist_ok=True)
    else:
        print(f"MinerU输出目录已存在: {mineru_dir}")

def create_test_file():
    """创建测试文件目录和示例文件"""
    test_dir = Path("/home/ubuntu/workspace/test_files")
    test_dir.mkdir(exist_ok=True)
    
    # 创建一个简单的文本文件作为测试
    test_txt = test_dir / "sample.txt"
    if not test_txt.exists():
        with open(test_txt, 'w', encoding='utf-8') as f:
            f.write("这是一个测试文档。\n\n")
            f.write("第一章 概述\n")
            f.write("本文档用于测试文档解析功能。\n\n")
            f.write("第二章 详细内容\n")
            f.write("这里包含了详细的内容描述。\n")
        print(f"创建测试文本文件: {test_txt}")
    
    print(f"测试文件目录: {test_dir}")
    print("注意: 要测试PDF解析，请将PDF文件放置在测试目录中并命名为 'sample.pdf'")

async def main():
    """主函数"""
    print("MinerU集成测试")
    print("=" * 50)
    
    # 测试配置
    test_configuration()
    
    # 创建测试文件
    create_test_file()
    
    # 运行解析器比较测试
    await test_parser_comparison()
    
    print("\n测试完成！")

if __name__ == "__main__":
    asyncio.run(main())