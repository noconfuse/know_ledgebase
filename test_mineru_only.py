#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
专门测试MinerU解析器的脚本
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

async def test_mineru_parser():
    """测试MinerU解析器"""
    print("\n=== 测试MinerU解析器 ===")
    
    # 使用knowledge/uploads/目录下的PDF文件进行测试
    test_pdf = "/home/ubuntu/workspace/knowledge/uploads/消防法2021年修正版.pdf"
    
    if not os.path.exists(test_pdf):
        print(f"错误: 测试文件不存在: {test_pdf}")
        return None
    
    print(f"测试文件: {test_pdf}")
    print(f"文件大小: {os.path.getsize(test_pdf) / (1024*1024):.2f} MB")
    
    try:
        # 使用MinerU解析器
        config = {
            "save_to_file": True,
            "max_workers": 1
        }
        
        print("开始创建MinerU解析任务...")
        task_id = await document_parser.parse_document(
            test_pdf,
            config,
            parser_type="mineru"
        )
        
        print(f"MinerU解析任务创建成功，任务ID: {task_id}")
        
        # 等待任务完成
        print("等待解析完成...")
        while True:
            status = document_parser.get_task_status(task_id)
            if not status:
                print("任务状态获取失败")
                break
                
            print(f"任务状态: {status['status']}, 进度: {status['progress']}%, 阶段: {status.get('current_stage', 'N/A')}")
            
            if status['status'] in ['completed', 'failed']:
                break
                
            await asyncio.sleep(3)
        
        if status['status'] == 'completed':
            result = document_parser.get_task_result(task_id)
            print("\n=== MinerU解析结果 ===")
            print(f"解析器类型: {result.get('parser_type')}")
            print(f"文档标题: {result.get('title', 'N/A')}")
            print(f"内容长度: {result.get('content_length', 0)} 字符")
            print(f"包含表格: {result.get('has_tables', False)}")
            print(f"包含图片: {result.get('has_images', False)}")
            
            if 'statistics' in result:
                stats = result['statistics']
                print(f"\n=== 统计信息 ===")
                print(f"文本块数量: {stats.get('text_blocks', 0)}")
                print(f"图片块数量: {stats.get('image_blocks', 0)}")
                print(f"表格块数量: {stats.get('table_blocks', 0)}")
                print(f"总块数量: {stats.get('total_blocks', 0)}")
            
            if 'metadata' in result:
                metadata = result['metadata']
                print(f"\n=== 元数据信息 ===")
                print(f"输出目录: {metadata.get('output_directory', 'N/A')}")
                if 'output_files' in metadata:
                    files = metadata['output_files']
                    print(f"Markdown文件: {files.get('markdown', 'N/A')}")
                    print(f"JSON文件: {files.get('json', 'N/A')}")
            
            if 'structure' in result:
                structure = result['structure']
                print(f"\n=== 文档结构 ===")
                headings = structure.get('headings', [])
                print(f"标题数量: {len(headings)}")
                if headings:
                    print("前5个标题:")
                    for i, heading in enumerate(headings[:5]):
                        print(f"  {i+1}. 级别{heading.get('level', 'N/A')}: {heading.get('text', 'N/A')}")
            
            # 显示部分内容
            content = result.get('content', '')
            if content:
                print(f"\n=== 内容预览 (前500字符) ===")
                print(content[:500])
                if len(content) > 500:
                    print("...")
            
            # 清理任务
            document_parser.cleanup_task(task_id)
            print(f"\n任务 {task_id} 清理完成")
            
            return task_id
        else:
            error_msg = status.get('error', 'Unknown error')
            print(f"MinerU解析失败: {error_msg}")
            return None
            
    except Exception as e:
        print(f"MinerU解析测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_configuration():
    """测试配置"""
    print("=== MinerU配置信息 ===")
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

async def main():
    """主函数"""
    print("MinerU解析器专项测试")
    print("=" * 50)
    
    # 测试配置
    test_configuration()
    
    # 测试MinerU解析器
    result = await test_mineru_parser()
    
    if result:
        print("\n✅ MinerU解析测试成功完成！")
    else:
        print("\n❌ MinerU解析测试失败！")
    
    print("\n测试完成！")

if __name__ == "__main__":
    asyncio.run(main())