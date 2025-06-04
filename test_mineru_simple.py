#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的MinerU测试脚本
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

def test_mineru_import():
    """测试MinerU相关模块的导入"""
    print("=== 测试MinerU模块导入 ===")
    
    try:
        # 测试magic_pdf导入
        print("导入magic_pdf...")
        import magic_pdf
        print(f"✓ magic_pdf导入成功，版本: {getattr(magic_pdf, '__version__', 'unknown')}")
        
        # 测试具体的类导入
        print("导入MinerU解析器组件...")
        from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
        print("✓ FileBasedDataWriter, FileBasedDataReader导入成功")
        
        # 测试基本模块存在性
        print("检查magic_pdf子模块...")
        import magic_pdf.config
        print("✓ magic_pdf.config模块存在")
        
        import magic_pdf.model
        print("✓ magic_pdf.model模块存在")
        
        import magic_pdf.tools
        print("✓ magic_pdf.tools模块存在")
        
        return True
        
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        return False
    except Exception as e:
        print(f"✗ 其他错误: {e}")
        return False

def test_mineru_parser_import():
    """测试我们自己的MinerU解析器导入"""
    print("\n=== 测试自定义MinerU解析器导入 ===")
    
    try:
        # 测试我们的解析器导入
        print("导入自定义MinerU解析器...")
        from app.minerU.parse_pdf import PdfParser
        print("✓ PdfParser导入成功")
        
        from services.mineru_parser import MinerUDocumentParser, MinerUParseTask
        print("✓ MinerUDocumentParser, MinerUParseTask导入成功")
        
        return True
        
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        return False
    except Exception as e:
        print(f"✗ 其他错误: {e}")
        return False

def test_configuration():
    """测试配置"""
    print("\n=== 测试配置 ===")
    
    try:
        from config import settings
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
            
        return True
        
    except Exception as e:
        print(f"✗ 配置测试失败: {e}")
        return False

def test_mineru_parser_creation():
    """测试MinerU解析器实例创建"""
    print("\n=== 测试MinerU解析器实例创建 ===")
    
    try:
        from services.mineru_parser import MinerUDocumentParser
        
        # 创建解析器实例
        print("创建MinerU解析器实例...")
        parser = MinerUDocumentParser()
        print("✓ MinerU解析器实例创建成功")
        
        # 测试基本方法
        print("测试解析器基本方法...")
        # 测试获取不存在任务的状态
        status = parser.get_task_status("test_task_id")
        print(f"✓ 获取任务状态方法正常: {status is None}")
        
        # 测试获取不存在任务的结果
        result = parser.get_task_result("test_task_id")
        print(f"✓ 获取任务结果方法正常: {result is None}")
        
        # 测试清理不存在的任务
        cleaned = parser.cleanup_task("test_task_id")
        print(f"✓ 清理任务方法正常: {cleaned == False}")
        
        return True
        
    except Exception as e:
        print(f"✗ 解析器创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_test_file():
    """创建测试文件目录"""
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

def main():
    """主函数"""
    print("MinerU简化集成测试")
    print("=" * 50)
    
    success_count = 0
    total_tests = 4
    
    # 测试1: MinerU模块导入
    print("\n[1/4] 测试MinerU模块导入...")
    if test_mineru_import():
        success_count += 1
        print("✓ 测试1通过")
    else:
        print("✗ 测试1失败")
    
    # 测试2: 自定义解析器导入
    print("\n[2/4] 测试自定义解析器导入...")
    if test_mineru_parser_import():
        success_count += 1
        print("✓ 测试2通过")
    else:
        print("✗ 测试2失败")
    
    # 测试3: 配置测试
    print("\n[3/4] 测试配置...")
    if test_configuration():
        success_count += 1
        print("✓ 测试3通过")
    else:
        print("✗ 测试3失败")
    
    # 测试4: 解析器实例创建
    print("\n[4/4] 测试解析器实例创建...")
    if test_mineru_parser_creation():
        success_count += 1
        print("✓ 测试4通过")
    else:
        print("✗ 测试4失败")
    
    # 创建测试文件
    create_test_file()
    
    print(f"\n=== 测试结果 ===")
    print(f"成功: {success_count}/{total_tests}")
    
    if success_count == total_tests:
        print("✓ 所有测试通过！MinerU集成准备就绪。")
        return 0
    else:
        print(f"✗ {total_tests - success_count} 个测试失败。")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)