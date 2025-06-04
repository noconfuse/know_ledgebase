#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的向量数据库构建器测试
"""

import asyncio
import logging
from pathlib import Path

from services.vector_store_builder import VectorStoreBuilder

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_document_collection():
    """测试文档收集功能"""
    
    # 测试目录
    test_directory = "/home/ubuntu/workspace/know_ledgebase/projects/ship_check/documents/国内航行海船法定检验技术规则（2020）/0总则"
    
    if not Path(test_directory).exists():
        logger.error(f"Test directory does not exist: {test_directory}")
        return False
    
    try:
        # 创建构建器实例
        builder = VectorStoreBuilder()
        
        # 创建模拟任务
        from services.vector_store_builder import VectorStoreTask
        task = VectorStoreTask(
            task_id="test_task",
            directory_path=test_directory,
            config={"chunk_size": 512, "chunk_overlap": 50}
        )
        
        # 测试文档收集
        logger.info("Testing document collection...")
        documents = builder._collect_documents(task)
        
        logger.info(f"Found {len(documents)} documents")
        
        for i, doc in enumerate(documents[:3]):  # 只显示前3个文档
            logger.info(f"Document {i+1}:")
            logger.info(f"  File: {doc.metadata.get('file_name', 'Unknown')}")
            logger.info(f"  Type: {doc.metadata.get('file_type', 'Unknown')}")
            logger.info(f"  Size: {doc.metadata.get('file_size', 0)} bytes")
            logger.info(f"  Content length: {len(doc.text)} characters")
            logger.info(f"  Has metadata: {doc.metadata.get('metadata_source', 'None')}")
            if 'total_pages' in doc.metadata:
                logger.info(f"  Total pages: {doc.metadata['total_pages']}")
            if 'total_text_blocks' in doc.metadata:
                logger.info(f"  Text blocks: {doc.metadata['total_text_blocks']}")
            logger.info("")
        
        return len(documents) > 0
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_node_parsers():
    """测试节点解析器"""
    
    try:
        builder = VectorStoreBuilder()
        
        # 测试不同文件类型的解析器
        config = {"chunk_size": 512, "chunk_overlap": 50}
        
        logger.info("Testing node parsers...")
        
        # 测试Markdown解析器
        md_parsers = builder._get_node_parser_for_file_type(".md", config)
        logger.info(f"Markdown parsers: {type(md_parsers)} - {len(md_parsers) if isinstance(md_parsers, list) else 1} parser(s)")
        
        # 测试HTML解析器
        html_parsers = builder._get_node_parser_for_file_type(".html", config)
        logger.info(f"HTML parsers: {type(html_parsers)} - {len(html_parsers) if isinstance(html_parsers, list) else 1} parser(s)")
        
        # 测试文本解析器
        txt_parser = builder._get_node_parser_for_file_type(".txt", config)
        logger.info(f"Text parser: {type(txt_parser)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Node parser test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_metadata_loading():
    """测试元数据加载"""
    
    try:
        builder = VectorStoreBuilder()
        
        # 测试元数据加载
        test_file = Path("/home/ubuntu/workspace/know_ledgebase/projects/ship_check/documents/国内航行海船法定检验技术规则（2020）/0总则/0总则.md")
        
        if test_file.exists():
            logger.info(f"Testing metadata loading for: {test_file}")
            metadata = builder._load_metadata(test_file)
            
            logger.info(f"Loaded metadata: {metadata}")
            return True
        else:
            logger.warning(f"Test file not found: {test_file}")
            return False
        
    except Exception as e:
        logger.error(f"Metadata loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    logger.info("Starting simplified vector store builder tests...")
    
    # 测试1：文档收集
    logger.info("\n=== Test 1: Document Collection ===")
    test1_result = test_document_collection()
    
    # 测试2：节点解析器
    logger.info("\n=== Test 2: Node Parsers ===")
    test2_result = test_node_parsers()
    
    # 测试3：元数据加载
    logger.info("\n=== Test 3: Metadata Loading ===")
    test3_result = test_metadata_loading()
    
    # 总结
    logger.info("\n=== Test Results ===")
    logger.info(f"Document Collection: {'PASS' if test1_result else 'FAIL'}")
    logger.info(f"Node Parsers: {'PASS' if test2_result else 'FAIL'}")
    logger.info(f"Metadata Loading: {'PASS' if test3_result else 'FAIL'}")
    
    all_passed = test1_result and test2_result and test3_result
    logger.info(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    
    return all_passed

if __name__ == "__main__":
    main()