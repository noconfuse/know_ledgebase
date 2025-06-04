#!/usr/bin/env python3
"""
PostgreSQL向量存储迁移测试脚本
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import List

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from config import settings
from postgres_vector_store import create_postgres_vector_store_builder
from services.rag_service import RAGService
from llama_index.core import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_postgres_connection():
    """测试PostgreSQL连接"""
    logger.info("Testing PostgreSQL connection...")
    
    try:
        import psycopg2
        
        # 测试连接
        conn = psycopg2.connect(
            host=settings.POSTGRES_HOST,
            port=settings.POSTGRES_PORT,
            database=settings.POSTGRES_DATABASE,
            user=settings.POSTGRES_USER,
            password=settings.POSTGRES_PASSWORD
        )
        
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()[0]
        logger.info(f"PostgreSQL connection successful: {version}")
        
        cursor.close()
        conn.close()
        
        return True
        
    except Exception as e:
        logger.error(f"PostgreSQL connection failed: {e}")
        return False

def test_postgres_vector_store():
    """测试PostgreSQL向量存储创建"""
    logger.info("Testing PostgreSQL vector store creation...")
    
    try:
        # 创建测试向量存储构建器
        builder = create_postgres_vector_store_builder(
            table_name="test_vector_store"
        )
        
        # 获取统计信息
        stats = builder.get_stats()
        logger.info(f"Vector store stats: {stats}")
        
        # 创建向量存储
        vector_store = builder.create_vector_store()
        logger.info("PostgreSQL vector store created successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"PostgreSQL vector store creation failed: {e}")
        return False

def test_embedding_model():
    """测试嵌入模型"""
    logger.info("Testing embedding model...")
    
    try:
        # 初始化嵌入模型
        embed_model = HuggingFaceEmbedding(
            model_name=settings.EMBED_MODEL_PATH,
            device=settings.GPU_DEVICE if settings.USE_GPU else "cpu",
            trust_remote_code=True
        )
        
        # 测试嵌入
        test_text = "这是一个测试文本"
        embedding = embed_model.get_text_embedding(test_text)
        
        logger.info(f"Embedding dimension: {len(embedding)}")
        logger.info("Embedding model test successful")
        
        return embed_model
        
    except Exception as e:
        logger.error(f"Embedding model test failed: {e}")
        return None

def test_document_indexing():
    """测试文档索引创建"""
    logger.info("Testing document indexing with PostgreSQL...")
    
    try:
        # 测试嵌入模型
        embed_model = test_embedding_model()
        if embed_model is None:
            return False
        
        # 创建测试文档
        test_documents = [
            Document(text="这是第一个测试文档，包含一些中文内容。", metadata={"source": "test1.txt"}),
            Document(text="This is the second test document with English content.", metadata={"source": "test2.txt"}),
            Document(text="这是第三个测试文档，用于验证向量存储功能。", metadata={"source": "test3.txt"})
        ]
        
        # 创建PostgreSQL向量存储构建器
        builder = create_postgres_vector_store_builder(
            table_name="test_migration_index"
        )
        
        # 删除现有表（如果存在）
        builder.delete_index()
        
        # 从文档创建索引
        from llama_index.core.node_parser import SentenceSplitter
        
        # 分割文档为节点
        splitter = SentenceSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )
        
        nodes = splitter.get_nodes_from_documents(test_documents)
        logger.info(f"Created {len(nodes)} nodes from {len(test_documents)} documents")
        
        # 创建索引
        index = builder.create_index_from_nodes(nodes, embed_model)
        logger.info("Document indexing successful")
        
        # 获取统计信息
        stats = builder.get_stats()
        logger.info(f"Index stats: {stats}")
        
        return True
        
    except Exception as e:
        logger.error(f"Document indexing failed: {e}")
        return False

async def test_rag_service():
    """测试RAG服务"""
    logger.info("Testing RAG service with PostgreSQL...")
    
    try:
        # 创建RAG服务
        rag_service = RAGService()
        
        # 测试加载索引
        index_id = "test_migration_index"
        success = await rag_service.load_index(index_id)
        
        if not success:
            logger.error(f"Failed to load index: {index_id}")
            return False
        
        # 测试检索
        query = "测试文档"
        results = await rag_service.hybrid_retrieve(index_id, query, top_k=3)
        
        logger.info(f"Retrieved {len(results)} results for query: '{query}'")
        for i, result in enumerate(results):
            logger.info(f"Result {i+1}: {result.node.text[:100]}... (score: {result.score})")
        
        logger.info("RAG service test successful")
        return True
        
    except Exception as e:
        logger.error(f"RAG service test failed: {e}")
        return False

def cleanup_test_data():
    """清理测试数据"""
    logger.info("Cleaning up test data...")
    
    try:
        # 删除测试表
        builder = create_postgres_vector_store_builder(
            table_name="test_vector_store"
        )
        builder.delete_index()
        
        builder = create_postgres_vector_store_builder(
            table_name="test_migration_index"
        )
        builder.delete_index()
        
        logger.info("Test data cleanup successful")
        
    except Exception as e:
        logger.error(f"Test data cleanup failed: {e}")

async def main():
    """主测试函数"""
    logger.info("Starting PostgreSQL migration tests...")
    
    # 显示当前配置
    logger.info(f"Vector store type: {settings.VECTOR_STORE_TYPE}")
    logger.info(f"PostgreSQL host: {settings.POSTGRES_HOST}")
    logger.info(f"PostgreSQL database: {settings.POSTGRES_DATABASE}")
    
    tests = [
        ("PostgreSQL Connection", test_postgres_connection),
        ("PostgreSQL Vector Store", test_postgres_vector_store),
        ("Document Indexing", test_document_indexing),
        ("RAG Service", test_rag_service)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running test: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            
            if result:
                logger.info(f"✅ {test_name} PASSED")
                passed += 1
            else:
                logger.error(f"❌ {test_name} FAILED")
                
        except Exception as e:
            logger.error(f"❌ {test_name} FAILED with exception: {e}")
    
    # 清理测试数据
    cleanup_test_data()
    
    # 显示结果
    logger.info(f"\n{'='*50}")
    logger.info(f"Test Results: {passed}/{total} tests passed")
    logger.info(f"Success rate: {passed/total*100:.1f}%")
    logger.info(f"{'='*50}")
    
    if passed == total:
        logger.info("🎉 All tests passed! PostgreSQL migration is ready.")
    else:
        logger.warning("⚠️  Some tests failed. Please check the logs and fix issues.")
    
    return passed == total

if __name__ == "__main__":
    asyncio.run(main())