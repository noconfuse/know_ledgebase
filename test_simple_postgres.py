#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的PostgreSQL向量存储测试
"""

import asyncio
import logging
from typing import List

from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.postgres import PGVectorStore

from config import settings
from postgres_vector_store import create_postgres_vector_store_builder
from services.rag_service import RAGService

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
        
        # 连接参数
        conn_params = {
            'host': settings.POSTGRES_HOST,
            'port': settings.POSTGRES_PORT,
            'database': settings.POSTGRES_DATABASE,
            'user': settings.POSTGRES_USER,
            'password': settings.POSTGRES_PASSWORD
        }
        
        # 测试连接
        conn = psycopg2.connect(**conn_params)
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()[0]
        cursor.close()
        conn.close()
        
        logger.info(f"PostgreSQL connection successful: {version}")
        return True
        
    except Exception as e:
        logger.error(f"PostgreSQL connection failed: {e}")
        return False

def test_basic_vector_store():
    """测试基本向量存储功能"""
    logger.info("Testing basic vector store functionality...")
    
    try:
        # 创建嵌入模型
        embed_model = HuggingFaceEmbedding(
            model_name=settings.EMBED_MODEL_PATH,
            device=settings.GPU_DEVICE if settings.USE_GPU else "cpu",
            trust_remote_code=True
        )
        
        # 创建测试文档
        documents = [
            Document(text="这是一个关于人工智能的测试文档。", metadata={"source": "test1.txt"}),
            Document(text="机器学习是人工智能的重要分支。", metadata={"source": "test2.txt"})
        ]
        
        # 分割文档
        splitter = SentenceSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )
        nodes = splitter.get_nodes_from_documents(documents)
        logger.info(f"Created {len(nodes)} nodes from {len(documents)} documents")
        
        # 创建向量存储
        vector_store = PGVectorStore.from_params(
            database=settings.POSTGRES_DATABASE,
            host=settings.POSTGRES_HOST,
            password=settings.POSTGRES_PASSWORD,
            port=settings.POSTGRES_PORT,
            user=settings.POSTGRES_USER,
            table_name="simple_test_table",
            embed_dim=1024,  # BGE模型的维度
        )
        
        # 创建存储上下文
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # 创建索引
        index = VectorStoreIndex(
            nodes=nodes,
            storage_context=storage_context,
            embed_model=embed_model,
            show_progress=True
        )
        
        logger.info("Vector index created successfully")
        
        # 测试查询
        from llama_index.core.llms import MockLLM
        from llama_index.core import Settings
        
        # 使用Mock LLM避免上下文大小问题
        Settings.llm = MockLLM(max_tokens=256)
        
        query_engine = index.as_query_engine(
            similarity_top_k=1,
            response_mode="no_text"
        )
        
        # 简单的相似性搜索测试
        retriever = index.as_retriever(similarity_top_k=2)
        nodes = retriever.retrieve("人工智能")
        
        if len(nodes) > 0:
            logger.info(f"Retrieved {len(nodes)} relevant nodes")
            logger.info(f"First node content: {nodes[0].text[:100]}...")
        else:
            raise Exception("No nodes retrieved")
        
        logger.info("Basic vector store test passed")
        
        # 清理
        try:
            # 删除测试表
            import psycopg2
            conn = psycopg2.connect(
                host=settings.POSTGRES_HOST,
                port=settings.POSTGRES_PORT,
                database=settings.POSTGRES_DATABASE,
                user=settings.POSTGRES_USER,
                password=settings.POSTGRES_PASSWORD
            )
            cursor = conn.cursor()
            cursor.execute("DROP TABLE IF EXISTS data_simple_test_table;")
            conn.commit()
            cursor.close()
            conn.close()
            logger.info("Test table cleaned up")
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"Basic vector store test failed: {e}")
        return False

def main():
    """主测试函数"""
    logger.info("Starting simplified PostgreSQL tests...")
    
    tests = [
        ("PostgreSQL Connection", test_postgres_connection),
        ("Basic Vector Store", test_basic_vector_store),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running test: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            result = test_func()
            if result:
                logger.info(f"✅ {test_name} PASSED")
                passed += 1
            else:
                logger.error(f"❌ {test_name} FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name} FAILED with exception: {e}")
    
    # 显示结果
    logger.info(f"\n{'='*50}")
    logger.info(f"Test Results: {passed}/{total} tests passed")
    logger.info(f"Success rate: {passed/total*100:.1f}%")
    logger.info(f"{'='*50}")
    
    if passed == total:
        logger.info("🎉 All tests passed! Basic PostgreSQL functionality is working.")
    else:
        logger.warning("⚠️  Some tests failed. Please check the logs.")
    
    return passed == total

if __name__ == "__main__":
    main()