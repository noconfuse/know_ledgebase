import os
from typing import Optional, List, Any, Dict
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.schema import BaseNode
from sqlalchemy import make_url
from config import settings
import logging

logger = logging.getLogger(__name__)

class PostgresVectorStoreBuilder:
    """PostgreSQL向量存储构建器"""
    
    def __init__(self, 
                 host: str = "localhost",
                 port: int = 5432,
                 database: str = "knowledge_base",
                 user: str = "postgres",
                 password: str = "postgres",
                 table_name: str = "vector_store",
                 embed_dim: int = 1024):
        """
        初始化PostgreSQL向量存储构建器
        
        Args:
            host: 数据库主机
            port: 数据库端口
            database: 数据库名称
            user: 用户名
            password: 密码
            table_name: 表名
            embed_dim: 向量维度
        """
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.table_name = table_name
        self.embed_dim = embed_dim
        
        # 构建连接URL
        self.connection_string = f"postgresql://{user}:{password}@{host}:{port}/{database}"
        
    def create_vector_store(self) -> PGVectorStore:
        """
        创建PostgreSQL向量存储
        
        Returns:
            PGVectorStore实例
        """
        try:
            # HNSW索引配置
            hnsw_kwargs = {
                "hnsw_m": 16,
                "hnsw_ef_construction": 64,
                "hnsw_ef_search": 40
            }
            
            vector_store = PGVectorStore.from_params(
                database=self.database,
                host=self.host,
                password=self.password,
                port=self.port,
                user=self.user,
                table_name=self.table_name,
                embed_dim=self.embed_dim,
                hnsw_kwargs=hnsw_kwargs
            )
            
            logger.info(f"成功创建PostgreSQL向量存储，表名: {self.table_name}")
            return vector_store
            
        except Exception as e:
            logger.error(f"创建PostgreSQL向量存储失败: {str(e)}")
            raise
    
    def create_index_from_nodes(self, nodes: List[BaseNode], embed_model) -> VectorStoreIndex:
        """
        从节点创建向量索引
        
        Args:
            nodes: 文档节点列表
            embed_model: 嵌入模型
            
        Returns:
            VectorStoreIndex实例
        """
        try:
            # 创建向量存储
            vector_store = self.create_vector_store()
            
            # 创建存储上下文
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            # 创建向量索引
            index = VectorStoreIndex(
                nodes=nodes,
                storage_context=storage_context,
                embed_model=embed_model,
                show_progress=True
            )
            
            logger.info(f"成功创建向量索引，包含 {len(nodes)} 个节点")
            return index
            
        except Exception as e:
            logger.error(f"创建向量索引失败: {str(e)}")
            raise
    
    def load_index(self, embed_model) -> Optional[VectorStoreIndex]:
        """
        加载现有的向量索引
        
        Args:
            embed_model: 嵌入模型
            
        Returns:
            VectorStoreIndex实例或None
        """
        try:
            # 创建向量存储
            vector_store = self.create_vector_store()
            
            # 创建存储上下文
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            # 加载向量索引
            index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
                embed_model=embed_model
            )
            
            logger.info(f"成功加载PostgreSQL向量索引")
            return index
            
        except Exception as e:
            logger.error(f"加载PostgreSQL向量索引失败: {str(e)}")
            return None
    
    def delete_index(self) -> bool:
        """
        删除向量索引（删除表）
        
        Returns:
            是否成功删除
        """
        try:
            import psycopg2
            
            # 连接数据库
            conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password
            )
            
            cursor = conn.cursor()
            
            # 删除表
            cursor.execute(f"DROP TABLE IF EXISTS {self.table_name} CASCADE;")
            conn.commit()
            
            cursor.close()
            conn.close()
            
            logger.info(f"成功删除向量索引表: {self.table_name}")
            return True
            
        except Exception as e:
            logger.error(f"删除向量索引失败: {str(e)}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取向量存储统计信息
        
        Returns:
            统计信息字典
        """
        try:
            import psycopg2
            
            # 连接数据库
            conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password
            )
            
            cursor = conn.cursor()
            
            # 检查表是否存在
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = %s
                );
            """, (self.table_name,))
            
            table_exists = cursor.fetchone()[0]
            
            if not table_exists:
                return {"table_exists": False, "row_count": 0}
            
            # 获取行数
            cursor.execute(f"SELECT COUNT(*) FROM {self.table_name};")
            row_count = cursor.fetchone()[0]
            
            cursor.close()
            conn.close()
            
            return {
                "table_exists": True,
                "row_count": row_count,
                "table_name": self.table_name,
                "embed_dim": self.embed_dim
            }
            
        except Exception as e:
            logger.error(f"获取统计信息失败: {str(e)}")
            return {"error": str(e)}
    
    @staticmethod
    def get_vector_store_tables(host: str, port: int, database: str, user: str, password: str, table_prefix: str) -> list:
        """
        获取所有以 table_prefix 开头的表名列表
        """
        import psycopg2
        try:
            conn = psycopg2.connect(
                host=host,
                port=port,
                database=database,
                user=user,
                password=password
            )
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT table_name FROM information_schema.tables 
                WHERE table_schema = 'public' AND table_name LIKE %s;
                """,
                (f"{table_prefix}%",)
            )
            tables = [row[0] for row in cursor.fetchall()]
            cursor.close()
            conn.close()
            return tables
        except Exception as e:
            logger.error(f"获取vector_store表失败: {str(e)}")
            return []

# 默认配置
DEFAULT_POSTGRES_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "knowledge_base",
    "user": "postgres",
    "password": "postgres",
    "table_name": "vector_store",
    "embed_dim": settings.VECTOR_DIM
}

def create_postgres_vector_store_builder(**kwargs) -> PostgresVectorStoreBuilder:
    """
    创建PostgreSQL向量存储构建器
    
    Args:
        **kwargs: 配置参数
        
    Returns:
        PostgresVectorStoreBuilder实例
    """
    config = DEFAULT_POSTGRES_CONFIG.copy()
    config.update(kwargs)
    
    return PostgresVectorStoreBuilder(**config)