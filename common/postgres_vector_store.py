import os
import json
from typing import Optional, List, Any, Dict
from pathlib import Path
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.schema import BaseNode
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from pandas.core.accessor import delegate_names
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
                 embed_dim: int = 1024,
                 persist_dir: str = None):
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
            persist_dir: 持久化目录，用于存储docstore和index_store
        """
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.table_name = table_name
        self.embed_dim = embed_dim
        
        # 设置持久化目录
        self.persist_dir = persist_dir or f"./storage/{table_name}"
        self.persist_path = Path(self.persist_dir)
        self.persist_path.mkdir(parents=True, exist_ok=True)
        
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
    
    def _create_storage_context(self, vector_store: PGVectorStore, nodes: List[BaseNode] = None) -> StorageContext:
        """
        创建统一的存储上下文，确保docstore和vector_store的一致性
        
        Args:
            vector_store: PostgreSQL向量存储
            nodes: 文档节点列表（用于创建时填充docstore）
            
        Returns:
            StorageContext实例
        """
        try:
            # 创建或加载docstore
            docstore_path = self.persist_path / "docstore.json"
            
            if docstore_path.exists():
                # 加载现有的docstore
                docstore = SimpleDocumentStore.from_persist_path(str(docstore_path))
                logger.info(f"加载现有docstore，包含 {len(docstore.docs)} 个文档")
            else:
                # 创建新的docstore
                docstore = SimpleDocumentStore()
                if nodes:
                    # 将节点添加到docstore
                    for node in nodes:
                        docstore.add_documents([node])
                    logger.info(f"创建新docstore，添加 {len(nodes)} 个节点")
                else:
                    logger.info("创建空的docstore")
            
            # 创建或加载index_store
            index_store_path = self.persist_path / "index_store.json"
            
            if index_store_path.exists():
                # 加载现有的index_store
                index_store = SimpleIndexStore.from_persist_path(str(index_store_path))
                logger.info("加载现有index_store")
            else:
                # 创建新的index_store
                index_store = SimpleIndexStore()
                logger.info("创建新index_store")
            
            # 创建存储上下文
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store,
                docstore=docstore,
                index_store=index_store
            )

            
            return storage_context
            
        except Exception as e:
            logger.error(f"创建存储上下文失败: {str(e)}")
            raise
    
    def _persist_storage_context(self, storage_context: StorageContext):
        """
        持久化存储上下文
        
        Args:
            storage_context: 存储上下文
        """
        try:
            # 持久化docstore
            docstore_path = self.persist_path / "docstore.json"
            storage_context.docstore.persist(str(docstore_path))
            
            # 持久化index_store
            index_store_path = self.persist_path / "index_store.json"
            storage_context.index_store.persist(str(index_store_path))
            
            logger.info(f"成功持久化存储上下文到 {self.persist_dir}")
            
        except Exception as e:
            logger.error(f"持久化存储上下文失败: {str(e)}")
            raise
    
    def create_index_from_nodes(self, nodes: List[BaseNode], embed_model) -> VectorStoreIndex:
        """
        从节点创建向量索引，使用统一的存储上下文管理
        
        Args:
            nodes: 文档节点列表
            embed_model: 嵌入模型
            
        Returns:
            VectorStoreIndex实例
        """
        try:
            # 创建向量存储
            vector_store = self.create_vector_store()
            
            # 创建统一的存储上下文，包含节点数据
            storage_context = self._create_storage_context(vector_store, nodes)
            
            # 创建向量索引
            index = VectorStoreIndex(
                nodes=nodes,
                storage_context=storage_context,
                embed_model=embed_model,
                show_progress=True
            )
            
            # 持久化存储上下文
            self._persist_storage_context(storage_context)
            
            logger.info(f"成功创建向量索引，包含 {len(nodes)} 个节点，并持久化存储上下文")
            return index
            
        except Exception as e:
            logger.error(f"创建向量索引失败: {str(e)}")
            raise
    
    def update_index_with_nodes(self, nodes: Optional[List[BaseNode]], embed_model, delete_doc_ids: Optional[List[str]] = None) -> VectorStoreIndex:
        """
        通过删除旧节点和插入新节点来更新现有向量索引，使用统一的存储上下文管理。
        如果索引不存在，则会根据提供的节点创建一个新索引。

        Args:
            nodes: 要插入的文档节点列表。
            embed_model: 嵌入模型。
            delete_doc_ids: 要删除的文档ID列表。

        Returns:
            更新后的 VectorStoreIndex 实例。
        """
        try:
            vector_store = self.create_vector_store()
            
            # 创建统一的存储上下文
            storage_context = self._create_storage_context(vector_store)
            
            # 从向量存储加载索引
            index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
                storage_context=storage_context,
                embed_model=embed_model
            )
            
            # 确保索引对象保留正确的storage_context
            # VectorStoreIndex.from_vector_store可能不会保留传入的storage_context
            index._storage_context = storage_context

            # 如果提供了 delete_doc_ids，则执行删除操作
            if delete_doc_ids:
                for doc_id in delete_doc_ids:
                    index.delete_ref_doc(doc_id, delete_from_docstore=True)
                    # 同时从持久化的docstore中删除
                    if doc_id in storage_context.docstore.docs:
                        del storage_context.docstore.docs[doc_id]
                logger.info(f"成功从索引中删除了 {len(delete_doc_ids)} 个文档。")

            # 如果提供了 nodes，则执行插入操作
            if nodes:
                index.insert_nodes(nodes)
                # 同时更新持久化的docstore
                for node in nodes:
                    storage_context.docstore.add_documents([node])
                logger.info(f"成功向索引中插入了 {len(nodes)} 个节点。")
            
            # 持久化更新后的存储上下文
            self._persist_storage_context(storage_context)

            logger.info(f"向量索引更新成功，并持久化存储上下文。")
            return index

        except Exception as e:
            logger.error(f"更新向量索引失败: {str(e)}")
            raise
    
    def _clear_table_data(self) -> bool:
        """
        清空表中的数据
        
        Returns:
            是否成功清空
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
            
            # 清空表数据
            cursor.execute(f"TRUNCATE TABLE {self.table_name};")
            conn.commit()
            
            cursor.close()
            conn.close()
            
            logger.info(f"成功清空表数据: {self.table_name}")
            return True
            
        except Exception as e:
            logger.error(f"清空表数据失败: {str(e)}")
            return False
    
    def load_index(self, embed_model) -> Optional[VectorStoreIndex]:
        """
        加载现有的向量索引，使用统一的存储上下文管理
        
        Args:
            embed_model: 嵌入模型
            
        Returns:
            VectorStoreIndex实例或None
        """
        try:
            # 创建向量存储
            vector_store = self.create_vector_store()
            
            # 创建统一的存储上下文（不传入nodes，会尝试加载现有的docstore）
            storage_context = self._create_storage_context(vector_store)
            
            # 加载向量索引
            index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
                storage_context=storage_context,
                embed_model=embed_model
            )
            
            # 确保索引对象保留正确的storage_context
            # VectorStoreIndex.from_vector_store可能不会保留传入的storage_context
            index._storage_context = storage_context
            
            # 检查docstore中的节点数量
            node_count = len(storage_context.docstore.docs) if storage_context.docstore else 0
            logger.info(f"成功加载PostgreSQL向量索引，docstore包含 {node_count} 个节点")
            
            return index
            
        except Exception as e:
            logger.error(f"加载PostgreSQL向量索引失败: {str(e)}")
            return None
    
    def delete_index(self) -> bool:
        """
        删除向量索引（删除表和持久化存储）
        
        Returns:
            是否成功删除
        """
        try:
            import psycopg2
            import shutil
            
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
            cursor.execute(f"DROP TABLE IF EXISTS data_{self.table_name} CASCADE;")
            conn.commit()
            
            cursor.close()
            conn.close()
            
            # 删除持久化存储目录
            if self.persist_path.exists():
                shutil.rmtree(self.persist_path)
                logger.info(f"成功删除持久化存储目录: {self.persist_dir}")
            
            logger.info(f"成功删除向量索引表和持久化存储: {self.table_name}")
            return True
            
        except Exception as e:
            logger.error(f"删除向量索引失败: {str(e)}")
            return False
    
    def clear_persistent_storage(self) -> bool:
        """
        清理持久化存储（保留数据库表）
        
        Returns:
            是否成功清理
        """
        try:
            import shutil
            
            if self.persist_path.exists():
                shutil.rmtree(self.persist_path)
                self.persist_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"成功清理持久化存储目录: {self.persist_dir}")
                return True
            else:
                logger.info(f"持久化存储目录不存在: {self.persist_dir}")
                return True
                
        except Exception as e:
            logger.error(f"清理持久化存储失败: {str(e)}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取向量存储统计信息，包括持久化存储信息
        
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
            
            row_count = 0
            if table_exists:
                # 获取行数
                cursor.execute(f"SELECT COUNT(*) FROM {self.table_name};")
                row_count = cursor.fetchone()[0]
            
            cursor.close()
            conn.close()
            
            # 检查持久化存储信息
            docstore_exists = (self.persist_path / "docstore.json").exists()
            index_store_exists = (self.persist_path / "index_store.json").exists()
            
            docstore_node_count = 0
            if docstore_exists:
                try:
                    docstore = SimpleDocumentStore.from_persist_path(str(self.persist_path / "docstore.json"))
                    docstore_node_count = len(docstore.docs)
                except Exception as e:
                    logger.warning(f"无法读取docstore统计信息: {e}")
            
            return {
                "table_exists": table_exists,
                "row_count": row_count,
                "table_name": self.table_name,
                "embed_dim": self.embed_dim,
                "persist_dir": self.persist_dir,
                "docstore_exists": docstore_exists,
                "index_store_exists": index_store_exists,
                "docstore_node_count": docstore_node_count,
                "storage_consistent": table_exists and docstore_exists and row_count > 0 and docstore_node_count > 0
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
    "embed_dim": settings.VECTOR_DIM,
    "persist_dir": None  # 默认为 ./storage/{table_name}
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