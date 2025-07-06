import time
import uuid
from typing import Dict, List, Optional, Any, AsyncGenerator, Tuple, ClassVar
from pathlib import Path
import logging
import json
from datetime import datetime, timedelta
import re
import hashlib

from llama_index.core import (
    VectorStoreIndex, 
    StorageContext, 
    load_index_from_storage,
)
from llama_index.core.retrievers import (
    VectorIndexRetriever,
    QueryFusionRetriever
)
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.tools import RetrieverTool, ToolMetadata
from llama_index.core.selectors import (
    PydanticSingleSelector,
)
from llama_index.core.llms import LLM

from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import RouterRetriever
from llama_index.core.chat_engine import CondensePlusContextChatEngine,SimpleChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.postprocessor import SentenceTransformerRerank, SimilarityPostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.postprocessor.types import BaseNodePostprocessor # Added import

from config import settings
from common.postgres_vector_store import create_postgres_vector_store_builder, PostgresVectorStoreBuilder
from utils.logging_config import setup_logging
from models import init_db
from dao.chat_dao import ChatDAO
from services.model_client_factory import ModelClientFactory

setup_logging()
logger = logging.getLogger(__name__)

# 查询意图分类常量
class QueryIntent:
    """查询意图分类"""
    LEGAL_LOOKUP = "legal_lookup"  # 查找法条
    POLICY_INTERPRETATION = "policy_interpretation"  # 政策解读
    COMPLIANCE_CHECK = "compliance_check"  # 合规检查
    CASE_ANALYSIS = "case_analysis"  # 案例分析
    GENERAL_QUESTION = "general_question"  # 一般问题


# Custom Node Postprocessor for Keyword-based Reranking
class UnifiedReranker(BaseNodePostprocessor):
    """统一重排序器，整合语义、结构和关键词重排序功能"""
    
    # 使用类变量定义字段
    keyword_field: str = "excerpt_keywords"
    header_path_field: str = "header_path"
    keyword_boost_factor: float = 1.2
    header_boost_factor: float = 3.0
    min_score_threshold: float = 0.1
    
    def __init__(
        self,
        semantic_reranker=None,
        keyword_field: str = "excerpt_keywords",
        header_path_field: str = "header_path",
        keyword_boost_factor: float = 1.2,
        header_boost_factor: float = 3.0,
        min_score_threshold: float = 0.1
    ):
        super().__init__()
        # 使用object.__setattr__来避免Pydantic验证
        object.__setattr__(self, 'semantic_reranker', semantic_reranker)
        object.__setattr__(self, 'keyword_field', keyword_field)
        object.__setattr__(self, 'header_path_field', header_path_field)
        object.__setattr__(self, 'keyword_boost_factor', keyword_boost_factor)
        object.__setattr__(self, 'header_boost_factor', header_boost_factor)
        object.__setattr__(self, 'min_score_threshold', min_score_threshold)
        
        # 预编译正则表达式模式
        object.__setattr__(self, 'chapter_patterns', [
            r'第([一二三四五六七八九十百千万\d]+)章',  # 第X章
            r'第([一二三四五六七八九十百千万\d]+)节',  # 第X节
            r'第([一二三四五六七八九十百千万\d]+)条',  # 第X条
            r'第([一二三四五六七八九十百千万\d]+)款',  # 第X款
            r'第([一二三四五六七八九十百千万\d]+)项',  # 第X项
            r'([一二三四五六七八九十百千万\d]+)、',    # X、
            r'\(([一二三四五六七八九十百千万\d]+)\)',  # (X)
        ])
    
    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        if query_bundle is None or not nodes:
            return nodes
        
        # 1. 语义重排序（如果提供了语义重排序器）
        if self.semantic_reranker:
            nodes = self.semantic_reranker.postprocess_nodes(nodes, query_bundle)
        
        # 2. 单次遍历应用所有重排序逻辑
        self._apply_unified_reranking_single_pass(nodes, query_bundle)
        
        # 3. 最终排序
        nodes.sort(key=lambda x: x.score or 0.0, reverse=True)
        
        return nodes
    
    def _apply_unified_reranking_single_pass(
        self, 
        nodes: List[NodeWithScore], 
        query_bundle: QueryBundle
    ) -> None:
        """单次遍历应用所有重排序逻辑"""
        query = query_bundle.query_str
        
        # 预处理查询信息
        query_keywords = set(query.lower().split())
        query_elements = self._extract_structural_elements(query)
        
        header_boost_count = 0
        keyword_boost_count = 0
        
        # 单次遍历处理所有节点
        for node_with_score in nodes:
            original_score = node_with_score.score or 0.0
            
            if original_score < self.min_score_threshold:
                continue
                
            metadata = node_with_score.node.metadata
            final_boost = 1.0
            
            # 1. Header path重排序逻辑
            if query_elements:
                header_path = metadata.get(self.header_path_field, "")
                
                path_similarity = self._calculate_header_path_similarity(query_elements, header_path)
                
                if path_similarity > 0:
                    header_boost = 1.0 + (self.header_boost_factor - 1.0) * path_similarity
                    final_boost *= header_boost
                    header_boost_count += 1
            
            # 2. 关键词重排序逻辑
            if query_keywords:
                excerpt_keywords_str = metadata.get(self.keyword_field, "")
                
                if isinstance(excerpt_keywords_str, str) and excerpt_keywords_str:
                    # 解析关键词字符串
                    actual_keywords_part = excerpt_keywords_str
                    if actual_keywords_part.startswith("'") and actual_keywords_part.endswith("'"):
                        actual_keywords_part = actual_keywords_part[1:-1]
                    
                    prefix = "关键词:"
                    if actual_keywords_part.startswith(prefix):
                        actual_keywords_part = actual_keywords_part[len(prefix):]
                    
                    node_keywords = set(k.strip().lower() for k in actual_keywords_part.split(',') if k.strip())
                    
                    if query_keywords.intersection(node_keywords):
                        final_boost *= self.keyword_boost_factor
                        keyword_boost_count += 1
            
            # 应用最终的综合提升分数
            if final_boost > 1.0:
                node_with_score.score = original_score * final_boost
        
        # 记录提升统计
        if header_boost_count > 0:
            logger.info(f"Applied header path boost to {header_boost_count} nodes")
        if keyword_boost_count > 0:
            logger.info(f"Applied keyword boost to {keyword_boost_count} nodes")
    
    def _extract_structural_elements(self, text: str) -> List[str]:
        """从文本中提取结构化元素（章、节、条等）"""
        elements = []
        for pattern in self.chapter_patterns:
            matches = re.findall(pattern, text)
            elements.extend(matches)
        return elements
    
    def _normalize_number(self, num_str: str) -> str:
        """标准化数字表示（中文数字转阿拉伯数字）"""
        chinese_to_arabic = {
            '一': '1', '二': '2', '三': '3', '四': '4', '五': '5',
            '六': '6', '七': '7', '八': '8', '九': '9', '十': '10',
            '十一': '11', '十二': '12', '十三': '13', '十四': '14', '十五': '15',
            '十六': '16', '十七': '17', '十八': '18', '十九': '19', '二十': '20'
        }
        
        if num_str in chinese_to_arabic:
            return chinese_to_arabic[num_str]
        
        if len(num_str) == 3 and num_str[1] == '十':
            tens = chinese_to_arabic.get(num_str[0], num_str[0])
            ones = chinese_to_arabic.get(num_str[2], num_str[2])
            if tens.isdigit() and ones.isdigit():
                return str(int(tens) * 10 + int(ones))
        
        if num_str.isdigit():
            return num_str
            
        return num_str
    
    def _calculate_header_path_similarity(self, query_elements: List[str], header_path: str) -> float:
        """计算查询元素与header_path的相似度"""
        if not query_elements or not header_path:
            return 0.0
        
        path_elements = self._extract_structural_elements(header_path)
        if not path_elements:
            return 0.0
        
        normalized_query = [self._normalize_number(elem) for elem in query_elements]
        normalized_path = [self._normalize_number(elem) for elem in path_elements]
        
        matches = 0
        for q_elem in normalized_query:
            if q_elem in normalized_path:
                matches += 1
        
        similarity = matches / len(normalized_query)
        if matches > 1:
            similarity *= 1.5
            
        return min(similarity, 1.0)



class ConversationSession:
    """对话会话"""
    def __init__(self, session_id: str, index_ids: List[str]):
        self.session_id = session_id
        self.index_ids = index_ids  # 支持多个索引
        self.created_at = time.time()
        self.last_activity = time.time()
        self.chat_engine = None
        self.memory = ChatMemoryBuffer.from_defaults(token_limit=3000)
        
    @property
    def message_count(self) -> int:
        """获取消息数量"""
        return ChatDAO.get_message_count(self.session_id)
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        """添加消息到数据库"""
        message = ChatDAO.add_message(self.session_id, role, content, metadata)
        if message:
            self.last_activity = time.time()
            # 更新数据库中的会话活动时间
            ChatDAO.update_session_activity(self.session_id)
        return message
    
    def get_chat_history(self, limit: int = None) -> List[Dict[str, Any]]:
        """从数据库获取聊天历史"""
        messages = ChatDAO.get_recent_messages(self.session_id, limit) if limit else ChatDAO.get_messages(self.session_id)
        return [msg.to_dict() for msg in messages]
    
    def load_history_to_memory(self) -> bool:
        """从数据库加载聊天历史到内存"""
        try:
            messages = ChatDAO.get_messages(self.session_id)
            
            # 重建聊天记忆
            if self.memory and messages:
                for msg in messages:
                    role = MessageRole.USER if msg.role == 'user' else MessageRole.ASSISTANT
                    chat_message = ChatMessage(role=role, content=msg.content)
                    self.memory.put(chat_message)
            
            logger.info(f"Loaded {len(messages)} messages from database to memory")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load chat history to memory: {e}")
            return False
    


class RAGService:
    """RAG检索增强服务"""
    
    _instance = None
    _initialized = False
    llm: LLM = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            # 初始化数据库
            try:
                init_db()
                logger.info("Database initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize database: {e}")
                raise
            
            self._setup_models()
            self.loaded_indexes: Dict[str, VectorStoreIndex] = {}
            self.sessions: Dict[str, ConversationSession] = {}
            self._retriever_cache: Dict[str, Any] = {}  # 统一的retriever缓存
            
            # 自动合并检索配置
            self.auto_merge_config = {
                'enable_by_default': True,  # 默认启用自动合并
                'max_expand_distance': 3,  # 最大扩散距离
                'min_merge_score': 0.3,  # 最小合并分数阈值
                'overlap_threshold': 0.7,  # 文本重叠阈值
                'max_merged_length': 8000  # 合并后最大长度
            }
            
            # 自动加载所有vector_store表的索引
            self._auto_load_vector_store_indexes()
            
            self._initialized = True
            logger.info("RAGService initialized with enhanced features")
    
    def _setup_models(self):
        """设置模型"""
        try:
            # 设置全局Settings配置
            from llama_index.core import Settings
            
            # 初始化嵌入模型
            self.embed_model = ModelClientFactory.create_embedding_client(settings.embedding_model_settings)
            
            # 初始化LLM
            self.llm = ModelClientFactory.create_llm_client(settings.llm_model_settings)
            
            # 创建语义重排序器
            semantic_reranker = ModelClientFactory.create_rerank_client(
                model_config=settings.rerank_model_settings,
                top_n=settings.RERANK_TOP_K
            )
            
            # 初始化统一重排序器
            self.unified_reranker = UnifiedReranker(
                semantic_reranker=semantic_reranker,
                keyword_field="excerpt_keywords",
                header_path_field="header_path",
                keyword_boost_factor=1.2,
                header_boost_factor=3.0,
                min_score_threshold=0.1
            )
            
            # 设置全局配置
            Settings.embed_model = self.embed_model
            Settings.llm = self.llm
            
            logger.info("RAG models initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG models: {e}")
            raise
    
    
   
    def _auto_load_vector_store_indexes(self):
        """自动加载所有vector_store表的索引"""
        try:
            if settings.VECTOR_STORE_TYPE != "postgres":
                logger.info("Vector store type is not postgres, skipping auto-load")
                return
            
            table_prefix = f"data_{settings.POSTGRES_TABLE_NAME}"
            # 获取所有vector_store表
            tables = PostgresVectorStoreBuilder.get_vector_store_tables(
                host=settings.POSTGRES_HOST,
                port=settings.POSTGRES_PORT,
                database=settings.POSTGRES_DATABASE,
                user=settings.POSTGRES_USER,
                password=settings.POSTGRES_PASSWORD,
                table_prefix=table_prefix
            )
            
            if not tables:
                logger.info(f"No {table_prefix} tables found")
                return
            
            logger.info(f"Found {len(tables)} {settings.POSTGRES_TABLE_NAME} tables, starting auto-load...")
            
            # 同步加载所有索引（避免事件循环冲突）
            success_count = 0
            for table_name in tables:
                try:
                    # 从表名提取索引ID
                    # 例如: vector_store_746b827d_94cb_4f3a -> 746b827d-94cb-4f3a
                    table_prefix_with_underscore = f"{table_prefix}_"
                    if table_name.startswith(table_prefix_with_underscore):
                        index_id = table_name[len(table_prefix_with_underscore):].replace("_", "-")
                        # 同步加载索引
                        logger.info(table_name)
                        if self._load_index_sysn(index_id):
                            success_count += 1
                            logger.info(f"Successfully loaded index {index_id} from table {table_name}")
                        else:
                            logger.error(f"Failed to load index {index_id} from table {table_name}")
                except Exception as e:
                    logger.error(f"Error loading index from table {table_name}: {e}")
            
            logger.info(f"Auto-loaded {success_count}/{len(tables)} indexes successfully")
            
        except Exception as e:
            logger.error(f"Error in auto-loading vector store indexes: {e}")
    

    def _load_index_sysn(self, index_id: str) -> bool:
        """同步加载向量索引"""
        try:
            if index_id in self.loaded_indexes:
                logger.info(f"Index {index_id} already loaded")
                return True
            
            if settings.VECTOR_STORE_TYPE == "postgres":
                # 加载PostgreSQL向量索引
                logger.info(f"Loading PostgreSQL vector index: {index_id}")
                
                # 创建PostgreSQL向量存储构建器
                postgres_builder = create_postgres_vector_store_builder(
                    host=settings.POSTGRES_HOST,
                    port=settings.POSTGRES_PORT,
                    database=settings.POSTGRES_DATABASE,
                    user=settings.POSTGRES_USER,
                    password=settings.POSTGRES_PASSWORD,
                    table_name=f"{settings.POSTGRES_TABLE_NAME}_{index_id.replace('-', '_')}",
                    embed_dim=settings.VECTOR_DIM
                )
                
                # 加载索引
                index = postgres_builder.load_index(self.embed_model)
                if index is None:
                    logger.error(f"Failed to load PostgreSQL index: {index_id}")
                    return False
                    
            else:
                # 加载FAISS索引（默认行为）
                index_path = Path(settings.INDEX_STORE_PATH) / index_id
                if not index_path.exists():
                    logger.error(f"Index path not found: {index_path}")
                    return False
                
                # 加载存储上下文
                storage_context = StorageContext.from_defaults(persist_dir=str(index_path))
                
                # 加载索引
                index = load_index_from_storage(
                    storage_context,
                    embed_model=self.embed_model
                )
            
            self.loaded_indexes[index_id] = index
            logger.info(f"Index {index_id} loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load index {index_id}: {e}")
            return False
    
    async def load_index(self, index_id: str) -> bool:
        """加载向量索引"""
        try:
            if index_id in self.loaded_indexes:
                logger.info(f"Index {index_id} already loaded")
                return True
            
            if settings.VECTOR_STORE_TYPE == "postgres":
                # 加载PostgreSQL向量索引
                logger.info(f"Loading PostgreSQL vector index: {index_id}")
                
                # 创建PostgreSQL向量存储构建器
                postgres_builder = create_postgres_vector_store_builder(
                    host=settings.POSTGRES_HOST,
                    port=settings.POSTGRES_PORT,
                    database=settings.POSTGRES_DATABASE,
                    user=settings.POSTGRES_USER,
                    password=settings.POSTGRES_PASSWORD,
                    table_name=f"{settings.POSTGRES_TABLE_NAME}_{index_id.replace('-', '_')}",
                    embed_dim=settings.VECTOR_DIM
                )
                
                # 加载索引
                index = postgres_builder.load_index(self.embed_model)
                if index is None:
                    logger.error(f"Failed to load PostgreSQL index: {index_id}")
                    return False
                    
            else:
                # 加载FAISS索引（默认行为）
                index_path = Path(settings.INDEX_STORE_PATH) / index_id
                if not index_path.exists():
                    logger.error(f"Index path not found: {index_path}")
                    return False
                
                # 加载存储上下文
                storage_context = StorageContext.from_defaults(persist_dir=str(index_path))
                
                # 加载索引
                index = load_index_from_storage(
                    storage_context,
                    embed_model=self.embed_model
                )
            
            self.loaded_indexes[index_id] = index
            logger.info(f"Index {index_id} loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load index {index_id}: {e}")
            return False
    
    def _get_or_create_retriever(self, index_ids: List[str], top_k: int = None, similarity_threshold: float = None):
        """获取或创建retriever（支持缓存复用）"""
        
        top_k = top_k or settings.RETRIEVAL_TOP_K
        cache_key = f"{'_'.join(sorted(index_ids))}_{top_k}"
        
        if cache_key in self._retriever_cache:
            logger.debug(f"Using cached retriever for key: {cache_key}")
            return self._retriever_cache[cache_key]
        
        # 创建新的retriever
        retriever = self._create_unified_retriever(index_ids, top_k, similarity_threshold)
        self._retriever_cache[cache_key] = retriever
        logger.info(f"Created and cached new retriever for key: {cache_key}")
        return retriever
    
    def _create_unified_retriever(self, index_ids: List[str], top_k: int, similarity_threshold: float = None):
        """创建统一的检索器（支持单索引和多索引）"""
        
        all_retrievers = []
        
        # 为每个索引创建检索器
        for index_id in index_ids:
            if index_id not in self.loaded_indexes:
                logger.warning(f"Index {index_id} not loaded, skipping")
                continue
                
            index = self.loaded_indexes[index_id]
            
            # 创建向量检索器
            vector_retriever = VectorIndexRetriever(
                index=index,
                similarity_top_k=top_k
            )

            all_retrievers.append(vector_retriever)
            
            # 尝试创建BM25检索器
            try:
                docstore = self._get_docstore_from_index(index, index_id)
                if docstore and docstore.docs:
                    bm25_retriever = BM25Retriever.from_defaults(
                        docstore=docstore,
                        similarity_top_k=top_k
                    )
                    all_retrievers.append(bm25_retriever)
                    logger.debug(f"Created BM25 retriever for index {index_id}")
            except Exception as e:
                logger.warning(f"Failed to create BM25 retriever for {index_id}: {e}")
        
        if not all_retrievers:
            raise ValueError("No valid retrievers created")
        
        # 根据检索器数量决定使用策略
        if len(all_retrievers) == 1:
            base_retriever = all_retrievers[0]
        else:
            # 设置检索器权重：向量检索器0.6，BM25检索器0.4
            retriever_weights = [0.6, 0.4] if len(all_retrievers) == 2 else None
            
            # 使用融合检索器
            base_retriever = QueryFusionRetriever(
                retrievers=all_retrievers,
                retriever_weights=retriever_weights,
                similarity_top_k=top_k,
                num_queries=3,
                mode="reciprocal_rerank",
                use_async=True
            )
        
      
        return base_retriever
    
    def _get_docstore_from_index(self, index: VectorStoreIndex, index_id: str):
        """从索引获取docstore"""
        if settings.VECTOR_STORE_TYPE == "postgres":
            if index._storage_context and index._storage_context.docstore:
                return index._storage_context.docstore
            else:
                # 重新获取storage_context
                postgres_builder = create_postgres_vector_store_builder(
                    host=settings.POSTGRES_HOST,
                    port=settings.POSTGRES_PORT,
                    database=settings.POSTGRES_DATABASE,
                    user=settings.POSTGRES_USER,
                    password=settings.POSTGRES_PASSWORD,
                    table_name=f"{settings.POSTGRES_TABLE_NAME}_{index_id.replace('-', '_')}",
                    embed_dim=settings.VECTOR_DIM
                )
                vector_store = postgres_builder.create_vector_store()
                storage_context = postgres_builder._create_storage_context(vector_store)
                return storage_context.docstore
        else:
            return index.docstore
    
    async def unified_retrieve(
        self, 
        index_ids: List[str],
        query: str, 
        top_k: int = None,
        similarity_threshold: float = None,
        use_cached_retriever: bool = True,
        enable_auto_merge: bool = None
    ) -> List[NodeWithScore]:
        """统一检索接口（支持单索引和多索引）"""
        try:
            # 确保所有索引已加载
            valid_index_ids = []
            for index_id in index_ids:
                if await self.load_index(index_id):
                    valid_index_ids.append(index_id)
                else:
                    logger.warning(f"Failed to load index: {index_id}")
            
            if not valid_index_ids:
                raise ValueError("No valid indexes loaded")
            
            top_k = top_k or settings.RETRIEVAL_TOP_K
            
            # 获取或创建retriever
            if use_cached_retriever:
                retriever = self._get_or_create_retriever(valid_index_ids, top_k, similarity_threshold)
            else:
                retriever = self._create_unified_retriever(valid_index_ids, top_k, similarity_threshold)
            
            # 执行检索
            query_bundle = QueryBundle(query_str=query)
            retrieved_nodes = await retriever.aretrieve(query_bundle)
            
            # 1. 统一重排序（整合语义、结构和关键词重排序）
            reranked_nodes = self.unified_reranker.postprocess_nodes(retrieved_nodes, query_bundle)
            
            # 2. 自动合并检索（在重排序之后执行，基于重排序后的分数进行合并）
            if enable_auto_merge is None:
                enable_auto_merge = self.auto_merge_config.get('enable_by_default', True)
            
            if enable_auto_merge:
                final_nodes = await self._auto_merge_retrieve(
                    reranked_nodes, valid_index_ids, query_bundle
                )
            else:
                final_nodes = reranked_nodes
            
            # 3. 最终截取top_k个结果
            final_nodes = final_nodes[:top_k]
            
            logger.info(f"Retrieved {len(final_nodes)} nodes for query: {query[:50]}...")
            return final_nodes
            
        except Exception as e:
            logger.error(f"Error in unified retrieve: {e}")
            raise
    
    # 保持向后兼容的方法
    async def hybrid_retrieve(self, index_id: str = None, enable_auto_merge: bool = None, **kwargs) -> List[NodeWithScore]:
        """混合检索（向后兼容）"""
        if index_id:
            return await self.unified_retrieve([index_id], enable_auto_merge=enable_auto_merge, **kwargs)
        else:
            raise ValueError("index_id is required for hybrid_retrieve")
    
    async def multi_index_retrieve(self, index_ids: List[str], enable_auto_merge: bool = None, **kwargs) -> List[NodeWithScore]:
        """多索引检索（向后兼容）"""
        return await self.unified_retrieve(index_ids, enable_auto_merge=enable_auto_merge, **kwargs)
    
    async def _auto_merge_retrieve(
        self, 
        nodes: List[NodeWithScore], 
        index_ids: List[str], 
        query_bundle: QueryBundle
    ) -> List[NodeWithScore]:
        """自动合并检索：以高分节点为中心向相邻节点扩散合并"""
        try:
            if not nodes:
                return nodes
            
            # 按分数排序，优先处理高分节点
            sorted_nodes = sorted(nodes, key=lambda x: x.score or 0, reverse=True)
            merged_nodes = []
            processed_node_ids = set()
            
            for anchor_node in sorted_nodes:
                if anchor_node.node.node_id in processed_node_ids:
                    continue
                
                # 以当前节点为锚点，向相邻节点扩散
                expanded_nodes = await self._expand_to_adjacent_nodes(
                    anchor_node, index_ids, processed_node_ids
                )
                
                if len(expanded_nodes) > 1:
                    # 合并扩散到的节点
                    merged_node = self._merge_expanded_nodes(expanded_nodes)
                    if merged_node:
                        merged_nodes.append(merged_node)
                else:
                    # 单个节点直接添加
                    merged_nodes.append(anchor_node)
                
                # 标记已处理的节点
                for node in expanded_nodes:
                    processed_node_ids.add(node.node.node_id)
            
            logger.info(f"Auto merge: processed {len(nodes)} nodes, merged to {len(merged_nodes)} nodes")
            return merged_nodes
            
        except Exception as e:
            logger.error(f"Error in auto merge retrieve: {e}")
            return nodes
    
    async def _expand_to_adjacent_nodes(
        self, 
        anchor_node: NodeWithScore, 
        index_ids: List[str],
        processed_node_ids: set
    ) -> List[NodeWithScore]:
        """以锚点节点为中心向相邻节点扩散"""
        try:
            max_distance = self.auto_merge_config.get('max_expand_distance', 3)
            min_score = self.auto_merge_config.get('min_merge_score', 0.3)
            
            # 获取锚点节点的信息
            anchor_file = anchor_node.node.metadata.get('original_file_path', '')
            anchor_node_id = anchor_node.node.node_id
            anchor_header_path = anchor_node.node.metadata.get('header_path', '')
            
            if not anchor_file:
                return [anchor_node]
            
            # 获取同级节点列表
            sibling_nodes = await self._get_sibling_nodes(anchor_file, anchor_header_path, index_ids)
            if not sibling_nodes:
                return [anchor_node]
            
            # 找到锚点在列表中的位置
            anchor_idx = -1
            for i, node in enumerate(sibling_nodes):
                if node.node.node_id == anchor_node_id:
                    anchor_idx = i
                    break
            
            if anchor_idx == -1:
                return [anchor_node]
            
            # 构建扩散列表，锚点在中间
            expanded_nodes = [anchor_node]
            
            # 向前找，找到符合条件的就插入到列表前面，断开就停止
            for i in range(anchor_idx - 1, max(anchor_idx - max_distance - 1, -1), -1):
                candidate = sibling_nodes[i]
                if (candidate.node.node_id not in processed_node_ids and
                    (candidate.score or 0) >= min_score and
                    self._should_merge_with_anchor(anchor_node, candidate)):
                    expanded_nodes.insert(0, candidate)  # 插入到前面
                else:
                    break  # 断开就停止
            
            # 向后找，找到符合条件的就插入到列表后面，断开就停止
            for i in range(anchor_idx + 1, min(anchor_idx + max_distance + 1, len(sibling_nodes))):
                candidate = sibling_nodes[i]
                if (candidate.node.node_id not in processed_node_ids and
                    (candidate.score or 0) >= min_score and
                    self._should_merge_with_anchor(anchor_node, candidate)):
                    expanded_nodes.append(candidate)  # 插入到后面
                else:
                    break  # 断开就停止
            
            return expanded_nodes
            
        except Exception as e:
            logger.error(f"Error expanding to adjacent nodes: {e}")
            return [anchor_node]
    
    async def _get_sibling_nodes(
        self, 
        file_path: str, 
        anchor_header_path: str,
        index_ids: List[str]
    ) -> List[NodeWithScore]:
        """获取指定文件中同级的所有节点，保持文档顺序"""
        sibling_nodes = []
        
        try:
            for index_id in index_ids:
                if index_id not in self.loaded_indexes:
                    continue
                
                index = self.loaded_indexes[index_id]
                docstore = self._get_docstore_from_index(index, index_id)
                
                if not docstore or not docstore.docs:
                    continue
                
                # 查找同一文件且同级的所有节点，保持docstore的自然顺序
                for doc_id, doc in docstore.docs.items():
                    if (doc.metadata.get('original_file_path', '') == file_path and 
                        doc.metadata.get('header_path', '') == anchor_header_path):
                        sibling_nodes.append(NodeWithScore(node=doc, score=0.5))
            
            return sibling_nodes
            
        except Exception as e:
            logger.error(f"Error getting sibling nodes: {e}")
            return []
    

    
    def _should_merge_with_anchor(self, anchor_node: NodeWithScore, candidate_node: NodeWithScore) -> bool:
        """判断候选节点是否应该与锚点节点合并"""
        try:
            # 检查文本重叠度
            anchor_text = anchor_node.node.get_content()
            candidate_text = candidate_node.node.get_content()
            
            overlap = self._calculate_text_overlap(anchor_text, candidate_text)
            overlap_threshold = self.auto_merge_config.get('overlap_threshold', 0.7)
            
            # 重叠度过高说明内容重复，不合并
            if overlap > overlap_threshold:
                return False
            
            # 检查合并后长度
            max_length = self.auto_merge_config.get('max_merged_length', 8000)
            if len(anchor_text) + len(candidate_text) > max_length:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking merge compatibility: {e}")
            return False
    
    def _merge_expanded_nodes(self, expanded_nodes: List[NodeWithScore]) -> Optional[NodeWithScore]:
        """合并扩散到的节点"""
        try:
            if not expanded_nodes:
                return None
            
            if len(expanded_nodes) == 1:
                return expanded_nodes[0]
            
            # 保持节点的自然顺序，不进行排序
            sorted_nodes = expanded_nodes
            base_node = max(expanded_nodes, key=lambda x: x.score or 0)  # 取最高分节点作为基础
            
            # 合并文本
            merged_texts = []
            for node in sorted_nodes:
                text = node.node.get_content().strip()
                if text:
                    merged_texts.append(text)
            
            merged_text = "\n\n".join(merged_texts)
            
            # 创建合并节点
            from llama_index.core.schema import TextNode
            merged_node = TextNode(
                text=merged_text,
                metadata=base_node.node.metadata.copy(),
                node_id=f"merged_{base_node.node.node_id}"
            )
            
            # 计算平均分数
            avg_score = sum(node.score or 0 for node in expanded_nodes) / len(expanded_nodes)
            
            return NodeWithScore(node=merged_node, score=avg_score)
            
        except Exception as e:
            logger.error(f"Error merging expanded nodes: {e}")
            return expanded_nodes[0] if expanded_nodes else None
    
    def _calculate_text_overlap(self, text1: str, text2: str) -> float:
        """计算两个文本的重叠比例"""
        if not text1 or not text2:
            return 0.0
        
        # 简单的字符级重叠计算
        text1_clean = re.sub(r'\s+', ' ', text1.lower().strip())
        text2_clean = re.sub(r'\s+', ' ', text2.lower().strip())
        
        # 如果一个文本完全包含另一个文本
        if text1_clean in text2_clean or text2_clean in text1_clean:
            return 1.0
        
        # 计算词级重叠
        words1 = set(text1_clean.split())
        words2 = set(text2_clean.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    

    

    

    

    

    
    def configure_auto_merge(
        self,
        enable_by_default: bool = None,
        max_expand_distance: int = None,
        min_merge_score: float = None,
        overlap_threshold: float = None,
        max_merged_length: int = None
    ):
        """配置自动合并检索参数
        
        Args:
            enable_by_default: 默认是否启用自动合并
            max_expand_distance: 最大扩散距离
            min_merge_score: 最小合并分数阈值
            overlap_threshold: 文本重叠阈值
            max_merged_length: 合并后最大长度
        """
        if enable_by_default is not None:
            self.auto_merge_config['enable_by_default'] = enable_by_default
        if max_expand_distance is not None:
            self.auto_merge_config['max_expand_distance'] = max_expand_distance
        if min_merge_score is not None:
            self.auto_merge_config['min_merge_score'] = min_merge_score
        if overlap_threshold is not None:
            self.auto_merge_config['overlap_threshold'] = overlap_threshold
        if max_merged_length is not None:
            self.auto_merge_config['max_merged_length'] = max_merged_length
        
        logger.info(f"Auto merge configuration updated: {self.auto_merge_config}")
    
    def get_auto_merge_config(self) -> Dict[str, Any]:
        """获取当前自动合并检索配置"""
        return self.auto_merge_config.copy()

    
    async def test_retrieval_recall(
        self, 
        index_id: str, 
        test_queries: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """测试检索召回率"""
        try:
            results = []
            total_queries = len(test_queries)
            
            for i, test_case in enumerate(test_queries):
                query = test_case["query"]
                expected_docs = test_case.get("expected_docs", [])
                
                # 执行检索
                retrieved_nodes = await self.hybrid_retrieve(
                    index_id, 
                    query, 
                    top_k=test_case.get("top_k", settings.RETRIEVAL_TOP_K)
                )
                
                # 计算召回率
                retrieved_doc_ids = [
                    node.node.metadata.get("file_path", "") 
                    for node in retrieved_nodes
                ]
                
                if expected_docs:
                    recall = len(set(retrieved_doc_ids) & set(expected_docs)) / len(expected_docs)
                else:
                    recall = None
                
                result = {
                    "query": query,
                    "retrieved_count": len(retrieved_nodes),
                    "retrieved_docs": retrieved_doc_ids[:5],  # 只返回前5个
                    "recall": recall,
                    "top_scores": [float(node.score) for node in retrieved_nodes[:3]]
                }
                results.append(result)
                
                logger.info(f"Processed test query {i+1}/{total_queries}")
            
            # 计算平均召回率
            valid_recalls = [r["recall"] for r in results if r["recall"] is not None]
            avg_recall = sum(valid_recalls) / len(valid_recalls) if valid_recalls else None
            
            return {
                "total_queries": total_queries,
                "average_recall": avg_recall,
                "results": results
            }
            
        except Exception as e:
            logger.error(f"Error in retrieval recall test: {e}")
            raise
    
    async def create_chat_session(
        self, 
        index_ids: List[str], 
        session_id: str = None,
        load_history: bool = True,
        user_id: str = None
    ) -> str:
        """创建对话会话（支持多索引）"""
        try:
            # 确保所有索引已加载
            valid_index_ids = []
            for index_id in index_ids:
                if await self.load_index(index_id):
                    valid_index_ids.append(index_id)
                else:
                    logger.warning(f"Failed to load index: {index_id}")
            
            if not valid_index_ids:
                raise ValueError("No valid indexes loaded")
            
            if not session_id:
                session_id = str(uuid.uuid4())
            
            # 在数据库中创建或更新会话
            db_session = ChatDAO.create_session(
                session_id=session_id,
                user_id=user_id,
                index_ids=valid_index_ids,
                metadata={
                    "created_at": datetime.utcnow().isoformat(),
                    "source": "rag_service"
                }
            )
            
            if not db_session:
                raise ValueError(f"Failed to create session in database: {session_id}")
            
            # 创建会话
            session = ConversationSession(session_id, valid_index_ids)
            session.user_id = user_id  # 添加用户ID
            
            # 加载历史记录到内存（如果需要）
            if load_history:
                session.load_history_to_memory()
            
            # 设置聊天引擎将在_setup_chat_engine_for_session中完成
            
            await self._setup_chat_engine_for_session(session, valid_index_ids)
            self.sessions[session_id] = session
            logger.info(f"Created chat session: {session_id} for user: {user_id} with indexes: {valid_index_ids}")
            return session_id
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.error(f"Error creating chat session: {e}")
            raise
    
    async def _setup_chat_engine_for_session(self, session: ConversationSession, index_ids: List[str]):
        """为会话设置聊天引擎"""
        try:
            # 确保所有索引已加载
            loaded_indices = []
            for index_id in index_ids:
                if index_id not in self.loaded_indexes:
                    await self.load_index(index_id)
                if index_id in self.loaded_indexes: # Check again after attempting to load
                    loaded_indices.append(self.loaded_indexes[index_id])
            
            if not loaded_indices:
                raise ValueError("No valid indices could be loaded for the session.")

            # 使用统一检索器方法
            logger.info("Using unified retriever for chat engine")

            # 使用统一重排序器
            node_postprocessors = [self.unified_reranker]

            # 使用统一检索器创建方法
            base_retriever = self._create_unified_retriever(
                index_ids=index_ids,
                top_k=settings.RETRIEVAL_TOP_K
            )
            
            logger.info(f"Created unified retriever for chat engine with {len(index_ids)} indices")
            
        
            # 创建聊天引擎
            session.chat_engine = CondensePlusContextChatEngine.from_defaults(
                retriever=base_retriever,
                memory=session.memory,
                node_postprocessors=node_postprocessors,
                llm=self.llm,
                verbose=True,
                system_prompt="你是一个专业的文档问答助手。请严格遵循以下规则：\n1. 严格根据给出的上下文回答用户问题，如果上下文没有直接相关信息，请明确说明'我没有此问题的完整信息'\n2. 使用中文回答\n3. 不要自我总结和概述，直接回答用户的具体问题\n4. 回答时不要提及'根据上下文内容'、'根据提供的信息'等表述，直接给出答案",
            )

            
        except Exception as e:
            logger.error(f"Error setting up chat engine for session {session.session_id}: {e}")
            raise
    
    async def chat_stream(
        self, 
        session_id: str, 
        message: str,
        user_id: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """流式对话"""
        try:
            if session_id not in self.sessions:
                # 尝试从数据库恢复会话
                if user_id:
                    db_session = ChatDAO.get_session_by_user(session_id, user_id)
                else:
                    db_session = ChatDAO.get_session(session_id)
                    
                if not db_session:
                    raise ValueError(f"Session not found: {session_id}")
                
                # 恢复会话到内存
                session = ConversationSession(session_id, db_session.index_ids)
                session.user_id = db_session.user_id
                session.load_history_to_memory()
                
                # 重新创建聊天引擎
                await self._setup_chat_engine_for_session(session, db_session.index_ids)
                self.sessions[session_id] = session
            
            session = self.sessions[session_id]
            session.last_activity = time.time()
            
            # 记录用户消息
            session.add_message("user", message)
            
            # 执行流式聊天（聊天引擎已使用统一检索器）
            streaming_response = await session.chat_engine.astream_chat(message)
            
            # 收集完整响应
            full_response = ""
            has_yielded_content = False
            
            async for token in streaming_response.async_response_gen():
                if token.strip():  # 只有非空token才计入内容
                    has_yielded_content = True
                full_response += token
                yield token
            
            # 检查是否需要fallback处理
            if not has_yielded_content or not full_response.strip() or full_response.strip().lower() in ["empty response", "none", "null"]:
                fallback_message = "抱歉，我无法为您提供有效的回答。请尝试重新表述您的问题。"
                logger.warning(f"Empty or invalid streaming response detected, using fallback message")
                yield fallback_message
                full_response = fallback_message
            
            source_nodes_data = []
            unique_sources = {}
            
            # 处理原始检索节点
            if hasattr(streaming_response, 'source_nodes') and streaming_response.source_nodes:
                for node in streaming_response.source_nodes:
                    if node.score > 0.6:
                        # 获取文件路径作为source
                        source = node.node.metadata.get('original_file_path', 'unknown')
                        # 提取文件名作为title
                        if source != 'unknown':
                            from pathlib import Path
                            title = Path(source).name
                        else:
                            title = source
                        
                        # 如果source已存在，保留score更高的
                        if source not in unique_sources or node.score > unique_sources[source]['score']:
                            unique_sources[source] = {
                                "content": node.node.text,
                                "score": float(node.score),
                                "metadata": node.node.metadata,
                                "source": source,
                                "title": title
                            }
            
            source_nodes_data = list(unique_sources.values())
            # 按分数排序
            source_nodes_data.sort(key=lambda x: x['score'], reverse=True)
            
            # Store source_nodes in session for later retrieval by frontend_service
            session.last_source_nodes = source_nodes_data

            # 记录助手响应
            session.add_message("assistant", full_response, {
                "source_nodes": source_nodes_data
            })
            
            logger.info(f"Completed streaming chat for session {session_id}")
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            logger.error(f"Error in streaming chat: {e}\nTraceback:\n{error_details}")
            yield f"Error: {str(e)}"
    
    async def chat(
        self, 
        session_id: str, 
        message: str,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """非流式对话"""
        try:
            if session_id not in self.sessions:
                # 尝试从数据库恢复会话
                if user_id:
                    db_session = ChatDAO.get_session_by_user(session_id, user_id)
                else:
                    db_session = ChatDAO.get_session(session_id)
                    
                if not db_session:
                    raise ValueError(f"Session not found: {session_id}")
                
                # 恢复会话到内存
                session = ConversationSession(session_id, db_session.index_ids)
                session.user_id = db_session.user_id
                session.load_history_to_memory()
                
                # 重新创建聊天引擎
                await self._setup_chat_engine_for_session(session, db_session.index_ids)
                self.sessions[session_id] = session
            
            session = self.sessions[session_id]
            session.last_activity = time.time()
            
            # 记录用户消息
            session.add_message("user", message)
            
            # 执行聊天（聊天引擎已使用统一检索器）
            response = await session.chat_engine.achat(message)
            logger.info(f"Raw LLM response: {response}")
            logger.info(f"Response type: {type(response)}")
            
            # 处理响应文本
            response_text = str(response).strip() if response else ""
            if not response_text or response_text.lower() in ["empty response", "none", "null"]:
                response_text = "抱歉，我无法为您提供有效的回答。请尝试重新表述您的问题。"
                logger.warning(f"Empty or invalid response detected, using fallback message")
            
            logger.info(f"Final response text: {response_text}")
            
            # 构建source_nodes信息，去重相同source的链接
            source_nodes_info = []
            unique_sources = {}
            
            # 处理原始检索节点
            if hasattr(response, 'source_nodes') and response.source_nodes:
                for node in response.source_nodes:
                    # 获取文件路径作为source
                    source = node.node.metadata.get('file_path', node.node.metadata.get('source', 'unknown'))
                    # 提取文件名作为title
                    if source != 'unknown':
                        from pathlib import Path
                        title = Path(source).name
                    else:
                        title = source
                    
                    # 如果source已存在，保留score更高的
                    if source not in unique_sources or node.score > unique_sources[source]['score']:
                        unique_sources[source] = {
                            "content": node.node.text[:200] + "...",
                            "score": float(node.score),
                            "metadata": node.node.metadata,
                            "source": source,
                            "title": title
                        }
            
            source_nodes_info = list(unique_sources.values())
            # 按分数排序
            source_nodes_info.sort(key=lambda x: x['score'], reverse=True)
            
            # 记录助手响应
            session.add_message("assistant", response_text, {
                "source_nodes": source_nodes_info
            })
            
            return {
                "response": response_text,
                "session_id": session_id,
                "source_nodes": source_nodes_info,
                 "raw_response_type": str(type(response)) # for debugging
            }
            
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            raise
    
    def get_session_info(self, session_id: str, user_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """获取会话信息"""
        # 先从内存中查找
        session = self.sessions.get(session_id)
        
        # 如果内存中没有，从数据库查找
        if user_id:
            db_session = ChatDAO.get_session_by_user(session_id, user_id)
        else:
            db_session = ChatDAO.get_session(session_id)
            
        if not db_session:
            return None
        
        # 获取消息数量
        message_count = ChatDAO.get_message_count(session_id)
        
        return {
            "session_id": db_session.session_id,
            "index_ids": db_session.index_ids,
            "created_at": db_session.created_at.isoformat() if db_session.created_at else None,
            "last_activity": db_session.last_activity.isoformat() if db_session.last_activity else None,
            "message_count": message_count,
            "is_active": db_session.is_active,
            "in_memory": session is not None
        }
    
    def get_chat_history(self, session_id: str, user_id: Optional[str] = None, limit: int = None) -> Optional[List[Dict[str, Any]]]:
        """获取聊天历史"""
        # 验证会话存在
        if user_id:
            db_session = ChatDAO.get_session_by_user(session_id, user_id)
        else:
            db_session = ChatDAO.get_session(session_id)
            
        if not db_session:
            return None
        
        # 从数据库获取消息
        if limit and limit > 0:
            messages = ChatDAO.get_recent_messages(session_id, limit)
        else:
            messages = ChatDAO.get_messages(session_id)
        
        return [msg.to_dict() for msg in messages]
    
    def export_chat_history(self, session_id: str, user_id: Optional[str] = None, export_path: str = None) -> Optional[str]:
        """导出聊天历史到文件"""
        # 验证会话存在
        if user_id:
            db_session = ChatDAO.get_session_by_user(session_id, user_id)
        else:
            db_session = ChatDAO.get_session(session_id)
            
        if not db_session:
            return None
        
        if not export_path:
            # 使用默认路径
            export_dir = Path(settings.UPLOAD_DIR) / "chat_exports"
            export_dir.mkdir(exist_ok=True)
            export_path = str(export_dir / f"{session_id}_{int(time.time())}.json")
        
        try:
            # 获取所有消息
            messages = ChatDAO.get_messages(session_id)
            
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "session_id": db_session.session_id,
                    "index_ids": db_session.index_ids,
                    "created_at": db_session.created_at.isoformat() if db_session.created_at else None,
                    "last_activity": db_session.last_activity.isoformat() if db_session.last_activity else None,
                    "message_count": len(messages),
                    "messages": [msg.to_dict() for msg in messages]
                }, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Exported chat history for session {session_id} to {export_path}")
            return export_path
            
        except Exception as e:
            logger.error(f"Failed to export chat history: {e}")
            return None
    
    def cleanup_expired_sessions(self):
        """清理过期会话"""
        # 清理内存中的过期会话（30天）
        current_time = time.time()
        memory_expire_time = settings.SESSION_SOFT_DELETE_DAYS * 24 * 3600  # 30天转换为秒
        expired_sessions = [
            session_id for session_id, session in self.sessions.items()
            if current_time - session.last_activity > memory_expire_time
        ]
        
        for session_id in expired_sessions:
            del self.sessions[session_id]
            logger.info(f"Cleaned up expired session from memory: {session_id}")
        
        # 清理数据库中的过期会话（软删除和硬删除）
        cleanup_result = ChatDAO.cleanup_expired_sessions()
        if cleanup_result["soft_deleted"] > 0 or cleanup_result["hard_deleted"] > 0:
            logger.info(f"Database cleanup: {cleanup_result['soft_deleted']} soft deleted, {cleanup_result['hard_deleted']} hard deleted")
        
        # 清理检索器缓存
        self.clear_retriever_cache()
    
    def clear_retriever_cache(self) -> int:
        """清理检索器缓存"""
        cache_count = len(self._retriever_cache)
        self._retriever_cache.clear()
        logger.info(f"Cleared {cache_count} cached retrievers")
        return cache_count
    
    def clear_retriever_cache_by_index(self, index_id: str) -> int:
        """清理特定索引相关的检索器缓存"""
        cleared_count = 0
        keys_to_remove = []
        
        for cache_key in self._retriever_cache.keys():
            # 缓存键格式: "index1_index2_top_k"
            index_part = cache_key.rsplit('_', 1)[0]  # 移除最后的top_k部分
            if index_id in index_part.split('_'):
                keys_to_remove.append(cache_key)
        
        for key in keys_to_remove:
            del self._retriever_cache[key]
            cleared_count += 1
        
        if cleared_count > 0:
            logger.info(f"Cleared {cleared_count} cached retrievers for index {index_id}")
        
        return cleared_count
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        return {
            "retriever_cache_size": len(self._retriever_cache),
            "retriever_cache_keys": list(self._retriever_cache.keys()),
            "loaded_indexes_count": len(self.loaded_indexes),
            "active_sessions_count": len(self.sessions)
        }
    
    def clear_all_caches(self) -> Dict[str, int]:
        """清理所有缓存"""
        retriever_cache_count = self.clear_retriever_cache()
        
        # 如果有其他缓存，也在这里清理
        result = {
            "retriever_cache_cleared": retriever_cache_count,
            "total_cleared": retriever_cache_count
        }
        
        logger.info(f"Cleared all caches: {result}")
        return result
    
    def get_loaded_indexes(self) -> List[str]:
        """获取已加载的索引列表"""
        return list(self.loaded_indexes.keys())
    
    def unload_index(self, index_id: str) -> bool:
        """卸载索引"""
        if index_id in self.loaded_indexes:
            del self.loaded_indexes[index_id]
            # 清理相关的检索器缓存
            cleared_cache_count = self.clear_retriever_cache_by_index(index_id)
            logger.info(f"Unloaded index: {index_id}, cleared {cleared_cache_count} related cached retrievers")
            return True
        return False
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """获取检索统计信息"""
        cache_stats = self.get_cache_stats()
        stats = {
            "loaded_indexes": list(self.loaded_indexes.keys()),
            "active_sessions": len(self.sessions),
            "cache_info": cache_stats
        }
      
        return stats
    
   

# 全局RAG服务实例 - 包含增强功能
rag_service = RAGService()