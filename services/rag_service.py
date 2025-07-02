import time
import uuid
from typing import Dict, List, Optional, Any, AsyncGenerator, Tuple, ClassVar
from pathlib import Path
import logging
import json
from datetime import datetime
import re

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

# Custom Node Postprocessor for Keyword-based Reranking
class KeywordMetadataReranker(BaseNodePostprocessor):
    keyword_field: str = "excerpt_keywords"
    boost_factor: float = 1.2  # Boost score by 20% for keyword match
    min_score_threshold: float = 0.1 # Only apply boost if original score is above this

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        if query_bundle is None or not nodes:
            return nodes

        # Simple query keyword extraction (lowercase, split by space)
        # Consider more sophisticated keyword extraction for queries if needed
        query_keywords = set(query_bundle.query_str.lower().split())
        if not query_keywords:
            return nodes

        new_nodes = []
        for node_with_score in nodes:
            original_score = node_with_score.score or 0.0
            boost_applied = False

            if original_score >= self.min_score_threshold:
                node = node_with_score.node
                metadata = node.metadata
                
                excerpt_keywords_str = metadata.get(self.keyword_field, "")
                if isinstance(excerpt_keywords_str, str) and excerpt_keywords_str:
                    actual_keywords_part = excerpt_keywords_str
                    # Remove potential surrounding single quotes if present
                    if actual_keywords_part.startswith("'") and actual_keywords_part.endswith("'"):
                        actual_keywords_part = actual_keywords_part[1:-1]
                    
                    prefix = "关键词:"
                    if actual_keywords_part.startswith(prefix):
                        actual_keywords_part = actual_keywords_part[len(prefix):]
                    
                    node_keywords = set(k.strip().lower() for k in actual_keywords_part.split(',') if k.strip())
                    
                    if query_keywords.intersection(node_keywords):
                        node_with_score.score = original_score * self.boost_factor
                        boost_applied = True
                        # logger.debug(f"Boosted node {node.node_id} for query '{query_bundle.query_str}' due to keyword match. Score: {original_score} -> {node_with_score.score}")
            
            new_nodes.append(node_with_score)
        
        # Re-sort nodes by score only if any boost was applied, to maintain stability otherwise
        # This sort should happen after all nodes are processed.
        # if any(n.score != orig_score for n, orig_score in zip(new_nodes, [nws.score for nws in nodes])):
        new_nodes.sort(key=lambda x: x.score or 0.0, reverse=True)
        
        return new_nodes

# Custom Node Postprocessor for Header Path-based Reranking
class HeaderPathReranker(BaseNodePostprocessor):
    """基于header_path的重排序器，专门处理章节条款查询"""
    header_path_field: str = "header_path"
    current_header_field: str = "current_header"
    boost_factor: float = 2.0  # 更高的提升因子，因为章节匹配很重要
    min_score_threshold: float = 0.05  # 较低的阈值，允许更多节点被提升
    
    # 预编译正则表达式模式
    chapter_patterns: ClassVar[List[str]] = [
        r'第([一二三四五六七八九十百千万\d]+)章',  # 第X章
        r'第([一二三四五六七八九十百千万\d]+)节',  # 第X节
        r'第([一二三四五六七八九十百千万\d]+)条',  # 第X条
        r'第([一二三四五六七八九十百千万\d]+)款',  # 第X款
        r'第([一二三四五六七八九十百千万\d]+)项',  # 第X项
        r'([一二三四五六七八九十百千万\d]+)、',    # X、
        r'\(([一二三四五六七八九十百千万\d]+)\)',  # (X)
    ]
        
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
        
        # 处理更复杂的中文数字
        if num_str in chinese_to_arabic:
            return chinese_to_arabic[num_str]
        
        # 处理"二十一"到"九十九"的情况
        if len(num_str) == 3 and num_str[1] == '十':
            tens = chinese_to_arabic.get(num_str[0], num_str[0])
            ones = chinese_to_arabic.get(num_str[2], num_str[2])
            if tens.isdigit() and ones.isdigit():
                return str(int(tens) * 10 + int(ones))
        
        # 如果已经是阿拉伯数字，直接返回
        if num_str.isdigit():
            return num_str
            
        return num_str
    
    def _calculate_header_path_similarity(self, query_elements: List[str], header_path: str) -> float:
        """计算查询元素与header_path的相似度"""
        if not query_elements or not header_path:
            return 0.0
        
        # 提取header_path中的结构化元素
        path_elements = self._extract_structural_elements(header_path)
        
        if not path_elements:
            return 0.0
        
        # 标准化所有元素
        normalized_query = [self._normalize_number(elem) for elem in query_elements]
        normalized_path = [self._normalize_number(elem) for elem in path_elements]
        
        # 计算匹配度
        matches = 0
        for q_elem in normalized_query:
            if q_elem in normalized_path:
                matches += 1
        
        # 返回匹配比例，考虑查询元素的重要性
        similarity = matches / len(normalized_query)
        
        # 如果匹配了多个元素，给予额外奖励
        if matches > 1:
            similarity *= 1.5
            
        return min(similarity, 1.0)
    
    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        if query_bundle is None or not nodes:
            return nodes
        
        query = query_bundle.query_str
        
        # 提取查询中的结构化元素
        query_elements = self._extract_structural_elements(query)
        
        if not query_elements:
            # 如果查询中没有结构化元素，不进行重排序
            return nodes
        
        logger.debug(f"Found structural elements in query: {query_elements}")
        
        new_nodes = []
        boost_applied_count = 0
        
        for node_with_score in nodes:
            original_score = node_with_score.score or 0.0
            
            if original_score >= self.min_score_threshold:
                node = node_with_score.node
                metadata = node.metadata
                
                # 获取header_path
                header_path = metadata.get(self.header_path_field, "")
                current_header = metadata.get(self.current_header_field, "")
                
                # 计算与header_path的相似度
                path_similarity = self._calculate_header_path_similarity(query_elements, header_path)
                header_similarity = self._calculate_header_path_similarity(query_elements, current_header)
                
                # 取最高相似度
                max_similarity = max(path_similarity, header_similarity)
                
                if max_similarity > 0:
                    # 根据相似度调整boost因子
                    dynamic_boost = 1.0 + (self.boost_factor - 1.0) * max_similarity
                    node_with_score.score = original_score * dynamic_boost
                    boost_applied_count += 1
                    
                    logger.debug(f"Boosted node {node.node_id[:8]}... for structural query. "
                               f"Similarity: {max_similarity:.2f}, Score: {original_score:.3f} -> {node_with_score.score:.3f}")
            
            new_nodes.append(node_with_score)
        
        if boost_applied_count > 0:
            logger.info(f"Applied header path boost to {boost_applied_count} nodes")
            # 重新排序
            new_nodes.sort(key=lambda x: x.score or 0.0, reverse=True)
        
        return new_nodes

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
            self._retriever_tool_cache: Dict[str, RetrieverTool] = {} # Modified cache type hint
            
            # 自动加载所有vector_store表的索引
            self._auto_load_vector_store_indexes()
            
            self._initialized = True
            logger.info("RAGService initialized")
    
    def _setup_models(self):
        """设置模型"""
        try:
            # 设置全局Settings配置
            from llama_index.core import Settings
            
            # 初始化嵌入模型
            self.embed_model = ModelClientFactory.create_embedding_client(settings.embedding_model_settings)
            
           
            self.keyword_reranker = KeywordMetadataReranker() # Initialize custom reranker
            self.header_path_reranker = HeaderPathReranker() # Initialize header path reranker
            
            # 初始化LLM
            self.llm = ModelClientFactory.create_llm_client(settings.llm_model_settings)
            
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
    
    async def hybrid_retrieve(
        self, 
        index_id: str, 
        query: str, 
        top_k: int = None,
        similarity_threshold: float = None
    ) -> List[NodeWithScore]:
        """混合检索（单索引）"""
        try:
            # 确保索引已加载
            if not await self.load_index(index_id):
                raise ValueError(f"Failed to load index: {index_id}")
            index = self.loaded_indexes[index_id]
            top_k = top_k or settings.RETRIEVAL_TOP_K
            similarity_threshold = similarity_threshold or settings.SIMILARITY_THRESHOLD
            
            # 创建向量检索器
            vector_retriever = VectorIndexRetriever(
                index=index,
                similarity_top_k=top_k
            )
            
            # 检查是否有数据可用于BM25检索
            try:
                # 对于PostgreSQL索引，需要从storage_context获取docstore
                docstore = None
                if settings.VECTOR_STORE_TYPE == "postgres":
                    # PostgreSQL索引的docstore在storage_context中
                    if index._storage_context and index._storage_context.docstore:
                        docstore = index._storage_context.docstore
                    else:
                        # 尝试重新获取storage_context
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
                        docstore = storage_context.docstore
                else:
                    # FAISS索引使用index.docstore
                    docstore = index.docstore
                
                if not docstore or not docstore.docs:
                    raise ValueError("BM25 retrieval failed: No documents available in the docstore.")
                
                # 创建BM25检索器
                bm25_retriever = BM25Retriever.from_defaults(
                    docstore=docstore,
                    similarity_top_k=top_k
                )
                
                logger.info(f"Created BM25 retriever with {len(docstore.docs)} documents")
                
                # 创建融合检索器
                fusion_retriever = QueryFusionRetriever(
                    retrievers=[vector_retriever, bm25_retriever],
                    similarity_top_k=top_k,
                    num_queries=3,  # 生成3个查询变体
                    mode="reciprocal_rerank",
                )
                
                # 执行检索
                query_bundle = QueryBundle(query_str=query)
                retrieved_nodes = await fusion_retriever.aretrieve(query_bundle)
            except Exception as e:
                logger.warning(f"BM25 retrieval failed: {e}, falling back to vector retrieval only")
                # 回退到仅使用向量检索
                query_bundle = QueryBundle(query_str=query)
                retrieved_nodes = await vector_retriever.aretrieve(query_bundle)
            
            # 应用多层重排序
            # 1. 首先应用语义重排序
            reranker = ModelClientFactory.create_rerank_client(model_config=settings.rerank_model_settings, top_n=top_k)
            reranked_nodes = reranker.postprocess_nodes(
                retrieved_nodes, query_bundle
            )
            
            # 2. 应用基于header_path的重排序（针对章节条款查询）
            header_reranked_nodes = self.header_path_reranker.postprocess_nodes(
                reranked_nodes, query_bundle
            )
            
            # 3. 应用关键词元数据重排序
            keyword_reranked_nodes = self.keyword_reranker.postprocess_nodes(
                header_reranked_nodes, query_bundle
            )
            logger.info(f'keyword_reranked_nodes:{len(keyword_reranked_nodes)}')
            # 过滤低相似度结果
            filtered_nodes = [
                node for node in keyword_reranked_nodes 
                if node.score >= similarity_threshold
            ]
            
            logger.info(f"Retrieved {len(filtered_nodes)} nodes for query: {query[:50]}...")
            return filtered_nodes
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.error(f"Error in hybrid retrieve: {e}")
            raise
    
    async def multi_index_retrieve(
        self, 
        index_ids: List[str], 
        query: str, 
        top_k: int = None,
        similarity_threshold: float = None
    ) -> List[NodeWithScore]:
        """多索引混合检索"""
        try:
            top_k = top_k or settings.RETRIEVAL_TOP_K
            similarity_threshold = similarity_threshold or settings.SIMILARITY_THRESHOLD
            
            all_retrievers = []
            
            # 为每个索引创建检索器
            for index_id in index_ids:
                if not await self.load_index(index_id):
                    logger.warning(f"Failed to load index: {index_id}, skipping")
                    continue
                    
                index = self.loaded_indexes[index_id]
                
                # 创建向量检索器
                vector_retriever = VectorIndexRetriever(
                    index=index,
                    similarity_top_k=top_k
                )
                all_retrievers.append(vector_retriever)
                
                # 尝试添加BM25检索器
                try:
                    # 对于PostgreSQL索引，需要从storage_context获取docstore
                    docstore = None
                    if settings.VECTOR_STORE_TYPE == "postgres":
                        # PostgreSQL索引的docstore在storage_context中
                        if index._storage_context and index._storage_context.docstore:
                            docstore = index._storage_context.docstore
                        else:
                            # 尝试重新获取storage_context
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
                            docstore = storage_context.docstore
                    else:
                        # FAISS索引使用index.docstore
                        docstore = index.docstore
                    
                    if docstore and docstore.docs:
                        bm25_retriever = BM25Retriever.from_defaults(
                            docstore=docstore,
                            similarity_top_k=top_k
                        )
                        all_retrievers.append(bm25_retriever)
                        logger.info(f"Created BM25 retriever for index {index_id} with {len(docstore.docs)} docs")
                    else:
                        logger.warning(f"BM25 retrieval failed: No documents available in index {index_id}")
                except Exception as e:
                    logger.warning(f"Failed to create BM25 retriever for {index_id}: {e}")
            
            if not all_retrievers:
                raise ValueError("No valid retrievers created")
            
            # 创建多检索器融合
            fusion_retriever = QueryFusionRetriever(
                retrievers=all_retrievers,
                similarity_top_k=top_k,
                num_queries=3,
                mode="reciprocal_rerank"
            )
            
            # 执行检索
            query_bundle = QueryBundle(query_str=query)
            retrieved_nodes = await fusion_retriever.aretrieve(query_bundle)
            # 应用多层重排序
            # 1. 首先应用语义重排序
            reranker = ModelClientFactory.create_rerank_client(model_config=settings.rerank_model_settings, top_n=top_k)
            reranked_nodes = reranker.postprocess_nodes(
                retrieved_nodes, query_bundle
            )
            
            # 2. 应用基于header_path的重排序（针对章节条款查询）
            header_reranked_nodes = self.header_path_reranker.postprocess_nodes(
                reranked_nodes, query_bundle
            )
            
            # 3. 应用关键词元数据重排序
            keyword_reranked_nodes = self.keyword_reranker.postprocess_nodes(
                header_reranked_nodes, query_bundle
            )
            
            # 过滤低相似度结果
            filtered_nodes = [
                node for node in keyword_reranked_nodes 
                if node.score >= similarity_threshold
            ]
            
            logger.info(f"Multi-index retrieved {len(filtered_nodes)} nodes from {len(index_ids)} indexes")
            return filtered_nodes
            
        except Exception as e:
            logger.error(f"Error in multi-index retrieve: {e}")
            raise
    
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

            reranker = ModelClientFactory.create_rerank_client(
                model_config=settings.rerank_model_settings,
                top_n=settings.RETRIEVAL_TOP_K
            )
            node_postprocessors = [reranker, self.header_path_reranker, self.keyword_reranker]

            # 导入必要的模块
            from llama_index.retrievers.bm25 import BM25Retriever
            from llama_index.core.retrievers import QueryFusionRetriever
            from services.filtered_retriever import FilteredRetriever
            
            # 统一处理单索引和多索引情况，使用混合检索
            all_retrievers = []
            
            # 为每个索引创建检索器
            for index_obj in loaded_indices:
                # 获取索引ID
                current_index_id = next((id for id, idx in self.loaded_indexes.items() if idx == index_obj), None)
                
                # 创建向量检索器（带重排序器）
                vector_retriever = VectorIndexRetriever(
                    index=index_obj,
                    similarity_top_k=settings.RETRIEVAL_TOP_K,  # 分配一半给向量检索
                )
                all_retrievers.append(vector_retriever)
                logger.info(f"Created vector retriever for index {current_index_id or 'unknown'}")
                
                # 尝试创建BM25检索器
                try:
                    if index_obj._storage_context and index_obj._storage_context.docstore:
                        bm25_retriever = BM25Retriever.from_defaults(
                            docstore=index_obj._storage_context.docstore,
                            similarity_top_k=settings.RETRIEVAL_TOP_K  
                        )
                        all_retrievers.append(bm25_retriever)
                        logger.info(f"Created BM25 retriever for index {current_index_id or 'unknown'} with {len(index_obj.docstore.docs)} docs")
                    else:
                        logger.warning(f"BM25 retrieval failed: No documents available in index {current_index_id or 'unknown'}")
                except Exception as e:
                    logger.warning(f"Failed to create BM25 retriever for index {current_index_id or 'unknown'}: {e}")
            
            if not all_retrievers:
                raise ValueError("No valid retrievers created")
            
            # 创建检索器 - 根据检索器数量决定使用单检索器还是融合检索器
            if len(all_retrievers) == 1:
                # 只有一个检索器，直接使用
                base_retriever = all_retrievers[0]
                logger.info("Using single retriever with node postprocessors")
            else:
                # 多个检索器，使用融合检索器
                try:
                    # QueryFusionRetriever不支持node_postprocessors参数
                    base_retriever = QueryFusionRetriever(
                        retrievers=all_retrievers,
                        similarity_top_k=settings.RETRIEVAL_TOP_K,
                        num_queries=4,  # 生成4个查询变体
                        mode="reciprocal_rerank",
                        use_async=True,
                        verbose=True
                    )
                    logger.info(f"Created QueryFusionRetriever with {len(all_retrievers)} retrievers")
                except Exception as e:
                    # 如果融合检索器创建失败，使用第一个检索器作为备选
                    logger.warning(f"Failed to create QueryFusionRetriever: {e}, falling back to first retriever")
                    base_retriever = all_retrievers[0]
            
            # 使用过滤检索器包装基础检索器，确保相似度过滤被应用
            retriever = FilteredRetriever(
                base_retriever=base_retriever,
                similarity_threshold=settings.SIMILARITY_THRESHOLD
            )
            logger.info(f"Wrapped retriever with similarity threshold filter: {settings.SIMILARITY_THRESHOLD}")
            
        
            # 创建聊天引擎
            session.chat_engine = CondensePlusContextChatEngine.from_defaults(
                retriever=base_retriever,
                memory=session.memory,
                node_postprocessors=node_postprocessors,
                llm=self.llm,
                verbose=True,
                system_prompt="你是一个专业的文档查询助手。请直接引用和展示提供的上下文文档中的原始内容来回答用户问题，不要对内容进行总结、概括或添加自己的评价。如果上下文信息不足以回答问题，请明确告知用户缺少相关信息。请保持原文档的完整性和准确性，用中文回答。",
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
            
            # 执行流式聊天
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
            if hasattr(streaming_response, 'source_nodes') and streaming_response.source_nodes:
                # 使用字典来去重，以source为key
                unique_sources = {}
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
            # 执行聊天
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
            if hasattr(response, 'source_nodes') and response.source_nodes:
                # 使用字典来去重，以source为key
                unique_sources = {}
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
    
    def get_loaded_indexes(self) -> List[str]:
        """获取已加载的索引列表"""
        return list(self.loaded_indexes.keys())
    
    def unload_index(self, index_id: str) -> bool:
        """卸载索引"""
        if index_id in self.loaded_indexes:
            del self.loaded_indexes[index_id]
            logger.info(f"Unloaded index: {index_id}")
            return True
        return False

# 全局RAG服务实例
rag_service = RAGService()