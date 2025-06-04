import asyncio
import time
import uuid
from typing import Dict, List, Optional, Any, AsyncGenerator, Tuple
from pathlib import Path
import logging
import json
from datetime import datetime

from llama_index.core import (
    VectorStoreIndex, 
    StorageContext, 
    load_index_from_storage,
    Settings,
    get_response_synthesizer
)
from llama_index.core.retrievers import (
    VectorIndexRetriever,
    QueryFusionRetriever
)
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.chat_engine import CondensePlusContextChatEngine,SimpleChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.postprocessor import SentenceTransformerRerank, SimilarityPostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.base.llms.types import ChatMessage, MessageRole

from config import settings
from postgres_vector_store import create_postgres_vector_store_builder
from utils.logging_config import setup_logging
from models import ChatDAO, init_db

setup_logging()
logger = logging.getLogger(__name__)

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
            self._initialized = True
            logger.info("RAGService initialized")
    
    def _setup_models(self):
        """设置模型"""
        try:
            # 设置全局Settings配置
            from llama_index.core import Settings
            Settings.context_window = settings.LLM_CONTEXT_WINDOW
            Settings.num_output = settings.LLM_NUM_OUTPUT
            
            # 初始化嵌入模型
            self.embed_model = HuggingFaceEmbedding(
                model_name=settings.EMBED_MODEL_PATH,
                device=settings.GPU_DEVICE if settings.USE_GPU else "cpu",
                trust_remote_code=True
            )
            
            # 初始化重排序模型
            self.reranker = SentenceTransformerRerank(
                model=settings.RERANK_MODEL_PATH,
                top_n=settings.RERANK_TOP_K,
                device=settings.GPU_DEVICE if settings.USE_GPU else "cpu"
            )
            
            # 初始化LLM
            self._setup_llm()
            
            # 设置全局配置
            Settings.embed_model = self.embed_model
            Settings.llm = self.llm
            
            logger.info("RAG models initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG models: {e}")
            raise
    
    def _setup_llm(self):
        """设置LLM"""
        if settings.LLM_API_BASE and settings.LLM_API_KEY:
            # 使用第三方API
            llm_kwargs = {
                "api_base": settings.LLM_API_BASE,
                "api_key": settings.LLM_API_KEY,
                "model": settings.LLM_MODEL_NAME,
                "max_tokens": settings.LLM_MAX_TOKENS,
                "temperature": settings.LLM_TEMPERATURE,
                "context_window": settings.LLM_CONTEXT_WINDOW,
                "is_chat_model": True,
                "timeout": 60.0,  # 添加超时设置
                "max_retries": 3   # 添加重试设置
            }
            
            # 只有当api_version不为空时才添加
            if settings.LLM_API_VERSION:
                llm_kwargs["api_version"] = settings.LLM_API_VERSION
            
            self.llm = OpenAILike(**llm_kwargs)
            logger.info(f"Using external LLM API: {settings.LLM_API_BASE} with model: {settings.LLM_MODEL_NAME}")
            logger.info(f"LLM config - max_tokens: {settings.LLM_MAX_TOKENS}, temperature: {settings.LLM_TEMPERATURE}")
        else:
            # 使用本地模型
            self.llm = HuggingFaceLLM(
                model_name=settings.LLM_MODEL_PATH,
                device_map="auto" if settings.USE_GPU else None,
                model_kwargs={
                    "torch_dtype": "auto",
                    "trust_remote_code": True
                },
                tokenizer_kwargs={
                    "trust_remote_code": True
                },
                max_new_tokens=settings.LLM_MAX_TOKENS,
                temperature=settings.LLM_TEMPERATURE
            )
            logger.info(f"Using local LLM: {settings.LLM_MODEL_PATH}")
    
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
                # 从docstore获取所有文档
                docs = list(index.docstore.docs.values())
                if not docs:
                    logger.warning("No documents found in docstore, using vector retrieval only")
                    # 如果没有文档，只使用向量检索
                    query_bundle = QueryBundle(query_str=query)
                    retrieved_nodes = await vector_retriever.aretrieve(query_bundle)
                else:
                    # 创建BM25检索器
                    bm25_retriever = BM25Retriever.from_defaults(
                        docstore=index.docstore,
                        similarity_top_k=top_k
                    )
                    
                    # 创建融合检索器
                    fusion_retriever = QueryFusionRetriever(
                        retrievers=[vector_retriever, bm25_retriever],
                        similarity_top_k=top_k,
                        num_queries=3,  # 生成3个查询变体
                        mode="reciprocal_rerank"
                    )
                    
                    # 执行检索
                    query_bundle = QueryBundle(query_str=query)
                    retrieved_nodes = await fusion_retriever.aretrieve(query_bundle)
            except Exception as e:
                logger.warning(f"BM25 retrieval failed: {e}, falling back to vector retrieval only")
                # 回退到仅使用向量检索
                query_bundle = QueryBundle(query_str=query)
                retrieved_nodes = await vector_retriever.aretrieve(query_bundle)
            
            # 应用重排序
            reranked_nodes = self.reranker.postprocess_nodes(
                retrieved_nodes, query_bundle
            )
            
            # 过滤低相似度结果
            filtered_nodes = [
                node for node in reranked_nodes 
                if node.score >= similarity_threshold
            ]
            
            logger.info(f"Retrieved {len(filtered_nodes)} nodes for query: {query[:50]}...")
            return filtered_nodes
            
        except Exception as e:
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
                    docs = list(index.docstore.docs.values())
                    if docs:
                        bm25_retriever = BM25Retriever.from_defaults(
                            docstore=index.docstore,
                            similarity_top_k=top_k
                        )
                        all_retrievers.append(bm25_retriever)
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
            
            # 应用重排序
            reranked_nodes = self.reranker.postprocess_nodes(
                retrieved_nodes, query_bundle
            )
            
            # 过滤低相似度结果
            filtered_nodes = [
                node for node in reranked_nodes 
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
        load_history: bool = True
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
            
            # 加载历史记录到内存（如果需要）
            if load_history:
                session.load_history_to_memory()
            
            # 设置聊天引擎将在_setup_chat_engine_for_session中完成
            
            await self._setup_chat_engine_for_session(session, valid_index_ids)
            self.sessions[session_id] = session
            logger.info(f"Created chat session: {session_id} with indexes: {valid_index_ids}")
            return session_id
            
        except Exception as e:
            logger.error(f"Error creating chat session: {e}")
            raise
    
    async def _setup_chat_engine_for_session(self, session: ConversationSession, index_ids: List[str]):
        """为会话设置聊天引擎"""
        try:
            # 确保所有索引已加载
            for index_id in index_ids:
                if index_id not in self.loaded_indexes:
                    await self.load_index(index_id)
            
            # 创建多索引检索器
            if len(index_ids) == 1:
                # 单索引情况
                index = self.loaded_indexes[index_ids[0]]
                vector_retriever = VectorIndexRetriever(
                    index=index,
                    similarity_top_k=settings.RETRIEVAL_TOP_K
                )
            else:
                # 多索引情况 - 创建自定义检索器
                class MultiIndexRetriever:
                    def __init__(self, rag_service, index_ids):
                        self.rag_service = rag_service
                        self.index_ids = index_ids
                    
                    async def aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
                        return await self.rag_service.multi_index_retrieve(
                            self.index_ids, 
                            query_bundle.query_str,
                            top_k=settings.RETRIEVAL_TOP_K
                        )
                    
                    def retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
                        # 同步版本，通过异步版本实现
                        import asyncio
                        loop = asyncio.get_event_loop()
                        return loop.run_until_complete(self.aretrieve(query_bundle))
                
                vector_retriever = MultiIndexRetriever(self, index_ids)
            

            # 创建聊天引擎
            session.chat_engine = CondensePlusContextChatEngine.from_defaults(
                retriever=vector_retriever,
                memory=session.memory,
                llm=self.llm,
                verbose=True,
                system_prompt="你是一个有用的AI助手。请基于提供的上下文信息回答用户的问题。如果上下文信息不足以回答问题，请诚实地告知用户，并尽可能提供相关的帮助。请用中文回答。",
            )

            
        except Exception as e:
            logger.error(f"Error setting up chat engine for session {session.session_id}: {e}")
            raise
    
    async def chat_stream(
        self, 
        session_id: str, 
        message: str
    ) -> AsyncGenerator[str, None]:
        """流式对话"""
        try:
            if session_id not in self.sessions:
                # 尝试从数据库恢复会话
                db_session = ChatDAO.get_session(session_id)
                if not db_session:
                    raise ValueError(f"Session not found: {session_id}")
                
                # 恢复会话到内存
                session = ConversationSession(session_id, db_session.index_ids)
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
            async for token in streaming_response.async_response_gen():
                full_response += token
                yield token
            
            # 记录助手响应
            session.add_message("assistant", full_response, {
                "source_nodes": [
                    {
                        "content": node.node.text[:200] + "...",
                        "score": float(node.score),
                        "metadata": node.node.metadata
                    }
                    for node in streaming_response.source_nodes
                ] if hasattr(streaming_response, 'source_nodes') else []
            })
            
            logger.info(f"Completed streaming chat for session {session_id}")
            
        except Exception as e:
            logger.error(f"Error in streaming chat: {e}")
            yield f"Error: {str(e)}"
    
    async def chat(
        self, 
        session_id: str, 
        message: str
    ) -> Dict[str, Any]:
        """非流式对话"""
        try:
            if session_id not in self.sessions:
                # 尝试从数据库恢复会话
                db_session = ChatDAO.get_session(session_id)
                if not db_session:
                    raise ValueError(f"Session not found: {session_id}")
                
                # 恢复会话到内存
                session = ConversationSession(session_id, db_session.index_ids)
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
            
            source_nodes_info = [
                {
                    "content": node.node.text[:200] + "...",
                    "score": float(node.score),
                    "metadata": node.node.metadata
                }
                for node in response.source_nodes
            ] if hasattr(response, 'source_nodes') and response.source_nodes else []
            
            # 记录助手响应
            session.add_message("assistant", response_text, {
                "source_nodes": source_nodes_info
            })
            
            return {
                "response": response_text,
                "source_nodes": source_nodes_info,
                "session_id": session_id,
                "message_count": session.message_count
            }
            
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            raise
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """获取会话信息"""
        # 先从内存中查找
        session = self.sessions.get(session_id)
        
        # 如果内存中没有，从数据库查找
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
    
    def get_chat_history(self, session_id: str, limit: int = None) -> Optional[List[Dict[str, Any]]]:
        """获取聊天历史"""
        # 验证会话存在
        db_session = ChatDAO.get_session(session_id)
        if not db_session:
            return None
        
        # 从数据库获取消息
        if limit and limit > 0:
            messages = ChatDAO.get_recent_messages(session_id, limit)
        else:
            messages = ChatDAO.get_messages(session_id)
        
        return [msg.to_dict() for msg in messages]
    
    def export_chat_history(self, session_id: str, export_path: str = None) -> Optional[str]:
        """导出聊天历史到文件"""
        # 验证会话存在
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
        # 清理内存中的过期会话
        current_time = time.time()
        expired_sessions = [
            session_id for session_id, session in self.sessions.items()
            if current_time - session.last_activity > settings.TASK_EXPIRE_TIME
        ]
        
        for session_id in expired_sessions:
            del self.sessions[session_id]
            logger.info(f"Cleaned up expired session from memory: {session_id}")
        
        # 清理数据库中的过期会话
        expire_hours = settings.TASK_EXPIRE_TIME / 3600  # 转换为小时
        cleaned_count = ChatDAO.cleanup_expired_sessions(expire_hours)
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} expired sessions from database")
    
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