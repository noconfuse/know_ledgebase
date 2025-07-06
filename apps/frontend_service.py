from fastapi import FastAPI, HTTPException, BackgroundTasks, Query, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import logging
import asyncio
import json
from contextlib import asynccontextmanager

import sys
import os

# 添加项目根目录到sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
from services.rag_service import rag_service
from auth.auth_routes import router as auth_router
from auth.dependencies import get_current_active_user, get_optional_current_user
from models.database import get_db
from dao.chat_dao import ChatDAO
from common.response import success_response, error_response, ErrorCodes
from common.exception_handler import setup_exception_handlers
from utils.logging_config import setup_logging, get_logger

# 初始化日志系统
setup_logging()
logger = get_logger(__name__)

# Pydantic模型
class RetrievalRequest(BaseModel):
    """检索请求"""
    index_id: str = Field(..., description="索引ID")
    query: str = Field(..., description="查询文本")
    top_k: Optional[int] = Field(default=None, description="返回结果数量")
    similarity_threshold: Optional[float] = Field(default=None, description="相似度阈值")

class ChatRequest(BaseModel):
    """聊天请求"""
    session_id: str = Field(..., description="会话ID")
    message: str = Field(..., description="用户消息")

class CreateSessionRequest(BaseModel):
    """创建会话请求"""
    index_ids: List[str] = Field(..., description="索引ID列表（支持多索引）")
    session_id: Optional[str] = Field(default=None, description="自定义会话ID")
    load_history: Optional[bool] = Field(default=True, description="是否加载历史记录")

class MultiIndexRetrievalRequest(BaseModel):
    """多索引检索请求"""
    index_ids: List[str] = Field(..., description="索引ID列表")
    query: str = Field(..., description="查询文本")
    top_k: Optional[int] = Field(default=None, description="返回结果数量")
    similarity_threshold: Optional[float] = Field(default=None, description="相似度阈值")

class ChatHistoryRequest(BaseModel):
    """聊天历史请求"""
    session_id: str = Field(..., description="会话ID")
    limit: Optional[int] = Field(default=None, description="返回消息数量限制")

class ExportHistoryRequest(BaseModel):
    """导出历史请求"""
    session_id: str = Field(..., description="会话ID")
    export_path: Optional[str] = Field(default=None, description="导出文件路径")

class ClearHistoryRequest(BaseModel):
    """清除历史请求"""
    session_id: str = Field(..., description="会话ID")

class RecallTestRequest(BaseModel):
    """召回测试请求"""
    index_id: str = Field(..., description="索引ID")
    test_queries: List[Dict[str, Any]] = Field(..., description="测试查询列表")

class EnhancedRetrievalRequest(BaseModel):
    """增强检索请求"""
    query: str = Field(..., description="查询文本")
    top_k: Optional[int] = Field(default=10, description="返回结果数量")
    enable_metadata_filter: Optional[bool] = Field(default=True, description="启用元数据过滤")
    enable_intent_classification: Optional[bool] = Field(default=True, description="启用意图分类")
    enable_cache: Optional[bool] = Field(default=True, description="启用缓存")
    custom_filters: Optional[Dict[str, Any]] = Field(default=None, description="自定义过滤条件")

class RetrievalStatsRequest(BaseModel):
    """检索统计请求"""
    include_cache_info: Optional[bool] = Field(default=True, description="包含缓存信息")
    include_filter_info: Optional[bool] = Field(default=True, description="包含过滤器信息")

class LLMConfigRequest(BaseModel):
    """LLM配置请求"""
    api_base: Optional[str] = Field(default=None, description="API基础URL")
    api_key: Optional[str] = Field(default=None, description="API密钥")
    model_name: str = Field(..., description="模型名称")
    max_tokens: Optional[int] = Field(default=2048, description="最大token数")
    temperature: Optional[float] = Field(default=0.1, description="温度参数")

# 生命周期管理
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    logger.info("Starting RAG Service...")
    
    # 启动时的初始化
    try:
        # 启动清理任务
        cleanup_task = asyncio.create_task(periodic_cleanup())
        
        logger.info("RAG Service started successfully")
        yield
        
    except Exception as e:
        logger.error(f"Failed to start RAG Service: {e}")
        raise
    finally:
        # 关闭时的清理
        logger.info("Shutting down RAG Service...")
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass

# 创建FastAPI应用
app = FastAPI(
    title="RAG检索增强服务",
    description="基于LlamaIndex的RAG检索增强服务，支持混合查询和流式对话",
    version="1.0.0",
    lifespan=lifespan
)

# 设置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 设置全局异常处理器
setup_exception_handlers(app)

# 挂载静态文件目录
app.mount("/uploads", StaticFiles(directory=f"{settings.KNOWLEDGE_BASE_DIR}/uploads"), name="uploads")

# 注册认证路由
app.include_router(auth_router)

# 定期清理任务
async def periodic_cleanup():
    """定期清理过期会话"""
    while True:
        try:
            await asyncio.sleep(1800)  # 每30分钟清理一次
            rag_service.cleanup_expired_sessions()
            logger.info("Completed periodic session cleanup")
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in periodic cleanup: {e}")

@app.get("/")
async def root() -> JSONResponse:
    """根路径"""
    return success_response(
        data={
            "service": "RAG Retrieval Service",
            "version": "1.0.0",
            "status": "running"
        },
        message="RAG服务运行正常"
    )
    

@app.get("/health")
async def health_check() -> JSONResponse:
    """健康检查"""
    return success_response(
        data={
            "status": "healthy",
            "loaded_indexes": rag_service.get_loaded_indexes(),
            "active_sessions": len(rag_service.sessions),
            "timestamp": asyncio.get_event_loop().time()
        },
        message="服务健康状态正常"
    )

@app.post("/index/load")
async def load_index(index_id: str) -> JSONResponse:
    """加载向量索引"""
    try:
        success = await rag_service.load_index(index_id)
        if success:
            return success_response(
                data={"index_id": index_id},
                message=f"索引 {index_id} 加载成功"
            )
        else:
            return error_response(
                message="索引未找到或加载失败",
                error_code=ErrorCodes.INDEX_NOT_FOUND,
                status_code=404
            )
            
    except Exception as e:
        logger.error(f"Error loading index {index_id}: {e}")
        return error_response(
            message="加载索引时发生错误",
            error_code=ErrorCodes.INDEX_LOAD_FAILED,
            status_code=500,
            details={"error": str(e)}
        )

@app.get("/index/list")
async def list_loaded_indexes() -> JSONResponse:
    """获取已加载的索引列表"""
    try:
        indexes = rag_service.get_loaded_indexes()
        return success_response(
            data={
                "loaded_indexes": indexes,
                "count": len(indexes)
            },
            message="获取索引列表成功"
        )
        
    except Exception as e:
        logger.error(f"Error listing indexes: {e}")
        return error_response(
            message="获取索引列表时发生错误",
            error_code=ErrorCodes.INTERNAL_ERROR,
            status_code=500,
            details={"error": str(e)}
        )

@app.post("/retrieve")
async def retrieve(request: RetrievalRequest) -> JSONResponse:
    """单索引混合检索"""
    try:
        nodes = await rag_service.hybrid_retrieve(
            index_id=request.index_id,
            query=request.query,
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold
        )
        
        # 转换为可序列化的格式
        results = [
            {
                "content": node.node.text,
                "score": float(node.score),
                "metadata": node.node.metadata
            }
            for node in nodes
        ]
        
        return success_response(
            data={
                "query": request.query,
                "results": results,
                "total_count": len(results)
            },
            message="检索完成"
        )
        
    except Exception as e:
        logger.error(f"Error in retrieval: {e}")
        return error_response(
            message="检索时发生错误",
            error_code=ErrorCodes.INTERNAL_ERROR,
            status_code=500,
            details={"error": str(e)}
        )

@app.post("/retrieve/multi")
async def multi_retrieve(request: MultiIndexRetrievalRequest) -> JSONResponse:
    """多索引混合检索"""
    try:
        nodes = await rag_service.multi_index_retrieve(
            index_ids=request.index_ids,
            query=request.query,
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold
        )
        
        # 转换为可序列化的格式
        results = [
            {
                "content": node.node.text,
                "score": float(node.score),
                "metadata": node.node.metadata
            }
            for node in nodes
        ]
        
        return success_response(
            data={
                "query": request.query,
                "index_ids": request.index_ids,
                "results": results,
                "total_count": len(results)
            },
            message="多索引检索完成"
        )
        
    except Exception as e:
        logger.error(f"Error in multi-index retrieval: {e}")
        return error_response(
            message="多索引检索时发生错误",
            error_code=ErrorCodes.INTERNAL_ERROR,
            status_code=500,
            details={"error": str(e)}
        )

@app.post("/test/recall")
async def test_retrieval_recall(request: RecallTestRequest) -> JSONResponse:
    """测试检索召回率"""
    try:
        results = await rag_service.test_retrieval_recall(
            index_id=request.index_id,
            test_queries=request.test_queries
        )
        
        return success_response(
            data=results,
            message="召回测试完成"
        )
        
    except Exception as e:
        logger.error(f"Error in recall test: {e}")
        return error_response(
            message="召回测试时发生错误",
            error_code=ErrorCodes.INTERNAL_ERROR,
            status_code=500,
            details={"error": str(e)}
        )

@app.post("/chat/session")
async def create_chat_session(
    request: CreateSessionRequest,
    current_user: dict = Depends(get_current_active_user)
) -> JSONResponse:
    """创建聊天会话（支持多索引）"""
    try:
        session_id = await rag_service.create_chat_session(
            user_id=current_user["id"],
            index_ids=request.index_ids,
            session_id=request.session_id,
            load_history=request.load_history
        )
        
        session_info = rag_service.get_session_info(session_id, current_user["id"])
        
        return success_response(
            data={
                "session_id": session_id,
                "index_ids": request.index_ids,
                "session_info": session_info
            },
            message="聊天会话创建成功"
        )
        
    except Exception as e:
        logger.error(f"Error creating chat session: {e}")
        return error_response(
            message="创建聊天会话时发生错误",
            error_code=ErrorCodes.INTERNAL_ERROR,
            status_code=500,
            details={"error": str(e)}
        )

@app.post("/chat")
async def chat(
    request: ChatRequest,
    current_user: dict = Depends(get_current_active_user)
) -> JSONResponse:
    """非流式聊天"""
    try:
        response = await rag_service.chat(
            session_id=request.session_id,
            message=request.message,
            user_id=current_user["id"]
        )
        
        return success_response(
            data=response,
            message="聊天响应成功"
        )
        
    except Exception as e:
        logger.error(f"Error in chat: {e}")
        return error_response(
            message="聊天时发生错误",
            error_code=ErrorCodes.INTERNAL_ERROR,
            status_code=500,
            details={"error": str(e)}
        )

@app.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    current_user: dict = Depends(get_current_active_user)
):
    """流式聊天"""
    try:
        # 自动创建会话：如果 session_id 不存在，自动创建
        session_info = rag_service.get_session_info(request.session_id, current_user["id"])
        if not session_info:
            # 自动创建会话，index_ids 使用所有已加载索引
            all_indexes = rag_service.get_loaded_indexes()
            await rag_service.create_chat_session(
                user_id=current_user["id"],
                index_ids=all_indexes,
                session_id=request.session_id,
                load_history=False
            )
            
        async def generate_response():
            async for token in rag_service.chat_stream(
                session_id=request.session_id,
                message=request.message,
                user_id=current_user["id"]
            ):
                yield f"data: {json.dumps({'token': token})}\n\n"
            source_nodes = []
            try:
                if request.session_id in rag_service.sessions:
                    session_obj = rag_service.sessions[request.session_id]
                    if hasattr(session_obj, 'last_source_nodes'):
                        retrieved_nodes = getattr(session_obj, 'last_source_nodes')
                        if retrieved_nodes is not None and isinstance(retrieved_nodes, list):
                            # Filter source_nodes by score > 0.6
                            source_nodes = [node for node in retrieved_nodes if node.get('score', 0) > 0.6]
                        elif retrieved_nodes is not None:
                            logger.warning(f"last_source_nodes for session {request.session_id} is not a list, type: {type(retrieved_nodes)}")
            except Exception as e_sources:
                logger.error(f"Error retrieving source_nodes for stream session {request.session_id}: {str(e_sources)}")
            if source_nodes:
                yield f"data: {json.dumps({'source_nodes': source_nodes})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        return StreamingResponse(
            generate_response(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream"
            }
        )
    except Exception as e:
        logger.error(f"Error in streaming chat: {e}")
        return error_response(
            message="流式聊天时发生错误",
            error_code=ErrorCodes.INTERNAL_ERROR,
            status_code=500,
        )

@app.get("/chat/session/{session_id}")
async def get_session_info(
    session_id: str,
    current_user: dict = Depends(get_current_active_user)
) -> JSONResponse:
    """获取会话信息"""
    try:
        session_info = rag_service.get_session_info(session_id, current_user["id"])
        if not session_info:
            return success_response(
                message="会话未找到",
                error_code=ErrorCodes.INDEX_NOT_FOUND,
                status_code=404
            )
        
        return success_response(
            data=session_info,
            message="获取会话信息成功"
        )
        
    except Exception as e:
        logger.error(f"Error getting session info: {e}")
        return error_response(
            message="获取会话信息时发生错误",
            error_code=ErrorCodes.INTERNAL_ERROR,
            status_code=500,
            details={"error": str(e)}
        )

@app.get("/chat/sessions")
async def list_sessions(
    current_user: dict = Depends(get_current_active_user),
    db = Depends(get_db)
) -> JSONResponse:
    """获取当前用户的所有会话列表"""
    try:
        # 从数据库获取用户的所有会话
        user_sessions = ChatDAO.get_user_sessions(current_user["id"])
        
        sessions = []
        for db_session in user_sessions:
            # 获取会话详细信息
            session_info = {
                "session_id": db_session.session_id,
                "index_ids": db_session.index_ids,
                "created_at": db_session.created_at.isoformat() if db_session.created_at else None,
                "session_metadata": db_session.session_metadata or {},
                "user_id": str(db_session.user_id)
            }
            
            # 如果会话在内存中，获取更多信息
            if db_session.session_id in rag_service.sessions:
                memory_session = rag_service.sessions[db_session.session_id]
                session_info.update({
                    "is_active": True,
                    "message_count": memory_session.message_count
                })
            else:
                session_info["is_active"] = False
                
            sessions.append(session_info)
        return success_response(
            data={
                "sessions": sessions,
                "count": len(sessions),
                "user_id": str(current_user["id"])
            },
            message="获取会话列表成功"
        )
        
    except Exception as e:
        logger.error(f"Error listing sessions: {e}")
        return error_response(
            message="获取会话列表时发生错误",
            error_code=ErrorCodes.INTERNAL_ERROR,
            status_code=500,
            details={"error": str(e)}
        )

@app.delete("/chat/session/{session_id}")
async def delete_session(
    session_id: str,
    current_user: dict = Depends(get_current_active_user)
) -> JSONResponse:
    """删除会话"""
    try:
        # 验证会话属于当前用户
        db_session = ChatDAO.get_session_by_user(session_id, current_user["id"])
        if not db_session:
            return error_response(
                message="会话未找到",
                error_code=ErrorCodes.INDEX_NOT_FOUND,
                status_code=404
            )
        
        # 从内存中删除会话
        if session_id in rag_service.sessions:
            del rag_service.sessions[session_id]
        
        # 从数据库中删除会话
        success = ChatDAO.delete_user_session(session_id, current_user["id"])
        if not success:
            return error_response(
                message="从数据库删除会话失败",
                error_code=ErrorCodes.INTERNAL_ERROR,
                status_code=500
            )
        
        return success_response(
            data={"session_id": session_id},
            message=f"会话 {session_id} 删除成功"
        )
            
    except Exception as e:
        logger.error(f"Error deleting session: {e}")
        return error_response(
            message="删除会话时发生错误",
            error_code=ErrorCodes.INTERNAL_ERROR,
            status_code=500,
            details={"error": str(e)}
        )

@app.get("/chat/history/{session_id}")
async def get_chat_history(
    session_id: str, 
    limit: Optional[int] = None,
    current_user: dict = Depends(get_current_active_user)
) -> JSONResponse:
    """获取聊天历史"""
    try:
        history = rag_service.get_chat_history(session_id, current_user["id"], limit)
        if history is None:
            return success_response(
                message="会话未找到",
                data={
                    "session_id": session_id,
                    "history": [],
                }
            )
        
        return success_response(
            data={
                "session_id": session_id,
                "history": history,
                "count": len(history)
            },
            message="获取聊天历史成功"
        )
        
    except Exception as e:
        logger.error(f"Error getting chat history: {e}")
        return error_response(
            message="获取聊天历史时发生错误",
            error_code=ErrorCodes.INTERNAL_ERROR,
            status_code=500,
            details={"error": str(e)}
        )

@app.delete("/chat/history/{session_id}")
async def clear_chat_history(
    session_id: str,
    current_user: dict = Depends(get_current_active_user)
) -> JSONResponse:
    """清除会话的聊天历史"""
    try:
        # 验证会话属于当前用户
        db_session = ChatDAO.get_session_by_user(session_id, current_user["id"])
        if not db_session:
            return error_response(
                message="会话未找到",
                error_code=ErrorCodes.INDEX_NOT_FOUND,
                status_code=404
            )
        
        # 清除数据库中的聊天历史
        success = ChatDAO.clear_chat_history(session_id, current_user["id"])
        if not success:
            return error_response(
                message="清除聊天历史失败",
                error_code=ErrorCodes.INTERNAL_ERROR,
                status_code=500
            )
        
        # 清除内存中的聊天历史
        if session_id in rag_service.sessions:
            session_obj = rag_service.sessions[session_id]
            if hasattr(session_obj, 'chat_history'):
                session_obj.chat_history.clear()
            # message_count是只读属性，会自动从数据库获取，无需手动设置
        
        return success_response(
            data={"session_id": session_id},
            message=f"会话 {session_id} 的聊天历史已清除"
        )
        
    except Exception as e:
        logger.error(f"Error clearing chat history: {e}")
        import traceback
        traceback.print_exc()
        return error_response(
            message="清除聊天历史时发生错误",
            error_code=ErrorCodes.INTERNAL_ERROR,
            status_code=500,
            details={"error": str(e)}
        )

@app.post("/chat/history/export")
async def export_chat_history(
    request: ExportHistoryRequest,
    current_user: dict = Depends(get_current_active_user)
) -> JSONResponse:
    """导出聊天历史"""
    try:
        history = rag_service.get_chat_history(request.session_id, current_user["id"])
        if history is None:
            return error_response(
                message="会话未找到",
                error_code=ErrorCodes.INDEX_NOT_FOUND,
                status_code=404
            )
        
        # 如果指定了导出路径，保存到文件
        if request.export_path:
            import json
            from datetime import datetime
            export_data = {
                "session_id": request.session_id,
                "export_time": datetime.utcnow().isoformat(),
                "history": history
            }
            
            with open(request.export_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            return success_response(
                data={
                    "session_id": request.session_id,
                    "export_path": request.export_path,
                    "message_count": len(history)
                },
                message="聊天历史导出成功"
            )
        else:
            # 直接返回历史数据
            return success_response(
                data={
                    "session_id": request.session_id,
                    "history": history,
                    "message_count": len(history)
                },
                message="获取聊天历史成功"
            )
        
    except Exception as e:
        logger.error(f"Error exporting chat history: {e}")
        return error_response(
            message="导出聊天历史时发生错误",
            error_code=ErrorCodes.INTERNAL_ERROR,
            status_code=500,
            details={"error": str(e)}
        )

@app.post("/llm/config")
async def update_llm_config(request: LLMConfigRequest) -> JSONResponse:
    """更新LLM配置"""
    try:
        # 更新全局配置
        if request.api_base:
            settings.LLM_API_BASE = request.api_base
        if request.api_key:
            settings.LLM_API_KEY = request.api_key
        
        settings.LLM_MODEL_NAME = request.model_name
        settings.LLM_MAX_TOKENS = request.max_tokens
        settings.LLM_TEMPERATURE = request.temperature
        
        # 重新初始化LLM
        rag_service._setup_llm()
        
        return success_response(
            data={
                "config": {
                    "api_base": settings.LLM_API_BASE,
                    "model_name": settings.LLM_MODEL_NAME,
                    "max_tokens": settings.LLM_MAX_TOKENS,
                    "temperature": settings.LLM_TEMPERATURE
                }
            },
            message="LLM配置更新成功"
        )
        
    except Exception as e:
        logger.error(f"Error updating LLM config: {e}")
        return error_response(
            message="更新LLM配置时发生错误",
            error_code=ErrorCodes.INTERNAL_ERROR,
            status_code=500,
            details={"error": str(e)}
        )

@app.get("/config")
async def get_config() -> JSONResponse:
    """获取服务配置"""
    return success_response(
        data={
            "retrieval_top_k": settings.RETRIEVAL_TOP_K,
            "rerank_top_k": settings.RERANK_TOP_K,
            "similarity_threshold": settings.SIMILARITY_THRESHOLD,
            "llm_config": {
                "api_base": settings.LLM_API_BASE,
                "model_name": settings.LLM_MODEL_NAME,
                "max_tokens": settings.LLM_MAX_TOKENS,
                "temperature": settings.LLM_TEMPERATURE
            },
            "use_gpu": settings.USE_GPU,
            "vector_store_type": settings.VECTOR_STORE_TYPE
        },
        message="获取服务配置成功"
    )

# ==================== 增强RAG功能API端点 ====================

@app.post("/retrieve/enhanced")
async def enhanced_retrieve(
    request: EnhancedRetrievalRequest,
    current_user: dict = Depends(get_optional_current_user)
) -> JSONResponse:
    """增强检索API"""
    try:
        # 检查增强功能是否可用
        if not hasattr(rag_service, 'enhanced_retrieve'):
            return error_response(
                message="增强检索功能不可用",
                error_code=ErrorCodes.INTERNAL_ERROR,
                status_code=503
            )
        
        nodes = await rag_service.enhanced_retrieve(
            query=request.query,
            top_k=request.top_k,
            enable_metadata_filter=request.enable_metadata_filter,
            enable_intent_classification=request.enable_intent_classification,
            enable_cache=request.enable_cache,
            custom_filters=request.custom_filters
        )
        
        # 转换为可序列化的格式
        results = []
        for node in nodes:
            result = {
                "content": node.node.text,
                "score": float(node.score),
                "metadata": node.node.metadata
            }
            
            # 添加增强信息
            if hasattr(node, 'enhanced_info'):
                result["enhanced_info"] = node.enhanced_info
            
            results.append(result)
        
        return success_response(
            data={
                "query": request.query,
                "results": results,
                "total_count": len(results),
                "enhanced_features": {
                    "metadata_filter_enabled": request.enable_metadata_filter,
                    "intent_classification_enabled": request.enable_intent_classification,
                    "cache_enabled": request.enable_cache
                }
            },
            message="增强检索完成"
        )
        
    except Exception as e:
        logger.error(f"Error in enhanced retrieval: {e}")
        return error_response(
            message="增强检索时发生错误",
            error_code=ErrorCodes.INTERNAL_ERROR,
            status_code=500,
            details={"error": str(e)}
        )

@app.get("/retrieve/stats")
async def get_retrieval_stats(
    include_cache_info: bool = Query(default=True, description="包含缓存信息"),
    include_filter_info: bool = Query(default=True, description="包含过滤器信息"),
    current_user: dict = Depends(get_optional_current_user)
) -> JSONResponse:
    """获取检索统计信息"""
    try:
        # 检查增强功能是否可用
        if not hasattr(rag_service, 'get_retrieval_stats'):
            return error_response(
                message="检索统计功能不可用",
                error_code=ErrorCodes.INTERNAL_ERROR,
                status_code=503
            )
        
        stats = rag_service.get_retrieval_stats()
        
        # 根据参数过滤信息
        if not include_cache_info:
            stats.pop('cache_size', None)
        
        if not include_filter_info:
            stats.pop('available_filters', None)
        
        return success_response(
            data=stats,
            message="获取检索统计信息成功"
        )
        
    except Exception as e:
        logger.error(f"Error getting retrieval stats: {e}")
        return error_response(
            message="获取检索统计信息时发生错误",
            error_code=ErrorCodes.INTERNAL_ERROR,
            status_code=500,
            details={"error": str(e)}
        )

@app.post("/cache/clear")
async def clear_all_caches(
    current_user: dict = Depends(get_current_active_user)
) -> JSONResponse:
    """清空所有缓存"""
    try:
        # 检查清理缓存功能是否可用
        if not hasattr(rag_service, 'clear_all_caches'):
            return error_response(
                message="清理缓存功能不可用",
                error_code=ErrorCodes.INTERNAL_ERROR,
                status_code=503
            )
        
        rag_service.clear_all_caches()
        
        return success_response(
            data={"cleared_at": asyncio.get_event_loop().time()},
            message="所有缓存已清空"
        )
        
    except Exception as e:
        logger.error(f"Error clearing caches: {e}")
        return error_response(
            message="清空缓存时发生错误",
            error_code=ErrorCodes.INTERNAL_ERROR,
            status_code=500,
            details={"error": str(e)}
        )

@app.get("/enhanced/status")
async def get_enhanced_status() -> JSONResponse:
    """获取增强功能状态"""
    try:
        enhanced_available = hasattr(rag_service, 'metadata_filter') and rag_service.metadata_filter is not None
        
        status = {
            "enhanced_features_available": enhanced_available,
            "features": {
                "metadata_filtering": hasattr(rag_service, 'metadata_filter') and rag_service.metadata_filter is not None,
                "intent_classification": hasattr(rag_service, 'intent_classifier') and rag_service.intent_classifier is not None,
                "query_caching": hasattr(rag_service, 'query_cache') and rag_service.query_cache is not None,
                "enhanced_retrieve": hasattr(rag_service, 'enhanced_retrieve')
            }
        }
        
        if enhanced_available:
            # 获取更多详细信息
            if hasattr(rag_service, 'intent_weights'):
                status["supported_intents"] = list(rag_service.intent_weights.keys())
            
            if hasattr(rag_service, 'query_cache') and rag_service.query_cache:
                status["cache_size"] = len(rag_service.query_cache.cache)
        
        return success_response(
            data=status,
            message="获取增强功能状态成功"
        )
        
    except Exception as e:
        logger.error(f"Error getting enhanced status: {e}")
        return error_response(
            message="获取增强功能状态时发生错误",
            error_code=ErrorCodes.INTERNAL_ERROR,
            status_code=500,
            details={"error": str(e)}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "frontend_service:app",
        host=settings.RAG_SERVICE_HOST,
        port=settings.RAG_SERVICE_PORT,
        reload=False
    )