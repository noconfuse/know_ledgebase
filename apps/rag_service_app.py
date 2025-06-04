from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
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

# 配置日志
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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

class RecallTestRequest(BaseModel):
    """召回测试请求"""
    index_id: str = Field(..., description="索引ID")
    test_queries: List[Dict[str, Any]] = Field(..., description="测试查询列表")

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
async def root():
    """根路径"""
    return {
        "service": "RAG Retrieval Service",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "loaded_indexes": rag_service.get_loaded_indexes(),
        "active_sessions": len(rag_service.sessions),
        "timestamp": asyncio.get_event_loop().time()
    }

@app.post("/index/load")
async def load_index(index_id: str):
    """加载向量索引"""
    try:
        success = await rag_service.load_index(index_id)
        if success:
            return {
                "message": f"Index {index_id} loaded successfully",
                "index_id": index_id
            }
        else:
            raise HTTPException(status_code=404, detail="Index not found or failed to load")
            
    except Exception as e:
        logger.error(f"Error loading index {index_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/index/unload")
async def unload_index(index_id: str):
    """卸载向量索引"""
    try:
        success = rag_service.unload_index(index_id)
        if success:
            return {
                "message": f"Index {index_id} unloaded successfully",
                "index_id": index_id
            }
        else:
            raise HTTPException(status_code=404, detail="Index not found")
            
    except Exception as e:
        logger.error(f"Error unloading index {index_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/index/list")
async def list_loaded_indexes():
    """获取已加载的索引列表"""
    try:
        indexes = rag_service.get_loaded_indexes()
        return {
            "loaded_indexes": indexes,
            "count": len(indexes)
        }
        
    except Exception as e:
        logger.error(f"Error listing indexes: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/retrieve")
async def retrieve(request: RetrievalRequest):
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
        
        return {
            "query": request.query,
            "results": results,
            "total_count": len(results)
        }
        
    except Exception as e:
        logger.error(f"Error in retrieval: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/retrieve/multi")
async def multi_retrieve(request: MultiIndexRetrievalRequest):
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
        
        return {
            "query": request.query,
            "index_ids": request.index_ids,
            "results": results,
            "total_count": len(results)
        }
        
    except Exception as e:
        logger.error(f"Error in multi-index retrieval: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/test/recall")
async def test_retrieval_recall(request: RecallTestRequest):
    """测试检索召回率"""
    try:
        results = await rag_service.test_retrieval_recall(
            index_id=request.index_id,
            test_queries=request.test_queries
        )
        
        return results
        
    except Exception as e:
        logger.error(f"Error in recall test: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/session")
async def create_chat_session(request: CreateSessionRequest):
    """创建聊天会话（支持多索引）"""
    try:
        session_id = await rag_service.create_chat_session(
            index_ids=request.index_ids,
            session_id=request.session_id,
            load_history=request.load_history
        )
        
        session_info = rag_service.get_session_info(session_id)
        
        return {
            "session_id": session_id,
            "index_ids": request.index_ids,
            "session_info": session_info,
            "message": "Chat session created successfully"
        }
        
    except Exception as e:
        logger.error(f"Error creating chat session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat(request: ChatRequest):
    """非流式聊天"""
    try:
        response = await rag_service.chat(
            session_id=request.session_id,
            message=request.message
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error in chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """流式聊天"""
    try:
        async def generate_response():
            async for token in rag_service.chat_stream(
                session_id=request.session_id,
                message=request.message
            ):
                # 使用Server-Sent Events格式
                yield f"data: {json.dumps({'token': token})}\n\n"
            
            # 发送结束标记
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
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chat/session/{session_id}")
async def get_session_info(session_id: str):
    """获取会话信息"""
    try:
        session_info = rag_service.get_session_info(session_id)
        if not session_info:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return session_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chat/sessions")
async def list_sessions():
    """获取所有会话列表"""
    try:
        sessions = []
        for session_id in rag_service.sessions.keys():
            session_info = rag_service.get_session_info(session_id)
            if session_info:
                sessions.append(session_info)
        
        return {
            "sessions": sessions,
            "count": len(sessions)
        }
        
    except Exception as e:
        logger.error(f"Error listing sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/chat/session/{session_id}")
async def delete_session(session_id: str):
    """删除会话"""
    try:
        if session_id in rag_service.sessions:
            del rag_service.sessions[session_id]
            return {
                "message": f"Session {session_id} deleted successfully",
                "session_id": session_id
            }
        else:
            raise HTTPException(status_code=404, detail="Session not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chat/history/{session_id}")
async def get_chat_history(session_id: str, limit: Optional[int] = None):
    """获取聊天历史"""
    try:
        history = rag_service.get_chat_history(session_id, limit)
        if history is None:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return {
            "session_id": session_id,
            "history": history,
            "count": len(history)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting chat history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/history/export")
async def export_chat_history(request: ExportHistoryRequest):
    """导出聊天历史"""
    try:
        export_path = rag_service.export_chat_history(
            session_id=request.session_id,
            export_path=request.export_path
        )
        
        if export_path is None:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return {
            "session_id": request.session_id,
            "export_path": export_path,
            "message": "Chat history exported successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting chat history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/llm/config")
async def update_llm_config(request: LLMConfigRequest):
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
        
        return {
            "message": "LLM configuration updated successfully",
            "config": {
                "api_base": settings.LLM_API_BASE,
                "model_name": settings.LLM_MODEL_NAME,
                "max_tokens": settings.LLM_MAX_TOKENS,
                "temperature": settings.LLM_TEMPERATURE
            }
        }
        
    except Exception as e:
        logger.error(f"Error updating LLM config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/config")
async def get_config():
    """获取服务配置"""
    return {
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
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "rag_service_app:app",
        host=settings.RAG_SERVICE_HOST,
        port=settings.RAG_SERVICE_PORT,
        reload=True
    )