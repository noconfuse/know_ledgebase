from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List, Union
import asyncio
import sys
import os
import shutil
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from auth.auth_routes import router
from models.parse_task import TaskStatus
from config import settings
from common.response import success_response, error_response, ErrorCodes
from common.exception_handler import setup_exception_handlers
from utils.logging_config import setup_logging, get_logger, logging
from services.vector_store_builder import vector_store_builder
from services.document_parser import document_parser
from services.directory_vectorizer import DirectoryVectorizer
from dao.index_dao import IndexDAO
from dao.chat_dao import ChatDAO
from dao.task_dao import TaskDAO
from common.postgres_vector_store import PostgresVectorStoreBuilder
from models.database import SessionLocal

# 初始化日志系统
setup_logging()
logger = get_logger(__name__)
watchfiles_logger = get_logger('watchfiles.main')
watchfiles_logger.setLevel(logging.WARNING)
# 初始化DAO
index_dao = IndexDAO()
chat_dao = ChatDAO()
task_dao = TaskDAO()

# 初始化服务
directory_vectorizer = DirectoryVectorizer(vector_store_builder)

# Pydantic模型
class ParseConfig(BaseModel):
    """解析配置"""
    parser_type: Optional[str] = Field(default=None, description="解析器类型: 'docling' 或 'mineru'，默认使用系统配置")
    ocr_enabled: Optional[bool] = Field(default=False, description="是否启用OCR")
    ocr_languages: Optional[List[str]] = Field(default=[], description="OCR语言")
    extract_tables: Optional[bool] = Field(default=True, description="是否提取表格")
    extract_images: Optional[bool] = Field(default=True, description="是否提取图片信息")
    # save_to_file参数已移除，默认都保存到文件
    max_workers: Optional[int] = Field(default=None, description="MinerU解析器的并发数量（仅对MinerU有效）")

class ParseFileRequest(BaseModel):
    """解析文件请求"""
    file_path: str = Field(..., description="文件路径")
    config: Optional[ParseConfig] = Field(default=None, description="解析配置")

class ParseDirectoryRequest(BaseModel):
    """解析目录请求"""
    directory_path: str = Field(..., description="目录路径")
    config: Optional[ParseConfig] = Field(default=None, description="解析配置")

class VectorStoreConfig(BaseModel):
    """向量数据库配置"""
    # 基础配置
    chunk_size: Optional[int] = Field(default=800, description="文本块大小")
    chunk_overlap: Optional[int] = Field(default=100, description="文本块重叠")
    
    # 智能元数据提取器配置
    extract_mode: Optional[str] = Field(default="enhanced", description="提取模式：basic（关键词、问答对、title）或enhanced（包含摘要等更多元数据）")
    min_chunk_size_for_summary: Optional[int] = Field(default=500, description="提取摘要的最小chunk大小（仅enhanced模式）")
    min_chunk_size_for_qa: Optional[int] = Field(default=300, description="提取问答对的最小chunk大小")
    max_keywords: Optional[int] = Field(default=5, description="最大关键词数量")
    num_questions: Optional[int] = Field(default=3, description="问答对数量")
    
    # 索引描述配置


class TaskResponse(BaseModel):
    """任务响应"""
    task_id: str
    status: str
    message: str

class TaskStatusResponse(BaseModel):
    """任务状态响应"""
    task_id: str
    status: str
    progress: int
    current_stage: Optional[str] = None
    stage_details: Optional[Dict[str, Any]] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: Union[datetime, float]
    started_at: Optional[Union[datetime, float]] = None
    completed_at: Optional[Union[datetime, float]] = None
    file_info: Optional[Dict[str, Any]] = None
    processing_logs: Optional[List[Dict[str, Any]]] = None
    parser_type: Optional[str] = None

class VectorizeDirectoryRequest(BaseModel):
    directory_path: str
    enhancement_config: Optional[Dict] = None

class VectorStoreRequest(BaseModel):
    """向量化任务请求"""
    parse_task_id: str = Field(..., description="解析任务ID")
    config: Optional[VectorStoreConfig] = Field(default=None, description="向量化配置")

# 生命周期管理
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    logger.info("Starting Document Service...")
    
    # 启动时的初始化
    try:
        # 确保上传目录存在
        Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
        Path(settings.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        
        # 启动清理任务
        # cleanup_task = asyncio.create_task(periodic_cleanup())
        
        logger.info("Document Service started successfully")
        yield
        
    except Exception as e:
        logger.error(f"Failed to start Document Service: {e}")
        raise
    finally:
        # 关闭时的清理
        logger.info("Shutting down Document Service...")
        # cleanup_task.cancel()
        # try:
        #     await cleanup_task
        # except asyncio.CancelledError:
        #     pass

# 创建FastAPI应用
app = FastAPI(
    title="文档解析服务",
    description="文档解析服务，支持OCR和GPU加速",
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

# 挂载静态文件目录
import os
static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# 设置全局异常处理器
setup_exception_handlers(app)

# 定期清理任务
async def periodic_cleanup():
    """定期清理过期任务"""
    while True:
        try:
            await asyncio.sleep(10800)  # 每小时清理一次
            document_parser.cleanup_expired_tasks()
            vector_store_builder.cleanup_expired_tasks()
            logger.info("Completed periodic cleanup")
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in periodic cleanup: {e}")

@app.get("/")
async def root() -> JSONResponse:
    """根路径"""
    return success_response(
        data={
            "service": "Document Parsing Service",
            "version": "1.0.0",
            "status": "running"
        },
        message="文档解析服务运行正常"
    )

@app.get("/health")
async def health_check() -> JSONResponse:
    """健康检查"""
    return success_response(
        data={
            "status": "healthy",
            "timestamp": asyncio.get_event_loop().time()
        },
        message="服务健康状态正常"
    )

@app.post("/vectorize/directory", response_model=TaskResponse)
async def vectorize_directory_endpoint(request: VectorizeDirectoryRequest, background_tasks: BackgroundTasks):
    """Vectorize a directory of documents."""
    if not os.path.isdir(request.directory_path):
        raise HTTPException(status_code=400, detail="Invalid directory path")

    task_id = f"vectorize-dir-{datetime.now().strftime('%Y%m%d%H%M%S')}-{os.path.basename(request.directory_path)}"
    
    background_tasks.add_task(
        directory_vectorizer.vectorize_directory, 
        request.directory_path, 
        request.enhancement_config
    )
    
    return success_response(
        data={
            "task_id": task_id,
            "status": "pending",
            "message": "Directory vectorization task started."
        }
    )

@app.post("/parse/upload", response_model=TaskResponse)
async def parse_uploaded_file(
    file: UploadFile = File(...),
    config: Optional[str] = Form(None),
    # save_to_file参数已移除，默认都保存到文件
    parser_type: Optional[str] = Form(None, description="解析器类型: 'docling' 或 'mineru'")
):
    """解析上传的文件"""
    try:
        # 验证文件类型
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in settings.SUPPORTED_FORMATS:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file format: {file_ext}"
            )
        
        # 验证文件大小
        if file.size > settings.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File size {file.size} exceeds maximum {settings.MAX_FILE_SIZE}"
            )
        
        # 保存上传的文件
        upload_path = Path(settings.UPLOAD_DIR) / file.filename
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 解析配置
        parse_config = {}
        if config:
            try:
                import json
                parse_config = json.loads(config)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid config JSON")
        
        # save_to_file参数已移除，默认都保存到文件
        
        # 启动解析任务
        task_id = await document_parser.parse_document(
            str(upload_path),
            parse_config,
            parser_type=parser_type
        )
        
        return TaskResponse(
            task_id=task_id,
            status="pending",
            message="Document parsing task created successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in parse_uploaded_file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/parse/file", response_model=TaskResponse)
async def parse_file_path(
    request: ParseFileRequest
):
    """解析指定路径的文件"""
    try:
        # 验证文件路径
        if not Path(request.file_path).exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        # 转换配置
        parse_config = request.config.dict() if request.config else {}
        
        # 启动解析任务
        task_id = await document_parser.parse_document(
            request.file_path,
            parse_config,
        )
        
        return TaskResponse(
            task_id=task_id,
            status="pending",
            message="Document parsing task created successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"Error in parse_file_path: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/parse/directory", response_model=TaskResponse)
async def parse_directory(
    request: ParseDirectoryRequest
):
    """解析指定目录的文件"""
    # 确保文件目录存在
    if not Path(request.directory_path).exists():
        raise HTTPException(status_code=404, detail="Directory not found")
    
    try:
        # 转换配置
        parse_config = request.config.dict() if request.config else {}
        
        # 启动解析任务
        task_id = await document_parser.parse_directory(
            request.directory_path,
            parse_config,
        )
        
        return TaskResponse(
            task_id=task_id,
            status="pending",
            message="Document parsing task created successfully"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in parse_directory: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/parse/status/{task_id}", response_model=TaskStatusResponse)
async def get_parse_status(task_id: str):
    """获取解析任务状态"""
    try:
        task_status = document_parser.get_task(task_id)
        if not task_status:
            raise HTTPException(status_code=404, detail="Task not found")
        
        return TaskStatusResponse(**task_status)
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"Error in get_parse_status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/parse/tasks")
async def get_all_parse_tasks():
    """获取所有解析任务"""
    try:
        tasks = document_parser.get_all_tasks()
        return {"tasks": tasks, "count": len(tasks)}
        
    except Exception as e:
        logger.error(f"Error in get_all_parse_tasks: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/vector-store/build", response_model=TaskResponse)
async def build_vector_store(
    request: VectorStoreRequest
):
    """从解析任务构建向量数据库"""
    try:
        # 验证解析任务是否存在
        parse_task = document_parser.get_task(request.parse_task_id)
        if not parse_task:
            raise HTTPException(status_code=404, detail="Parse task not found")
        
        if parse_task.get('status') != TaskStatus.COMPLETED:
            raise HTTPException(status_code=400, detail="Parse task not completed")
        
        # 转换配置
        build_config = request.config.dict() if request.config else {}
        
        # 确保配置包含所有必要的参数
        default_config = {
            "extract_mode": "enhanced",
            "min_chunk_size_for_summary": 500,
            "min_chunk_size_for_qa": 300,
            "max_keywords": 5,
            "num_questions": 3
        }
        
        # 合并默认配置和用户配置
        final_config = {**default_config, **build_config}
        
        # 启动构建任务
        task_id = await vector_store_builder.build_vector_store(
            request.parse_task_id,
            final_config
        )
        
        return TaskResponse(
            task_id=task_id,
            status="pending",
            message="Vector store build task created successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in build_vector_store: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/vector-store/status/{task_id}", response_model=TaskStatusResponse)
async def get_vector_store_status(task_id: str):
    """获取向量数据库构建状态"""
    try:
        task_status = vector_store_builder.get_task_status(task_id)
        if not task_status:
            raise HTTPException(status_code=404, detail="Task not found")
        
        return TaskStatusResponse(**task_status)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_vector_store_status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tasks/parse")
async def list_parse_tasks(limit: int = 100, offset: int = 0, status: str = None):
    """列出解析任务"""
    try:
        tasks = task_dao.list_parse_tasks(limit=limit, offset=offset, status=status)
        return {
            "tasks": [task.to_dict() for task in tasks],
            "total": len(tasks)
        }
    except Exception as e:
        logger.error(f"Error listing parse tasks: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tasks/vector-store")
async def list_vector_store_tasks(limit: int = 100, offset: int = 0, status: str = None, parse_task_id: str = None):
    """列出向量化任务"""
    try:
        tasks = task_dao.list_vector_store_tasks(limit=limit, offset=offset, status=status, parse_task_id=parse_task_id)
        return {
            "tasks": [task.to_dict() for task in tasks],
            "total": len(tasks)
        }
    except Exception as e:
        logger.error(f"Error listing vector store tasks: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tasks/parse/{task_id}")
async def get_parse_task(task_id: str):
    """获取解析任务详情"""
    try:
        task = task_dao.get_parse_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Parse task not found")
        return task.to_dict()
    except Exception as e:
        logger.error(f"Error getting parse task: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tasks/vector-store/{task_id}")
async def get_vector_store_task(task_id: str):
    """获取向量化任务详情"""
    try:
        task = task_dao.get_vector_store_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Vector store task not found")
        return task.to_dict()
    except Exception as e:
        logger.error(f"Error getting vector store task: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/vector-store/tasks")
async def get_all_vector_store_tasks():
    """获取所有向量数据库构建任务"""
    try:
        tasks = vector_store_builder.get_all_tasks()
        return {"tasks": tasks, "count": len(tasks)}
        
    except Exception as e:
        logger.error(f"Error in get_all_vector_store_tasks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/logs/download")
async def download_log_file(log_type: str = "main", date: Optional[str] = None):
    """下载日志文件"""
    try:
        from fastapi.responses import FileResponse
        import os
        from datetime import datetime
        
        log_dir = Path(settings.LOG_DIR)
        
        if log_type == "main":
            if date:
                log_file = log_dir / f"app_{date}.log"
            else:
                log_file = log_dir / "app.log"
        elif log_type == "docling":
            if date:
                log_file = log_dir / f"docling_{date}.log"
            else:
                log_file = log_dir / "docling.log"
        elif log_type == "progress":
            if date:
                log_file = log_dir / f"progress_{date}.log"
            else:
                log_file = log_dir / "progress.log"
        else:
            raise HTTPException(status_code=400, detail="Invalid log type. Use 'main', 'docling', or 'progress'")
        
        if not log_file.exists():
            raise HTTPException(status_code=404, detail="Log file not found")
        
        return FileResponse(
            path=str(log_file),
            filename=log_file.name,
            media_type="text/plain"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in download_log_file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/logs/list")
async def list_log_files():
    """列出可用的日志文件"""
    try:
        log_dir = Path(settings.LOG_DIR)
        
        if not log_dir.exists():
            return {"log_files": [], "message": "Log directory not found"}
        
        log_files = []
        for log_file in log_dir.glob("*.log"):
            stat = log_file.stat()
            log_files.append({
                "name": log_file.name,
                "size": stat.st_size,
                "modified": stat.st_mtime,
                "type": log_file.stem.split("_")[0] if "_" in log_file.stem else log_file.stem
            })
        
        # 按修改时间排序
        log_files.sort(key=lambda x: x["modified"], reverse=True)
        
        return {
            "log_files": log_files,
            "total_files": len(log_files)
        }
        
    except Exception as e:
        logger.error(f"Error in list_log_files: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/config")
async def get_config():
    """获取服务配置"""
    return {
        "max_file_size": settings.MAX_FILE_SIZE,
        "supported_formats": settings.SUPPORTED_FORMATS,
        "ocr_enabled": settings.OCR_ENABLED,
        "use_gpu": settings.USE_GPU,
        "log_level": settings.LOG_LEVEL,
        "log_dir": settings.LOG_DIR,
        "enable_file_logging": settings.ENABLE_FILE_LOGGING,
        "enable_docling_logging": settings.ENABLE_DOCLING_LOGGING
    }

@app.delete("/vector-store/index/{index_id}")
async def delete_vector_index(index_id: str):
    """删除向量索引
    
    删除指定的向量索引，包括：
    1. 从indexes表中删除索引记录
    2. 删除PGVector中对应的数据表
    3. 从内存中卸载索引
    
    Args:
        index_id: 索引ID
        
    Returns:
        删除结果
    """
    try:
        # 创建数据库会话
        db = SessionLocal()
        
        try:
            # 1. 检查索引是否存在
            index_info = IndexDAO.get_index_by_id(db, index_id)
            if not index_info:
                return error_response(
                    message=f"索引 {index_id} 不存在",
                    error_code=ErrorCodes.INDEX_NOT_FOUND,
                    status_code=404
                )
            
            # 2. 从内存中卸载索引（如果已加载）
            try:
                from services.rag_service import RAGService
                rag_service = RAGService()
                rag_service.unload_index(index_id)
                logger.info(f"Unloaded index from memory: {index_id}")
            except Exception as e:
                logger.warning(f"Failed to unload index from memory: {e}")
            
            # 3. 删除PGVector中的数据表（如果使用PostgreSQL向量存储）
            if settings.VECTOR_STORE_TYPE == "postgres":
                try:
                    import psycopg2
                    
                    # 构建表名
                    table_name = f"{settings.POSTGRES_TABLE_NAME}_{index_id.replace('-', '_')}"
                    
                    # 直接连接数据库删除表，避免创建新表
                    conn = psycopg2.connect(
                        host=settings.POSTGRES_HOST,
                        port=settings.POSTGRES_PORT,
                        database=settings.POSTGRES_DATABASE,
                        user=settings.POSTGRES_USER,
                        password=settings.POSTGRES_PASSWORD
                    )
                    
                    cursor = conn.cursor()
                    
                    # 删除表（包括data_前缀的表）
                    cursor.execute(f"DROP TABLE IF EXISTS data_{table_name} CASCADE;")
                    conn.commit()
                    
                    cursor.close()
                    conn.close()
                    
                    logger.info(f"Deleted PostgreSQL vector table: data_{table_name}")
                        
                except Exception as e:
                    logger.error(f"Error deleting PostgreSQL vector table: {e}")
                    # 继续执行，不因为向量表删除失败而中断整个流程
            
            # 4. 删除FAISS索引文件（如果使用FAISS）
            elif settings.VECTOR_STORE_TYPE == "faiss":
                try:
                    index_path = Path(settings.INDEX_STORE_PATH) / index_id
                    if index_path.exists():
                        shutil.rmtree(index_path)
                        logger.info(f"Deleted FAISS index directory: {index_path}")
                except Exception as e:
                    logger.error(f"Error deleting FAISS index directory: {e}")
            
            # 5. 从indexes表中删除索引记录
            if IndexDAO.delete_index(db, index_id):
                logger.info(f"Deleted index record from database: {index_id}")
                
                return success_response(
                    data={"index_id": index_id},
                    message=f"成功删除向量索引 {index_id}"
                )
            else:
                return error_response(
                    message=f"删除索引记录失败: {index_id}",
                    error_code=ErrorCodes.INTERNAL_ERROR,
                    status_code=500
                )
                
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Error in delete_vector_index: {e}")
        return error_response(
            message=f"删除向量索引时发生错误: {str(e)}",
            error_code=ErrorCodes.INTERNAL_ERROR,
            status_code=500
        )


@app.get('/test_collect_documents/{parse_task_id}')
async def test_collect_documents(parse_task_id: str):
    """测试收集文档"""
    # 2. 测试收集文档
    try:
        from services.vector_store_builder import VectorStoreBuilder
        vector_builder = VectorStoreBuilder()
        all_tasks = vector_builder.test_collection_documents(parse_task_id)
        return success_response(
            data={"all_tasks": all_tasks},
            message=f"成功测试收集文档 {len(all_tasks)} 个"
        )
    except Exception as e:
        logger.error(f"Error testing collection documents: {e}")
        return error_response(
            message=f"测试收集文档时发生错误: {str(e)}",
            error_code=ErrorCodes.INTERNAL_ERROR,
            status_code=500
        )

# 管理界面API端点
@app.get("/admin/dashboard")
async def admin_dashboard(
    limit: int = 20,
    offset: int = 0,
    task_type: Optional[str] = None
):
    """管理界面仪表板 - 分页查看解析任务、向量化任务、对话和向量索引"""
    try:
        with SessionLocal() as db:
            # 获取解析任务
            parse_tasks = task_dao.list_parse_tasks(
                limit=limit, 
                offset=offset
            )
            parse_tasks_count = task_dao.get_parse_tasks_count()
            
            # 获取向量化任务
            vector_tasks = task_dao.list_vector_store_tasks(
                limit=limit,
                offset=offset
            )
            vector_tasks_count = task_dao.get_vector_tasks_count()
            
            # 获取对话会话
            chat_sessions = chat_dao.get_user_sessions(
                limit=limit,
                offset=offset
            )
            chat_sessions_count = chat_dao.get_sessions_count()
            
            # 获取向量索引
            vector_indexes = index_dao.get_all_indexes(
                db,
                limit=limit,
                offset=offset
            )
            vector_indexes_count = index_dao.get_indexes_count(db)
            
            return success_response(
                data={
                    "parse_tasks": {
                        "items": [{
                            "task_id": task.task_id,
                            "status": task.status.value if hasattr(task.status, 'value') else str(task.status),
                            "file_path": task.file_path,
                            "parent_task_id": task.parent_task_id,
                            "created_at": task.created_at.isoformat() if task.created_at else None,
                            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
                            "progress": task.progress
                        } for task in parse_tasks],
                        "total": parse_tasks_count,
                        "limit": limit,
                        "offset": offset
                    },
                    "vector_tasks": {
                        "items": [{
                            "task_id": task.task_id,
                            "status": task.status.value if hasattr(task.status, 'value') else str(task.status),
                            "created_at": task.created_at.isoformat() if task.created_at else None,
                            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
                            "progress": task.progress
                        } for task in vector_tasks],
                        "total": vector_tasks_count,
                        "limit": limit,
                        "offset": offset
                    },
                    "chat_sessions": {
                        "items": [{
                            "session_id": session.session_id,
                            "user_id": str(session.user_id),
                            "index_ids": session.index_ids,
                            "created_at": session.created_at.isoformat() if session.created_at else None,
                            "last_activity": session.last_activity.isoformat() if session.last_activity else None,
                            "is_active": session.is_active
                        } for session in chat_sessions],
                        "total": chat_sessions_count,
                        "limit": limit,
                        "offset": offset
                    },
                    "vector_indexes": {
                        "items": [{
                            "index_id": index.index_id,
                            "index_description": index.index_description,
                            "origin_file_path": index.origin_file_path,
                            "document_count": index.document_count,
                            "node_count": index.node_count,
                            "vector_dimension": index.vector_dimension,
                            "created_at": index.created_at.isoformat() if index.created_at else None
                        } for index in vector_indexes],
                        "total": vector_indexes_count,
                        "limit": limit,
                        "offset": offset
                    }
                },
                message="管理界面数据获取成功"
            )
    except Exception as e:
        logger.error(f"Error getting admin dashboard data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/admin/parse-tasks")
async def get_parse_tasks_admin(
    limit: int = 20,
    offset: int = 0
):
    """获取解析任务列表（管理界面）"""
    try:
        tasks = task_dao.list_parse_tasks(limit=limit, offset=offset)
        total = task_dao.get_parse_tasks_count()
        
        return success_response(
            data={
                "items": [{
                    "task_id": task.task_id,
                    "status": task.status.value if hasattr(task.status, 'value') else str(task.status),
                    "file_path": task.file_path,
                    "created_at": task.created_at.isoformat() if task.created_at else None,
                    "started_at": task.started_at.isoformat() if task.started_at else None,
                    "completed_at": task.completed_at.isoformat() if task.completed_at else None,
                    "progress": task.progress,
                    "error": task.error
                } for task in tasks],
                "total": total,
                "limit": limit,
                "offset": offset
            },
            message="解析任务列表获取成功"
        )
    except Exception as e:
        logger.error(f"Error getting parse tasks: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/admin/vector-tasks")
async def get_vector_tasks_admin(
    limit: int = 20,
    offset: int = 0
):
    """获取向量化任务列表（管理界面）"""
    try:
        with SessionLocal() as db:
            tasks = task_dao.list_vector_store_tasks(limit=limit, offset=offset)
            total = task_dao.get_vector_tasks_count()
            
            return success_response(
                data={
                    "items": [{
                        "task_id": task.task_id,
                        "parse_task_id": task.parse_task_id,
                        "index_id": task.index_id,
                        "status": task.status.value if hasattr(task.status, 'value') else str(task.status),
                        "created_at": task.created_at.isoformat() if task.created_at else None,
                        "started_at": task.started_at.isoformat() if task.started_at else None,
                        "completed_at": task.completed_at.isoformat() if task.completed_at else None,
                        "progress": task.progress,
                        "error": task.error
                    } for task in tasks],
                    "total": total,
                    "limit": limit,
                    "offset": offset
                },
                message="向量化任务列表获取成功"
            )
    except Exception as e:
         logger.error(f"Error getting vector tasks: {e}")
         raise HTTPException(status_code=500, detail=str(e))

@app.get("/admin")
async def admin_interface():
    """管理界面"""
    static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
    admin_file = os.path.join(static_dir, "admin.html")
    if os.path.exists(admin_file):
        return FileResponse(admin_file)
    else:
        raise HTTPException(status_code=404, detail="Admin interface not found")

@app.get("/admin/upload-files")
async def get_upload_files():
    """获取uploads目录下的文件列表"""
    try:
        upload_dir = Path(settings.UPLOAD_DIR)
        if not upload_dir.exists():
            return success_response(
                data={"files": [], "directories": []},
                message="上传目录不存在"
            )
        
        files = []
        directories = []
        
        def scan_directory(dir_path: Path, relative_path: str = ""):
            """递归扫描目录"""
            try:
                for item in dir_path.iterdir():
                    if item.is_file():
                        # 计算相对路径
                        rel_path = os.path.join(relative_path, item.name) if relative_path else item.name
                        files.append({
                            "name": item.name,
                            "path": str(item),
                            "relative_path": rel_path,
                            "size": item.stat().st_size,
                            "modified_at": datetime.fromtimestamp(item.stat().st_mtime).isoformat()
                        })
                    elif item.is_dir():
                        # 计算相对路径
                        rel_path = os.path.join(relative_path, item.name) if relative_path else item.name
                        directories.append({
                            "name": item.name,
                            "path": str(item),
                            "relative_path": rel_path
                        })
                        # 递归扫描子目录
                        scan_directory(item, rel_path)
            except PermissionError:
                logger.warning(f"Permission denied accessing {dir_path}")
            except Exception as e:
                logger.error(f"Error scanning directory {dir_path}: {e}")
        
        scan_directory(upload_dir)
        
        return success_response(
            data={
                "files": files,
                "directories": directories,
                "upload_dir": str(upload_dir)
            },
            message="文件列表获取成功"
        )
    except Exception as e:
        logger.error(f"Error getting upload files: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/admin/vector-indexes/{index_id}/description")
async def update_index_description(index_id: str, description: str = Form(...)):
    """更新向量索引描述"""
    try:
        with SessionLocal() as db:
            updated_index = index_dao.update_index_description(db, index_id, description)
            if updated_index:
                return success_response(
                    data={
                        "index_id": index_id,
                        "description": description
                    },
                    message="索引描述更新成功"
                )
            else:
                return error_response(
                    message=f"索引 {index_id} 不存在",
                    error_code=ErrorCodes.INDEX_NOT_FOUND,
                    status_code=404
                )
    except Exception as e:
        logger.error(f"Error updating index description: {e}")
        return error_response(
            message=f"更新索引描述时发生错误: {str(e)}",
            error_code=ErrorCodes.INTERNAL_ERROR,
            status_code=500
        )

@app.post("/admin/chat-sessions/{session_id}/reset-index")
async def reset_session_index(session_id: str, index_ids: List[str] = Form(...)):
    """重置会话索引绑定"""
    try:
        # 验证索引是否存在
        with SessionLocal() as db:
            valid_index_ids = []
            for index_id in index_ids:
                index_exists = index_dao.get_index_by_id(db, index_id)
                if index_exists:
                    valid_index_ids.append(index_id)
                else:
                    logger.warning(f"Index {index_id} does not exist")
            
            if not valid_index_ids:
                return error_response(
                    message="没有有效的索引ID",
                    error_code=ErrorCodes.VALIDATION_ERROR,
                    status_code=400
                )
        
        # 更新会话的索引绑定
        success = chat_dao.update_session_indexes(session_id, valid_index_ids)
        
        if success:
            return success_response(
                data={
                    "session_id": session_id,
                    "index_ids": valid_index_ids
                },
                message="会话索引重置成功"
            )
        else:
            return error_response(
                message=f"会话 {session_id} 不存在或重置失败",
                error_code=ErrorCodes.SESSION_NOT_FOUND,
                status_code=404
            )
    except Exception as e:
        logger.error(f"Error resetting session index: {e}")
        return error_response(
            message=f"重置会话索引时发生错误: {str(e)}",
            error_code=ErrorCodes.INTERNAL_ERROR,
            status_code=500
        )

@app.get("/admin/chat-sessions")
async def get_chat_sessions_admin(
    limit: int = 20,
    offset: int = 0,
    user_id: Optional[str] = None
):
    """获取对话会话列表（管理界面）"""
    try:
        sessions = chat_dao.get_user_sessions(
            user_id=user_id,
            limit=limit,
            offset=offset
        )
        total = chat_dao.get_sessions_count(user_id=user_id)
        
        # 获取用户信息
        session_items = []
        with SessionLocal() as db:
            for session in sessions:
                user_info = None
                if session.user_id:
                    from dao.user_dao import UserDAO
                    user = UserDAO.get_user_by_id(db, str(session.user_id))
                    if user:
                        user_info = {
                            "username": user.username,
                            "email": user.email,
                            "full_name": user.full_name
                        }
                
                session_items.append({
                    "session_id": session.session_id,
                    "user_id": str(session.user_id),
                    "user_info": user_info,
                    "index_ids": session.index_ids,
                    "created_at": session.created_at.isoformat() if session.created_at else None,
                    "last_activity": session.last_activity.isoformat() if session.last_activity else None,
                    "is_active": session.is_active,
                    "session_metadata": session.session_metadata
                })
        
        return success_response(
            data={
                "items": session_items,
                "total": total,
                "limit": limit,
                "offset": offset
            },
            message="对话会话列表获取成功"
        )
    except Exception as e:
        logger.error(f"Error getting chat sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/admin/vector-indexes")
async def get_vector_indexes_admin(
    limit: int = 20,
    offset: int = 0
):
    """获取向量索引列表（管理界面）"""
    try:
        with SessionLocal() as db:
            indexes = index_dao.get_all_indexes(db, limit=limit, offset=offset)
            total = index_dao.get_indexes_count(db)
            
            return success_response(
                data={
                    "items": [{
                        "index_id": index.index_id,
                        "index_description": index.index_description,
                        "origin_file_path": index.origin_file_path,
                        "document_count": index.document_count,
                        "node_count": index.node_count,
                        "vector_dimension": index.vector_dimension,
                        "processing_config": index.processing_config,
                        "created_at": index.created_at.isoformat() if index.created_at else None
                    } for index in indexes],
                    "total": total,
                    "limit": limit,
                    "offset": offset
                },
                message="向量索引列表获取成功"
            )
    except Exception as e:
        logger.error(f"Error getting vector indexes: {e}")
        raise HTTPException(status_code=500, detail=str(e))





if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "backend_service:app",
        host=settings.DOC_PARSE_HOST,
        port=settings.DOC_PARSE_PORT,
        reload=False,
    )