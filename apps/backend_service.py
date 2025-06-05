from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import asyncio
import sys
import os
import shutil
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
from common.response import success_response, error_response, ErrorCodes
from common.exception_handler import setup_exception_handlers
from utils.logging_config import setup_logging, get_logger
from services.vector_store_builder import vector_store_builder
from services.document_parser import document_parser

# 初始化日志系统
setup_logging()
logger = get_logger(__name__)

# Pydantic模型
class ParseConfig(BaseModel):
    """解析配置"""
    parser_type: Optional[str] = Field(default=None, description="解析器类型: 'docling' 或 'mineru'，默认使用系统配置")
    ocr_enabled: Optional[bool] = Field(default=True, description="是否启用OCR")
    ocr_languages: Optional[List[str]] = Field(default=["ch_sim", "en"], description="OCR语言")
    extract_tables: Optional[bool] = Field(default=True, description="是否提取表格")
    extract_images: Optional[bool] = Field(default=True, description="是否提取图片信息")
    save_to_file: Optional[bool] = Field(default=False, description="是否保存结果到文件")
    max_workers: Optional[int] = Field(default=None, description="MinerU解析器的并发数量（仅对MinerU有效）")

class VectorStoreConfig(BaseModel):
    """向量数据库配置"""
    extract_keywords: Optional[bool] = Field(default=True, description="是否提取关键词")
    extract_summary: Optional[bool] = Field(default=True, description="是否提取摘要")
    generate_qa: Optional[bool] = Field(default=True, description="是否生成问答对")
    chunk_size: Optional[int] = Field(default=512, description="文本块大小")
    chunk_overlap: Optional[int] = Field(default=50, description="文本块重叠")

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
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    file_info: Optional[Dict[str, Any]] = None
    processing_logs: Optional[List[Dict[str, Any]]] = None
    parser_type: Optional[str] = None

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
        cleanup_task = asyncio.create_task(periodic_cleanup())
        
        logger.info("Document Service started successfully")
        yield
        
    except Exception as e:
        logger.error(f"Failed to start Document Service: {e}")
        raise
    finally:
        # 关闭时的清理
        logger.info("Shutting down Document Service...")
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass

# 创建FastAPI应用
app = FastAPI(
    title="文档解析服务",
    description="基于Docling的文档解析服务，支持OCR和GPU加速",
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

# 定期清理任务
async def periodic_cleanup():
    """定期清理过期任务"""
    while True:
        try:
            await asyncio.sleep(3600)  # 每小时清理一次
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

@app.post("/parse/upload", response_model=TaskResponse)
async def parse_uploaded_file(
    file: UploadFile = File(...),
    config: Optional[str] = Form(None),
    save_to_file: bool = Form(False),
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
        
        parse_config["save_to_file"] = save_to_file
        
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
    file_path: str,
    config: Optional[ParseConfig] = None,
    parser_type: Optional[str] = None
):
    """解析指定路径的文件"""
    try:
        # 验证文件路径
        if not Path(file_path).exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        # 转换配置
        parse_config = config.dict() if config else {}
        
        # 从配置中获取parser_type，如果参数中没有指定的话
        selected_parser_type = parser_type or parse_config.get("parser_type")
        
        # 启动解析任务
        task_id = await document_parser.parse_document(
            file_path,
            parse_config,
            parser_type=selected_parser_type
        )
        
        return TaskResponse(
            task_id=task_id,
            status="pending",
            message="Document parsing task created successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in parse_file_path: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/parse/status/{task_id}", response_model=TaskStatusResponse)
async def get_parse_status(task_id: str):
    """获取解析任务状态"""
    try:
        task_status = document_parser.get_task_status(task_id)
        if not task_status:
            raise HTTPException(status_code=404, detail="Task not found")
        
        return TaskStatusResponse(**task_status)
        
    except HTTPException:
        raise
    except Exception as e:
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
    directory_path: str,
    config: Optional[VectorStoreConfig] = None
):
    """构建向量数据库"""
    try:
        # 验证目录路径
        if not Path(directory_path).exists():
            raise HTTPException(status_code=404, detail="Directory not found")
        
        if not Path(directory_path).is_dir():
            raise HTTPException(status_code=400, detail="Path is not a directory")
        
        # 转换配置
        build_config = config.dict() if config else {}
        
        # 启动构建任务
        task_id = await vector_store_builder.build_vector_store(
            directory_path,
            build_config
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

@app.get("/vector-store/tasks")
async def get_all_vector_store_tasks():
    """获取所有向量数据库构建任务"""
    try:
        tasks = vector_store_builder.get_all_tasks()
        return {"tasks": tasks, "count": len(tasks)}
        
    except Exception as e:
        logger.error(f"Error in get_all_vector_store_tasks: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/parse/logs/{task_id}")
async def get_task_logs(task_id: str, limit: int = 50):
    """获取任务处理日志"""
    try:
        task_status = document_parser.get_task_status(task_id)
        if not task_status:
            raise HTTPException(status_code=404, detail="Task not found")
        
        processing_logs = task_status.get("processing_logs", [])
        
        # 限制返回的日志数量
        if limit > 0:
            processing_logs = processing_logs[-limit:]
        
        return {
            "task_id": task_id,
            "logs": processing_logs,
            "total_logs": len(task_status.get("processing_logs", [])),
            "returned_logs": len(processing_logs)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_task_logs: {e}")
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
        "chunk_size": settings.CHUNK_SIZE,
        "chunk_overlap": settings.CHUNK_OVERLAP,
        "log_level": settings.LOG_LEVEL,
        "log_dir": settings.LOG_DIR,
        "enable_file_logging": settings.ENABLE_FILE_LOGGING,
        "enable_docling_logging": settings.ENABLE_DOCLING_LOGGING
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend_service:app",
        host=settings.DOC_PARSE_HOST,
        port=settings.DOC_PARSE_PORT,
        reload=True
    )