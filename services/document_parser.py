#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import uuid
import time
import asyncio
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor
import json
import traceback
import os
from datetime import datetime

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TesseractCliOcrOptions
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.backend.docling_parse_backend import DoclingParseDocumentBackend

from config import settings
from utils.logging_config import setup_logging, log_progress, get_logger
from services.mineru_parser import MinerUDocumentParser

# 初始化日志系统
setup_logging()
logger = get_logger(__name__)

class TaskStatus:
    """任务状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class ParseTask:
    """解析任务"""
    def __init__(self, task_id: str, file_path: str, config: Dict[str, Any]):
        self.task_id = task_id
        self.file_path = file_path
        self.config = config
        self.status = TaskStatus.PENDING
        self.progress = 0
        self.current_stage = "初始化"
        self.stage_details = {}
        self.result = None
        self.error = None
        self.created_at = time.time()
        self.started_at = None
        self.completed_at = None
        self.processing_logs = []
        self.file_info = self._get_file_info()
        
        # 记录任务创建日志
        logger.info(f"创建解析任务 {task_id}，文件: {file_path}")
        self._log_progress(0, "任务创建", "解析任务已创建")
    
    def _get_file_info(self) -> Dict[str, Any]:
        """获取文件信息"""
        try:
            file_path = Path(self.file_path)
            stat = file_path.stat()
            return {
                "name": file_path.name,
                "size": stat.st_size,
                "extension": file_path.suffix,
                "created_time": stat.st_ctime,
                "modified_time": stat.st_mtime
            }
        except Exception as e:
            logger.warning(f"获取文件信息失败: {e}")
            return {}
    
    def update_progress(self, progress: int, stage: str, message: str, details: Dict[str, Any] = None):
        """更新任务进度"""
        self.progress = progress
        self.current_stage = stage
        if details:
            self.stage_details.update(details)
        
        # 添加处理日志
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "progress": progress,
            "stage": stage,
            "message": message,
            "details": details or {}
        }
        self.processing_logs.append(log_entry)
        
        # 记录进度日志
        self._log_progress(progress, stage, message, details)
        logger.debug(f"任务 {self.task_id} 进度更新: {progress}% - {stage} - {message}")
    
    def _log_progress(self, progress: int, stage: str, message: str, details: Dict[str, Any] = None):
        """记录进度到专用日志"""
        log_progress(
            task_id=self.task_id,
            progress=progress,
            stage=stage,
            message=message,
            details={
                "file_path": self.file_path,
                "file_info": self.file_info,
                **(details or {})
            }
        )
    
    def add_processing_log(self, level: str, message: str, details: Dict[str, Any] = None):
        """添加处理日志"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "message": message,
            "details": details or {}
        }
        self.processing_logs.append(log_entry)
        
        # 根据级别记录到相应的日志器
        log_method = getattr(logger, level.lower(), logger.info)
        log_method(f"任务 {self.task_id}: {message}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "file_path": self.file_path,
            "status": self.status,
            "progress": self.progress,
            "current_stage": self.current_stage,
            "stage_details": self.stage_details,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "file_info": self.file_info,
            "processing_logs": self.processing_logs[-10:] if len(self.processing_logs) > 10 else self.processing_logs  # 只返回最近10条日志
        }

class DocumentParser:
    """文档解析器 - 基于Docling的单例实现"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            logger.info("开始初始化DocumentParser...")
            self.tasks: Dict[str, ParseTask] = {}
            self.executor = ThreadPoolExecutor(max_workers=settings.MAX_CONCURRENT_TASKS)
            self.converter = None
            
            # 初始化MinerU解析器
            self.mineru_parser = MinerUDocumentParser()
            
            # 使用默认配置初始化转换器
            self._setup_converter()
            self._initialized = True
            logger.info(f"DocumentParser初始化完成 - 最大并发任务数: {settings.MAX_CONCURRENT_TASKS}")
            logger.info(f"支持的文件格式: {settings.SUPPORTED_FORMATS}")
            logger.info(f"最大文件大小: {settings.MAX_FILE_SIZE / (1024*1024):.1f}MB")
            logger.info(f"默认解析器类型: {settings.DEFAULT_PARSER_TYPE}")
            logger.info(f"OCR启用状态: {settings.OCR_ENABLED}")
            if settings.OCR_ENABLED:
                logger.info(f"OCR语言: {settings.OCR_LANGUAGES}")
                logger.info(f"GPU使用状态: {settings.USE_GPU}")
    
    def _setup_converter(self, config: Optional[Dict[str, Any]] = None):
        """设置文档转换器"""
        try:
            logger.info("开始配置Docling文档转换器...")
            
            # 从config中获取配置，如果没有则使用默认设置
            ocr_enabled = config.get("ocr_enabled", settings.OCR_ENABLED) if config else settings.OCR_ENABLED
            ocr_languages = config.get("ocr_languages", settings.OCR_LANGUAGES) if config else settings.OCR_LANGUAGES
            
            # 配置PDF处理选项
            pdf_options = PdfPipelineOptions()
            pdf_options.do_ocr = ocr_enabled
            pdf_options.do_table_structure = True
            pdf_options.table_structure_options.do_cell_matching = True
            pdf_options.artifacts_path = settings.DOCLING_MODEL_PATH
            logger.info(f"PDF处理选项配置: OCR={ocr_enabled}, 表格结构提取=True")
            
            # 如果启用OCR，配置OCR后端
            if ocr_enabled:
                logger.info("配置EasyOCR引擎...")
                # 使用EasyOCR引擎避免Tesseract OSD相关问题
                from docling.datamodel.pipeline_options import EasyOcrOptions
                pdf_options.ocr_options = EasyOcrOptions(
                    lang=ocr_languages,
                    force_full_page_ocr=True,  # 强制全页OCR
                    use_gpu=settings.USE_GPU, 
                    download_enabled=False,  # 启用模型下载以支持OCR功能
                    model_storage_directory=settings.EASY_ORC_MODEL_PATH
                )
                logger.info(f"EasyOCR配置完成 - 语言: {ocr_languages}, GPU: {settings.USE_GPU}")
                logger.info(f"EasyOCR模型路径: {settings.EASY_ORC_MODEL_PATH}")
            else:
                logger.info(f"非OCR模式, 跳过OCR配置")
            
            # 创建转换器
            logger.info("创建DocumentConverter实例...")
            self.converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options),
                }
            )
            
            logger.info(f"DocumentConverter创建成功 - OCR: {ocr_enabled}, GPU: {settings.USE_GPU}")
            logger.info("Docling文档转换器配置完成")
            
        except Exception as e:
            logger.error(f"初始化DocumentConverter失败: {e}")
            logger.error(f"错误详情: {traceback.format_exc()}")
            raise
    
    async def parse_document(
        self, 
        file_path: str, 
        config: Optional[Dict[str, Any]] = None,
        parser_type: Optional[str] = None
    ) -> str:
        """解析文档并返回任务ID
        
        Args:
            file_path: 文档文件路径
            config: 解析配置
            parser_type: 解析器类型 ('docling' 或 'mineru')，默认使用配置中的默认值
        
        Returns:
            任务ID
        """
        
        # 确定使用的解析器类型
        selected_parser = parser_type or config.get("parser_type") if config else settings.DEFAULT_PARSER_TYPE
        if selected_parser not in ["docling", "mineru"]:
            raise ValueError(f"Unsupported parser type: {selected_parser}. Must be 'docling' or 'mineru'")
        
        # 验证文件
        if not Path(file_path).exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_size = Path(file_path).stat().st_size
        if file_size > settings.MAX_FILE_SIZE:
            raise ValueError(f"File size {file_size} exceeds maximum {settings.MAX_FILE_SIZE}")
        
        file_ext = Path(file_path).suffix.lower()
        
        # 根据解析器类型验证文件格式
        if selected_parser == "mineru":
            # MinerU主要支持PDF文件
            if file_ext != ".pdf":
                raise ValueError(f"MinerU parser only supports PDF files, got: {file_ext}")
        else:
            # Docling支持多种格式
            if file_ext not in settings.SUPPORTED_FORMATS:
                raise ValueError(f"Unsupported file format for Docling parser: {file_ext}")
        
        # 如果使用MinerU解析器，直接委托给MinerU解析器
        if selected_parser == "mineru":
            logger.info(f"Using MinerU parser for file: {file_path}")
            return await self.mineru_parser.parse_document(file_path, config)
        
        # 使用Docling解析器
        logger.info(f"Using Docling parser for file: {file_path}")
        
        # 创建任务
        task_id = str(uuid.uuid4())
        task_config = config or {}
        task_config["parser_type"] = selected_parser
        
        task = ParseTask(task_id, file_path, task_config)
        self.tasks[task_id] = task
        
        # 异步执行解析
        asyncio.create_task(self._execute_parse_task(task))
        
        logger.info(f"Created Docling parse task {task_id} for file: {file_path}")
        return task_id
    
    async def _execute_parse_task(self, task: ParseTask):
        """执行解析任务"""
        try:
            task.status = TaskStatus.RUNNING
            task.started_at = time.time()
            task.update_progress(5, "任务启动", "开始执行解析任务")
            
            logger.info(f"开始执行解析任务 {task.task_id}，文件: {task.file_path}")
            task.add_processing_log("info", "任务开始执行", {
                "file_size": task.file_info.get("size", 0),
                "file_extension": task.file_info.get("extension", "")
            })
            
            # 文件预检查
            task.update_progress(10, "文件检查", "验证文件有效性")
            await self._validate_file(task)
            
            # 在线程池中执行解析
            task.update_progress(15, "准备解析", "提交到解析线程池")
            loop = asyncio.get_event_loop()
            
            logger.debug(f"任务 {task.task_id} 提交到线程池执行")
            result = await loop.run_in_executor(
                self.executor, 
                self._parse_file_sync, 
                task
            )
            
            task.update_progress(85, "解析完成", "文档解析成功完成")
            
            # 保存结果到文件（如果需要）
            if task.config.get("save_to_file", False):
                task.update_progress(90, "保存结果", "保存解析结果到文件")
                output_path = await self._save_result_to_file(task.task_id, result)
                task.add_processing_log("info", "结果已保存到文件", {"output_path": output_path})
                
                # 保存完成后，移除详细信息，只保留精简的元数据
                simplified_result = {
                    "document_id": result["document_id"],
                    "title": result["title"],
                    "file_type": result["file_type"],
                    "page_count": result["page_count"],
                    "content_length": result["content_length"],
                    "parsed_at": result["parsed_at"],
                    "has_tables": result["has_tables"],
                    "has_images": result["has_images"],
                    "output_file": output_path
                }
                result = simplified_result
            
            # 计算处理时间
            processing_time = time.time() - task.started_at
            
            task.result = result
            task.status = TaskStatus.COMPLETED
            task.update_progress(100, "任务完成", "解析任务成功完成", {
                "processing_time": processing_time,
                "content_length": len(result.get("content", "")),
                "tables_count": len(result.get("tables", [])),
                "images_count": len(result.get("images", []))
            })
            task.completed_at = time.time()
            
            logger.info(f"解析任务 {task.task_id} 成功完成，耗时: {processing_time:.2f}秒")
            task.add_processing_log("info", "任务执行完成", {
                "processing_time": processing_time,
                "final_status": "success"
            })
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = time.time()
            processing_time = task.completed_at - (task.started_at or task.created_at)
            
            task.update_progress(task.progress, "任务失败", f"解析失败: {str(e)}", {
                "error_type": type(e).__name__,
                "processing_time": processing_time
            })
            
            logger.error(f"解析任务 {task.task_id} 失败: {e}")
            logger.error(f"错误详情: {traceback.format_exc()}")
            task.add_processing_log("error", "任务执行失败", {
                "error_message": str(e),
                "error_type": type(e).__name__,
                "traceback": traceback.format_exc()
            })
    
    def _parse_file_sync(self, task: ParseTask) -> Dict[str, Any]:
        """同步解析文件"""
        try:
            # 根据任务配置重新设置转换器
            task.update_progress(15, "配置转换器", "根据任务配置设置文档转换器")
            if task.config:
                logger.info(f"使用任务配置重新设置转换器: {task.config}")
                self._setup_converter(task.config)
                task.add_processing_log("info", "转换器配置更新", {
                    "task_config": task.config
                })
            
            # 开始文档转换
            task.update_progress(20, "文档加载", "开始加载文档文件")
            logger.info(f"开始转换文档: {task.file_path}")
            
            # 获取当前配置用于日志记录
            current_ocr_enabled = task.config.get("ocr_enabled", settings.OCR_ENABLED) if task.config else settings.OCR_ENABLED
            current_ocr_languages = task.config.get("ocr_languages", settings.OCR_LANGUAGES) if task.config else settings.OCR_LANGUAGES
            
            task.add_processing_log("info", "开始Docling文档转换", {
                "converter_config": {
                    "ocr_enabled": current_ocr_enabled,
                    "ocr_languages": current_ocr_languages,
                    "gpu_enabled": settings.USE_GPU
                }
            })
            
            # 执行文档转换
            task.update_progress(30, "文档转换", "执行Docling文档转换")
            start_time = time.time()
            
            logger.debug(f"调用Docling转换器处理文件: {task.file_path}")
            result = self.converter.convert(task.file_path)
            
            conversion_time = time.time() - start_time
            task.update_progress(60, "转换完成", "Docling转换完成", {
                "conversion_time": conversion_time
            })
            
            logger.info(f"Docling转换完成，耗时: {conversion_time:.2f}秒")
            task.add_processing_log("info", "Docling转换完成", {
                "conversion_time": conversion_time,
                "result_type": type(result).__name__
            })
            
            # 提取文档内容
            task.update_progress(70, "内容提取", "提取文档内容和元数据")
            document = result.document
            
            logger.debug(f"开始提取文档内容，文档类型: {type(document).__name__}")
            task.add_processing_log("debug", "开始提取文档内容", {
                "document_type": type(document).__name__,
                "has_pages": hasattr(document, 'pages')
            })
            
            # 构建返回结果
            task.update_progress(75, "数据提取", "提取文档标题和内容")
            
            # 提取基本信息
            document_title = getattr(document, 'title', Path(task.file_path).stem)
            logger.debug(f"文档标题: {document_title}")
            
            # 导出markdown内容
            logger.debug("开始导出Markdown内容")
            content_start_time = time.time()
            markdown_content = document.export_to_markdown()
            content_export_time = time.time() - content_start_time
            
            logger.info(f"Markdown内容导出完成，耗时: {content_export_time:.2f}秒，内容长度: {len(markdown_content)}")
            task.add_processing_log("info", "Markdown内容导出完成", {
                "content_length": len(markdown_content),
                "export_time": content_export_time
            })
            
            # 提取元数据
            task.update_progress(78, "元数据提取", "提取文档元数据")
            page_count = len(document.pages) if hasattr(document, 'pages') else 1
            file_size = Path(task.file_path).stat().st_size
            
            logger.debug(f"文档元数据 - 页数: {page_count}, 文件大小: {file_size}")
            
            # 提取表格
            task.update_progress(80, "表格提取", "提取文档中的表格")
            tables = self._extract_tables(document, task)
            
            # 提取图片
            task.update_progress(82, "图片提取", "提取文档中的图片信息")
            images = self._extract_images(document, task)
            
            # 跳过文档结构提取
            task.update_progress(85, "结构提取", "跳过文档结构提取")
            
            # 保存原始解析结果到文件（用于后处理）
            task.update_progress(87, "文件保存", "保存原始解析结果")
            output_dir = os.path.join(settings.OUTPUT_DIR, task.task_id)
            os.makedirs(output_dir, exist_ok=True)
            
            # 保存原始markdown文件
            original_md_path = os.path.join(output_dir, "content.md")
            with open(original_md_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            
            # 构建content_list.json格式的数据（简化版本，主要包含文本内容）
            content_list = []
            # 将markdown内容按段落分割并构建content_list
            lines = markdown_content.split('\n')
            for i, line in enumerate(lines):
                if line.strip():
                    content_list.append({
                        "type": "text",
                        "text": line.strip(),
                        "text_level": 1 if line.startswith('#') else 0,
                        "page_idx": 1
                    })
            
            # 保存content_list.json
            content_list_path = os.path.join(output_dir, "content_list.json")
            with open(content_list_path, 'w', encoding='utf-8') as f:
                json.dump(content_list, f, ensure_ascii=False, indent=2)
            
            # 使用DocumentPostProcessor进行后处理
            task.update_progress(90, "后处理", "使用DocumentPostProcessor增强结构化")
            from .document_post_processor import DocumentPostProcessor
            post_processor = DocumentPostProcessor()
            enhanced_result = post_processor.process_document(content_list_path, output_dir)
            
            # 读取增强后的内容
            if enhanced_result and "single_file" in enhanced_result:
                enhanced_files = enhanced_result["single_file"]
                if "markdown" in enhanced_files and os.path.exists(enhanced_files["markdown"]):
                    with open(enhanced_files["markdown"], 'r', encoding='utf-8') as f:
                        enhanced_markdown = f.read()
                    # 使用增强后的内容
                    markdown_content = enhanced_markdown
                    logger.info("已使用增强处理后的markdown内容")
            
            # 构建精简的元数据结果
            parsed_result = {
                "document_id": str(uuid.uuid4()),
                "title": document_title,
                "file_type": Path(task.file_path).suffix,
                "page_count": page_count,
                "content_length": len(markdown_content),
                "parsed_at": time.time(),
                "has_tables": len(tables) > 0,
                "has_images": len(images) > 0,
                "parser_type": "docling",
                "enhanced": enhanced_result is not None
            }
            
            # 如果需要保存到文件，则包含完整信息用于保存
            if task.config.get("save_to_file", False):
                parsed_result.update({
                    "file_path": task.file_path,
                    "content": markdown_content,
                    "metadata": {
                        "title": document_title,
                        "file_type": Path(task.file_path).suffix,
                        "page_count": page_count,
                        "content_length": len(markdown_content),
                        "parsed_at": time.time(),
                        "has_tables": len(tables) > 0,
                        "has_images": len(images) > 0,
                        "parser_type": "docling",
                        "output_directory": output_dir,
                        "enhanced": enhanced_result is not None
                    },
                    "tables": tables,
                    "images": images
                })
                
                # 不再包含文档结构信息
            
            logger.info(f"文档解析完成 - 页数: {page_count}, 表格: {len(tables)}, 图片: {len(images)}")
            task.add_processing_log("info", "文档解析结果构建完成", {
                "page_count": page_count,
                "tables_count": len(tables),
                "images_count": len(images),
                "total_content_length": len(markdown_content)
            })
            
            return parsed_result
            
        except Exception as e:
            logger.error(f"解析文件 {task.file_path} 时出错: {e}")
            logger.error(f"错误详情: {traceback.format_exc()}")
            task.add_processing_log("error", "文件解析过程出错", {
                "error_message": str(e),
                "error_type": type(e).__name__,
                "traceback": traceback.format_exc()
            })
            raise
    
    async def _validate_file(self, task: ParseTask):
        """验证文件有效性"""
        try:
            file_path = Path(task.file_path)
            
            # 检查文件是否存在
            if not file_path.exists():
                raise FileNotFoundError(f"文件不存在: {task.file_path}")
            
            # 检查文件大小
            file_size = file_path.stat().st_size
            if file_size == 0:
                raise ValueError(f"文件为空: {task.file_path}")
            
            if file_size > settings.MAX_FILE_SIZE:
                raise ValueError(f"文件大小 {file_size} 超过最大限制 {settings.MAX_FILE_SIZE}")
            
            # 检查文件格式
            file_ext = file_path.suffix.lower()
            if file_ext not in settings.SUPPORTED_FORMATS:
                raise ValueError(f"不支持的文件格式: {file_ext}")
            
            logger.debug(f"文件验证通过: {task.file_path}")
            task.add_processing_log("debug", "文件验证通过", {
                "file_size": file_size,
                "file_extension": file_ext
            })
            
        except Exception as e:
            logger.error(f"文件验证失败: {e}")
            task.add_processing_log("error", "文件验证失败", {"error": str(e)})
            raise
    
    def _extract_tables(self, document, task: ParseTask) -> List[Dict[str, Any]]:
        """提取表格数据"""
        tables = []
        try:
            if hasattr(document, 'tables'):
                logger.debug(f"发现 {len(document.tables)} 个表格")
                task.add_processing_log("debug", f"开始提取 {len(document.tables)} 个表格")
                
                for i, table in enumerate(document.tables):
                    try:
                        table_data = {}
                        if hasattr(table, 'export_to_dataframe'):
                            df = table.export_to_dataframe()
                            table_data = df.to_dict() if df is not None else {}
                            logger.debug(f"表格 {i} 导出成功，形状: {df.shape if df is not None else 'N/A'}")
                        
                        table_info = {
                            "table_id": i,
                            "data": table_data,
                            "caption": getattr(table, 'caption', ''),
                            "row_count": len(table_data.get('index', [])) if table_data else 0,
                            "col_count": len([k for k in table_data.keys() if k != 'index']) if table_data else 0
                        }
                        tables.append(table_info)
                        
                    except Exception as table_error:
                        logger.warning(f"提取表格 {i} 时出错: {table_error}")
                        task.add_processing_log("warning", f"表格 {i} 提取失败", {"error": str(table_error)})
                        
                logger.info(f"成功提取 {len(tables)} 个表格")
                task.add_processing_log("info", f"表格提取完成，共 {len(tables)} 个表格")
            else:
                logger.debug("文档中未发现表格")
                task.add_processing_log("debug", "文档中未发现表格")
                
        except Exception as e:
            logger.warning(f"提取表格时出错: {e}")
            task.add_processing_log("warning", "表格提取过程出错", {"error": str(e)})
        return tables
    
    def _extract_images(self, document, task: ParseTask) -> List[Dict[str, Any]]:
        """提取图片信息"""
        images = []
        try:
            if hasattr(document, 'pictures'):
                logger.debug(f"发现 {len(document.pictures)} 个图片")
                task.add_processing_log("debug", f"开始提取 {len(document.pictures)} 个图片信息")
                
                for i, picture in enumerate(document.pictures):
                    try:
                        image_info = {
                            "image_id": i,
                            "caption": getattr(picture, 'caption', ''),
                            "size": getattr(picture, 'size', {}),
                            "position": getattr(picture, 'position', {}),
                            "type": getattr(picture, 'type', 'unknown')
                        }
                        images.append(image_info)
                        logger.debug(f"图片 {i} 信息提取成功")
                        
                    except Exception as image_error:
                        logger.warning(f"提取图片 {i} 信息时出错: {image_error}")
                        task.add_processing_log("warning", f"图片 {i} 信息提取失败", {"error": str(image_error)})
                        
                logger.info(f"成功提取 {len(images)} 个图片信息")
                task.add_processing_log("info", f"图片信息提取完成，共 {len(images)} 个图片")
            else:
                logger.debug("文档中未发现图片")
                task.add_processing_log("debug", "文档中未发现图片")
                
        except Exception as e:
            logger.warning(f"提取图片信息时出错: {e}")
            task.add_processing_log("warning", "图片信息提取过程出错", {"error": str(e)})
        return images
    
    async def _save_result_to_file(self, task_id: str, result: Dict[str, Any]) -> str:
        """保存解析结果到文件"""
        try:
            output_dir = Path(settings.OUTPUT_DIR) / task_id
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存markdown内容
            md_file = output_dir / "content.md"
            with open(md_file, 'w', encoding='utf-8') as f:
                f.write(result['content'])
            
            # 保存结果JSON
            json_file = output_dir / "result.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Results saved to {output_dir}")
            return str(output_dir)
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            raise
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务状态"""
        # 首先检查Docling任务
        task = self.tasks.get(task_id)
        if task:
            task_dict = task.to_dict()
            task_dict["parser_type"] = "docling"
            return task_dict
        
        # 检查MinerU任务
        mineru_status = self.mineru_parser.get_task_status(task_id)
        if mineru_status:
            mineru_status["parser_type"] = "mineru"
            return mineru_status
        
        return None
    
    def get_all_tasks(self) -> List[Dict[str, Any]]:
        """获取所有任务状态"""
        return [task.to_dict() for task in self.tasks.values()]
    
    def get_task_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务结果"""
        # 首先检查Docling任务
        task = self.tasks.get(task_id)
        if task and task.status == TaskStatus.COMPLETED:
            result = task.result.copy() if task.result else {}
            result["parser_type"] = "docling"
            return result
        
        # 检查MinerU任务
        mineru_result = self.mineru_parser.get_task_result(task_id)
        if mineru_result:
            mineru_result["parser_type"] = "mineru"
            return mineru_result
        
        return None
    
    def cleanup_expired_tasks(self):
        """清理过期任务"""
        current_time = time.time()
        expired_tasks = [
            task_id for task_id, task in self.tasks.items()
            if current_time - task.created_at > settings.TASK_EXPIRE_TIME
        ]
        
        for task_id in expired_tasks:
            del self.tasks[task_id]
            logger.info(f"Cleaned up expired task: {task_id}")
    
    def cleanup_task(self, task_id: str) -> bool:
        """清理任务"""
        # 尝试清理Docling任务
        if task_id in self.tasks:
            del self.tasks[task_id]
            logger.info(f"清理Docling任务: {task_id}")
            return True
        
        # 尝试清理MinerU任务
        if self.mineru_parser.cleanup_task(task_id):
            return True
        
        return False

# 全局解析器实例
document_parser = DocumentParser()