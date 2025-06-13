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
import mimetypes
from datetime import datetime

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TesseractCliOcrOptions
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.backend.docling_parse_backend import DoclingParseDocumentBackend

from config import settings
from utils.logging_config import setup_logging, log_progress, get_logger
from services.mineru_parser import MinerUDocumentParser
from dao.task_dao import TaskDAO

# 初始化日志系统
setup_logging()
logger = get_logger(__name__)

from models.task_models import ParseTask
from models.parse_task import TaskStatus

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
            
            # 初始化任务DAO
            self.task_dao = TaskDAO()
            
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
        
        # 获取文件信息
        file_path_obj = Path(file_path)
        file_stat = file_path_obj.stat()
        
        # 创建数据库任务对象
        task = ParseTask(
            task_id=task_id,
            file_path=file_path,
            file_name=file_path_obj.name,
            file_size=file_stat.st_size,
            file_extension=file_path_obj.suffix,
            parser_type="docling",
            config=task_config,
            status=TaskStatus.PENDING
        )
        self.tasks[task_id] = task
        
        # 保存任务到数据库
        self._save_task_to_db(task)
        
        # 异步执行解析
        asyncio.create_task(self._execute_parse_task(task))
        
        logger.info(f"Created Docling parse task {task_id} for file: {file_path}")
        return task_id
    
    async def _execute_parse_task(self, task: ParseTask):
        """执行解析任务"""
        try:
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.utcnow()
            task.progress = 5
            task.current_stage = "任务启动"
            
            # 更新数据库中的任务状态
            self._update_task_in_db(task)
            
            logger.info(f"开始执行解析任务 {task.task_id}，文件: {task.file_path}")
            
            # 添加处理日志
            if not task.processing_logs:
                task.processing_logs = []
            task.processing_logs.append({
                "timestamp": datetime.now().isoformat(),
                "level": "info",
                "message": "任务开始执行",
                "details": {
                    "file_size": task.file_size,
                    "file_extension": task.file_extension
                }
            })
            
            # 文件预检查
            task.progress = 10
            task.current_stage = "文件检查"
            self._update_task_in_db(task)
            await self._validate_file(task)
            
            # 在线程池中执行解析
            task.progress = 15
            task.current_stage = "准备解析"
            self._update_task_in_db(task)
            loop = asyncio.get_event_loop()
            
            logger.debug(f"任务 {task.task_id} 提交到线程池执行")
            result = await loop.run_in_executor(
                self.executor, 
                self._parse_file_sync, 
                task
            )
            
            task.progress = 85
            task.current_stage = "解析完成"
            self._update_task_in_db(task)
            
            # 默认保存结果到文件
            task.progress = 90
            task.current_stage = "保存结果"
            self._update_task_in_db(task)
            output_path = await self._save_result_to_file(task.task_id, result)
            
            # 保存完成后，添加输出文件路径到结果中
            result["output_file"] = output_path
            
            # 计算处理时间
            processing_time = time.time() - task.started_at.timestamp() if task.started_at else 0
            
            task.result = result
            task.status = TaskStatus.COMPLETED
            task.progress = 100
            task.current_stage = "任务完成"
            if not task.stage_details:
                task.stage_details = {}
            task.stage_details.update({
                "processing_time": processing_time,
                "content_length": len(result.get("content", "")),
                "tables_count": len(result.get("tables", [])),
                "images_count": len(result.get("images", []))
            })
            self._update_task_in_db(task)
            task.completed_at = datetime.utcnow()
            
            # 更新数据库中的任务状态
            self._update_task_in_db(task)
            
            logger.info(f"解析任务 {task.task_id} 成功完成，耗时: {processing_time:.2f}秒")
            
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = datetime.utcnow()
            processing_time = time.time() - (task.started_at.timestamp() if task.started_at else task.created_at.timestamp())
            
            task.current_stage = "任务失败"
            task.error = f"解析失败: {str(e)}"
            if not task.stage_details:
                task.stage_details = {}
            task.stage_details.update({
                "error_type": type(e).__name__,
                "processing_time": processing_time
            })
            self._update_task_in_db(task)
            
            # 更新数据库中的任务状态
            self._update_task_in_db(task)
            
            logger.error(f"解析任务 {task.task_id} 失败: {e}")
            logger.error(f"错误详情: {traceback.format_exc()}")
            
    
    def _parse_file_sync(self, task: ParseTask) -> Dict[str, Any]:
        """同步解析文件"""
        try:
            # 根据任务配置重新设置转换器
            task.progress = 15
            task.current_stage = "配置转换器"
            self._update_task_in_db(task)
            if task.config:
                logger.info(f"使用任务配置重新设置转换器: {task.config}")
                self._setup_converter(task.config)
                
            
            # 开始文档转换
            task.progress = 20
            task.current_stage = "文档加载"
            self._update_task_in_db(task)
            logger.info(f"开始转换文档: {task.file_path}")
            
            # 获取当前配置用于日志记录
            current_ocr_enabled = task.config.get("ocr_enabled", settings.OCR_ENABLED) if task.config else settings.OCR_ENABLED
            current_ocr_languages = task.config.get("ocr_languages", settings.OCR_LANGUAGES) if task.config else settings.OCR_LANGUAGES
            
      
            
            # 执行文档转换
            task.progress = 30
            task.current_stage = "文档转换"
            self._update_task_in_db(task)
            start_time = time.time()
            
            logger.debug(f"调用Docling转换器处理文件: {task.file_path}")
            result = self.converter.convert(task.file_path)
            
            conversion_time = time.time() - start_time
            task.progress = 60
            task.current_stage = "转换完成"
            if not task.stage_details:
                task.stage_details = {}
            task.stage_details.update({
                "conversion_time": conversion_time
            })
            self._update_task_in_db(task)
            
            logger.info(f"Docling转换完成，耗时: {conversion_time:.2f}秒")
           
            
            # 提取文档内容
            task.progress = 70
            task.current_stage = "内容提取"
            self._update_task_in_db(task)
            document = result.document
            
            logger.debug(f"开始提取文档内容，文档类型: {type(document).__name__}")
          
            
            # 构建返回结果
            task.progress = 75
            task.current_stage = "数据提取"
            self._update_task_in_db(task)
            
            # 提取基本信息
            document_title = getattr(document, 'title', Path(task.file_path).stem)
            logger.debug(f"文档标题: {document_title}")
            
            # 导出markdown内容
            logger.debug("开始导出Markdown内容")
            content_start_time = time.time()
            markdown_content = document.export_to_markdown()
            content_export_time = time.time() - content_start_time
            
            logger.info(f"Markdown内容导出完成，耗时: {content_export_time:.2f}秒，内容长度: {len(markdown_content)}")
          
            
            # 提取元数据
            task.progress = 78
            task.current_stage = "元数据提取"
            self._update_task_in_db(task)
            page_count = len(document.pages) if hasattr(document, 'pages') else 1
            file_size = Path(task.file_path).stat().st_size
            
            logger.debug(f"文档元数据 - 页数: {page_count}, 文件大小: {file_size}")
            
            # 提取表格
            task.progress = 80
            task.current_stage = "表格提取"
            self._update_task_in_db(task)
            tables = self._extract_tables(document, task)
            
            # 提取图片
            task.progress = 82
            task.current_stage = "图片提取"
            self._update_task_in_db(task)
            images = self._extract_images(document, task)
            
            # 跳过文档结构提取
            task.progress = 85
            task.current_stage = "结构提取"
            self._update_task_in_db(task)
            
            # 保存原始解析结果到文件（用于后处理）
            task.progress = 87
            task.current_stage = "文件保存"
            self._update_task_in_db(task)
            output_dir = os.path.join(settings.OUTPUT_DIR, task.task_id)
            os.makedirs(output_dir, exist_ok=True)
            
            # 保存原始markdown文件 - 使用原始文档名称
            file_stem = Path(task.file_path).stem
            original_md_path = os.path.join(output_dir, f"{file_stem}.md")
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
            
            # 保存content_list.json - 使用原始文档名称
            content_list_path = os.path.join(output_dir, f"{file_stem}_content_list.json")
            with open(content_list_path, 'w', encoding='utf-8') as f:
                json.dump(content_list, f, ensure_ascii=False, indent=2)
            
            # 使用DocumentPostProcessor进行后处理
            task.progress = 90
            task.current_stage = "后处理"
            self._update_task_in_db(task)
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
            
            # 构建精简的解析结果，去除重复字段并扁平化
            parsed_result = {
                "document_id": str(uuid.uuid4()),
                "title": document_title,
                "page_count": page_count,
                "content_length": len(markdown_content),
                "parsed_at": time.time(),
                "has_tables": len(tables) > 0,
                "has_images": len(images) > 0,
                "output_directory": output_dir,
                "enhanced": enhanced_result is not None,
                "tables": tables,
                "images": images
            }
                
                # 不再包含文档结构信息
            
            logger.info(f"文档解析完成 - 页数: {page_count}, 表格: {len(tables)}, 图片: {len(images)}")
          
            
            return parsed_result
            
        except Exception as e:
            logger.error(f"解析文件 {task.file_path} 时出错: {e}")
            logger.error(f"错误详情: {traceback.format_exc()}")
            
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
          
            
        except Exception as e:
            logger.error(f"文件验证失败: {e}")
            raise
    
    def _extract_tables(self, document, task: ParseTask) -> List[Dict[str, Any]]:
        """提取表格数据"""
        tables = []
        try:
            if hasattr(document, 'tables'):
                logger.debug(f"发现 {len(document.tables)} 个表格")
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
                        
                logger.info(f"成功提取 {len(tables)} 个表格")
            else:
                logger.debug("文档中未发现表格")
                
        except Exception as e:
            logger.warning(f"提取表格时出错: {e}")
        return tables
    
    def _extract_images(self, document, task: ParseTask) -> List[Dict[str, Any]]:
        """提取图片信息"""
        images = []
        try:
            if hasattr(document, 'pictures'):
                logger.debug(f"发现 {len(document.pictures)} 个图片")
                
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
                        
                logger.info(f"成功提取 {len(images)} 个图片信息")
            else:
                logger.debug("文档中未发现图片")
                
        except Exception as e:
            logger.warning(f"提取图片信息时出错: {e}")
        return images
    
    async def _save_result_to_file(self, task_id: str, result: Dict[str, Any]) -> str:
        """保存解析结果到文件"""
        try:
            output_dir = Path(settings.OUTPUT_DIR) / task_id
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存markdown内容 - 使用原始文档名称
            file_stem = Path(result['file_path']).stem
            md_file = output_dir / f"{file_stem}.md"
            with open(md_file, 'w', encoding='utf-8') as f:
                f.write(result['content'])
            
            # 保存结果JSON - 使用原始文档名称
            json_file = output_dir / f"{file_stem}.json"
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
            # 从数据库中删除任务
            self.task_dao.delete_parse_task(task_id)
            logger.info(f"清理Docling任务: {task_id}")
            return True
        
        # 尝试清理MinerU任务
        if self.mineru_parser.cleanup_task(task_id):
            return True
        
        return False
    
    def _save_task_to_db(self, task: ParseTask):
        """保存任务到数据库"""
        try:
            file_path = Path(task.file_path)
            task_data = {
                'task_id': task.task_id,
                'file_path': task.file_path,
                'file_name': file_path.name,
                'file_size': task.file_size,
                'file_extension': file_path.suffix,
                'mime_type': mimetypes.guess_type(task.file_path)[0],
                'parser_type': task.config.get('parser_type'),
                'status': task.status,
                'progress': task.progress,
                'current_stage': task.current_stage,
                'stage_details': task.stage_details,
                'config': task.config,
                'processing_logs': task.processing_logs
            }
            self.task_dao.create_parse_task(task_data)
        except Exception as e:
            logger.error(f"保存任务到数据库失败: {e}")
    
    def _update_task_in_db(self, task: ParseTask):
        """更新数据库中的任务"""
        try:
            update_data = {
                'status': task.status,
                'progress': task.progress,
                'current_stage': task.current_stage,
                'stage_details': task.stage_details,
                'result': task.result,
                'error': task.error,
                'processing_logs': task.processing_logs
            }
            
            if task.started_at:
                update_data['started_at'] = datetime.fromtimestamp(task.started_at)
            if task.completed_at:
                update_data['completed_at'] = datetime.fromtimestamp(task.completed_at)
                
            self.task_dao.update_parse_task(task.task_id, update_data)
        except Exception as e:
            logger.error(f"更新数据库中的任务失败: {e}")

# 全局解析器实例
document_parser = DocumentParser()