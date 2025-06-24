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
from docling.datamodel.pipeline_options import PdfPipelineOptions, TesseractCliOcrOptions, RapidOcrOptions
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.backend.docling_parse_backend import DoclingParseDocumentBackend
from config import settings
from utils.logging_config import setup_logging, log_progress, get_logger
# from services.mineru_parser import MinerUDocumentParser
from dao.task_dao import TaskDAO
from utils import document_utils

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
            # self.mineru_parser = MinerUDocumentParser()
            
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
            ocr_languages = (config or {}).get("ocr_languages") or settings.OCR_LANGUAGES

            ocr_backend = config.get("ocr_backend", settings.OCR_BACKEND) if config else settings.OCR_BACKEND

            # 配置PDF处理选项
            pdf_options = PdfPipelineOptions()
            pdf_options.do_ocr = ocr_enabled
            pdf_options.do_table_structure = True
            pdf_options.table_structure_options.do_cell_matching = True
            pdf_options.artifacts_path = settings.DOCLING_MODEL_PATH
            logger.info(f"PDF处理选项配置: OCR={ocr_enabled}, 表格结构提取=True, OCR后端={ocr_backend}")

            # 如果启用OCR，配置OCR后端
            if ocr_enabled:
                if ocr_backend == "tesseract":
                    logger.info("配置Tesseract OCR引擎...")
                    # 支持自定义参数
                    pdf_options.ocr_options = TesseractCliOcrOptions(
                        lang=ocr_languages,
                        force_full_page_ocr=True,  # 强制全页OCR
                    )
                    logger.info(f"Tesseract OCR配置完成 - 语言: {ocr_languages}")
                elif ocr_backend == "rapidorc":
                    # RapidOCR分支仅作保留，docling官方暂未支持
                    logger.info("配置RapidOCR引擎...")
                    det_model_path = os.path.join(
                        settings.RAPID_OCR_MODEL_PATH, "PP-OCRv4", "en_PP-OCRv3_det_infer.onnx"
                    )
                    rec_model_path = os.path.join(
                        settings.RAPID_OCR_MODEL_PATH, "PP-OCRv4", "ch_PP-OCRv4_rec_server_infer.onnx"
                    )
                    cls_model_path = os.path.join(
                        settings.RAPID_OCR_MODEL_PATH, "PP-OCRv3", "ch_ppocr_mobile_v2.0_cls_train.onnx"
                    )

                    pdf_options.ocr_options = RapidOcrOptions(
                        force_full_page_ocr=True,
                        det_model_path=det_model_path,
                        rec_model_path=rec_model_path,
                        cls_model_path=cls_model_path
                    )
                    logger.info(f"RapidOCR配置完成 - 语言: {ocr_languages}, GPU: {settings.USE_GPU}")
                else:
                    logger.info("配置EasyOCR引擎...")
                    from docling.datamodel.pipeline_options import EasyOcrOptions
                    pdf_options.ocr_options = EasyOcrOptions(
                        lang=ocr_languages,
                        force_full_page_ocr=True,  # 强制全页OCR
                        use_gpu=settings.USE_GPU, 
                        download_enabled=True,  # 启用模型下载以支持OCR功能
                        model_storage_directory=settings.EASY_OCR_MODEL_PATH
                    )
                    logger.info(f"EasyOCR配置完成 - 语言: {ocr_languages}, GPU: {settings.USE_GPU}")
                    logger.info(f"EasyOCR模型路径: {settings.EASY_OCR_MODEL_PATH}")
            else:
                logger.info(f"非OCR模式, 跳过OCR配置")

            # 创建转换器
            logger.info("创建DocumentConverter实例...")
            self.converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options),
                }
            )
            logger.info(f"DocumentConverter创建成功 - OCR: {ocr_enabled}, GPU: {settings.USE_GPU}, OCR后端: {ocr_backend}")
            logger.info("Docling文档转换器配置完成")
        except Exception as e:
            logger.error(f"初始化DocumentConverter失败: {e}")
            logger.error(f"错误详情: {traceback.format_exc()}")
            raise
    
    async def parse_directory(
        self,
        directory_path:str,
        config: Optional[Dict[str, Any]] = None,
        parser_type: Optional[str] = None
    ) -> str:
        """解析目录下的所有文件并返回任务ID
        
        Args:
            directory_path: 目录路径
            config: 解析配置
            parser_type: 解析器类型 ('docling' 或 'mineru')，默认使用配置中的默认值
        
        Returns:
            任务ID
        """
        # 验证目录是否存在
        directory = Path(directory_path)
        if not directory.exists() or not directory.is_dir():
            raise NotADirectoryError(f"目录不存在或不是有效目录: {directory_path}")
            
        logger.info(f"开始解析目录: {directory_path}")
        
        # 创建主任务ID
        main_task_id = str(uuid.uuid4())
        
        # 递归获取所有支持的文件
        supported_files = []
        for file_path in directory.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in settings.SUPPORTED_FORMATS:
                supported_files.append(str(file_path))
        
        if not supported_files:
            raise ValueError(f"目录中未找到支持的文件格式: {directory_path}")
            
        logger.info(f"支持的文件: {', '.join(supported_files)}")
        # 控制并发数量
        # 可以从 settings 中获取，或者使用默认值
        # 例如：settings.MAX_CONCURRENT_PARSES_PER_DIRECTORY
        semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_PARSES_PER_DIRECTORY)

        # 创建子任务并跟踪
        subtasks = []
        for file_path in supported_files:
            async def parse_single_document(path):
                async with semaphore:
                    try:
                        subtask_id = await self.parse_document(
                            file_path=path,
                            config=config,
                            parser_type=parser_type
                        )
                        logger.debug(f"创建子任务 {subtask_id} 用于文件: {path}")
                        return {"task_id": subtask_id, "file_path": path}
                    except Exception as e:
                        logger.error(f"处理文件 {path} 时出错: {e}")
                        return None
            subtasks.append(parse_single_document(file_path))
        
        # 等待所有子任务完成
        results = await asyncio.gather(*subtasks)
        subtasks = [r for r in results if r is not None]

        if not subtasks:
            raise ValueError(f"目录中没有成功创建的解析任务: {directory_path}")
        
        # 创建目录解析的主任务记录
        main_task = ParseTask(
            task_id=main_task_id,
            file_path=directory_path,
            file_name=directory.name,
            parser_type=parser_type or settings.DEFAULT_PARSER_TYPE,
            config={
                "is_directory": True,
                "subtasks": subtasks,
                "total_files": len(supported_files),
                **({} if config is None else config)
            },
            status=TaskStatus.RUNNING
        )
        
        self.tasks[main_task_id] = main_task
        self._save_task_to_db(main_task)
        
        # 启动异步任务监控子任务进度
        asyncio.create_task(self._monitor_directory_task(main_task))
        
        logger.info(f"目录解析任务已创建: {main_task_id}, 包含 {len(subtasks)} 个子任务")
        return main_task_id

    async def _monitor_directory_task(self, main_task: ParseTask):
        """异步监控目录解析任务的子任务进度和状态"""
        logger.info(f"开始监控目录解析任务: {main_task.task_id}")
        
        # 获取子任务ID列表
        subtask_ids = [sub.get("task_id") for sub in main_task.config.get("subtasks", []) if sub.get("task_id")]
        
        if not subtask_ids:
            logger.warning(f"目录任务 {main_task.task_id} 没有子任务，直接标记为完成。")
            main_task.status = TaskStatus.COMPLETED
            main_task.progress = 100
            main_task.completed_at = datetime.utcnow()
            self._update_task_in_db(main_task)
            return

        total_subtasks = len(subtask_ids)
        completed_subtasks = 0
        failed_subtasks = 0

        while completed_subtasks + failed_subtasks < total_subtasks:
            await asyncio.sleep(settings.TASK_MONITOR_INTERVAL_SECONDS)  # 定期检查
            
            current_completed = 0
            current_failed = 0
            
            # 从数据库重新加载主任务以获取最新的子任务状态
            reloaded_main_task = self.task_dao.get_parse_task(main_task.task_id)
            if not reloaded_main_task:
                logger.error(f"监控任务 {main_task.task_id} 失败: 无法从数据库加载任务。")
                break # 退出循环，避免无限等待
            
            # 遍历子任务，检查其状态
            for sub_config in reloaded_main_task.config.get("subtasks", []):
                sub_task_id = sub_config.get("task_id")
                if sub_task_id:
                    sub_task = self.task_dao.get_parse_task(sub_task_id)
                    if sub_task:
                        if sub_task.status == TaskStatus.COMPLETED:
                            current_completed += 1
                        elif sub_task.status == TaskStatus.FAILED:
                            current_failed += 1
            
            # 更新已完成/失败的子任务计数
            completed_subtasks = current_completed
            failed_subtasks = current_failed

            # 更新主任务进度
            main_task.progress = int(((completed_subtasks + failed_subtasks) / total_subtasks) * 100)
            main_task.current_stage = f"处理中 ({completed_subtasks}/{total_subtasks} 完成)"
            self._update_task_in_db(main_task)
            logger.debug(f"目录任务 {main_task.task_id} 进度: {main_task.progress}%")

        # 所有子任务处理完毕，更新主任务最终状态
        main_task.completed_at = datetime.utcnow()
        if failed_subtasks > 0:
            main_task.status = TaskStatus.FAILED
            main_task.error = f"{failed_subtasks} 个子任务失败。"
            main_task.current_stage = "任务失败"
            logger.error(f"目录任务 {main_task.task_id} 失败: {failed_subtasks} 个子任务失败。")
        else:
            main_task.status = TaskStatus.COMPLETED
            main_task.current_stage = "任务完成"
            logger.info(f"目录任务 {main_task.task_id} 成功完成。")
        
        main_task.progress = 100
        self._update_task_in_db(main_task)

    
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
        # if selected_parser == "mineru":
        #     logger.info(f"Using MinerU parser for file: {file_path}")
        #     return await self.mineru_parser.parse_document(file_path, config)
        
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

            # 任务完成时同步 output_directory 字段
            if result and result.get("output_directory"):
                task.output_directory = result["output_directory"]

            # 默认保存结果到文件
            # task.progress = 90
            # task.current_stage = "保存结果"
            # self._update_task_in_db(task)
            # output_path = await self._save_result_to_file(task.task_id, result)
            
            # 保存完成后，添加输出文件路径到结果中
            # result["output_file"] = output_path
            
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
                "content_length": result.get("content_length", 0),
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
            
    def _parse_file_to_markdown(self, file_path: str, config: dict = None) -> str:
        """辅助函数：用docling解析单个文件为markdown，返回markdown内容"""
        result = self.converter.convert(file_path)
        document = result.document
        markdown_content = document.export_to_markdown()
        return markdown_content

    def _parse_file_sync(self, task: ParseTask) -> Dict[str, Any]:
        try:
            file_path = str(task.file_path)
            file_ext = Path(file_path).suffix.lower()
            if file_ext in ['.html', '.htm']:
                try:
                    from bs4 import BeautifulSoup
                except ImportError:
                    raise ImportError("请先安装 beautifulsoup4: pip install beautifulsoup4")
                try:
                    import requests
                except ImportError:
                    raise ImportError("请先安装 requests: pip install requests")
                logger.info(f"使用BeautifulSoup处理HTML文件: {file_path}")
                with open(file_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                soup = BeautifulSoup(html_content, 'html.parser')
                # 2. 查找所有超链接，针对PDF/Word/图片等用docling解析
                for a in soup.find_all('a', href=True):
                    href = a['href']
                    # 判断是否为本地或远程文件，且为PDF/Word/图片
                    if any(href.lower().endswith(ext) for ext in ['.pdf', '.doc', '.docx', '.ppt', '.pptx', '.jpg', '.jpeg', '.png', '.bmp']):
                        # 下载远程文件到本地
                        if href.startswith('http'):
                            local_filename = os.path.join(str(settings.OUTPUT_DIR), str(task.task_id), os.path.basename(href))
                            os.makedirs(os.path.dirname(local_filename), exist_ok=True)
                            try:
                                r = requests.get(href, timeout=20)
                                with open(local_filename, 'wb') as f:
                                    f.write(r.content)
                                logger.info(f"已下载远程文件: {href} -> {local_filename}")
                            except Exception as e:
                                logger.warning(f"下载远程文件失败: {href}, {e}")
                                continue
                        else:
                            local_filename = os.path.join(os.path.dirname(file_path), href)
                        # 用docling解析该文件，输出markdown
                        try:
                            config = task.config if isinstance(task.config, dict) else {}
                            sub_md_content = self._parse_file_to_markdown(local_filename, config)
                            # 保存子markdown到本地
                            sub_md_path = os.path.join(str(settings.OUTPUT_DIR), str(task.task_id), f"{Path(local_filename).stem}.md")
                            with open(sub_md_path, 'w', encoding='utf-8') as f:
                                f.write(sub_md_content)
                            # 替换a标签的href为本地md路径
                            a['href'] = os.path.relpath(sub_md_path, os.path.dirname(file_path))
                        except Exception as e:
                            logger.warning(f"docling解析超链接文件失败: {local_filename}, {e}")
                            continue
                # 3. 直接保存处理后的HTML到输出目录
                output_dir = os.path.join(str(settings.OUTPUT_DIR), str(task.task_id))
                os.makedirs(output_dir, exist_ok=True)
                file_stem = Path(file_path).stem
                html_out_path = os.path.join(output_dir, f"{file_stem}.html")
                with open(html_out_path, 'w', encoding='utf-8') as f:
                    f.write(str(soup))
                logger.info(f"HTML处理完成，输出: {html_out_path}")
                # 构建结果（不再包含 content 字段）
                parsed_result = {
                    "document_id": str(uuid.uuid4()),
                    "title": file_stem,
                    "content_length": len(html_out_path),
                    "parsed_at": time.time(),
                    "output_directory": output_dir,
                    "has_tables": False,
                    "has_images": False,
                    "tables": [],
                    "images": []
                }
                return parsed_result
            # 其它类型用docling
            task.progress = 15
            task.current_stage = "配置转换器"
            self._update_task_in_db(task)
            if task.config:
                logger.info(f"使用任务配置重新设置转换器: {task.config}")
                self._setup_converter(task.config)
            task.progress = 20
            task.current_stage = "文档加载"
            self._update_task_in_db(task)
            logger.info(f"开始转换文档: {task.file_path}")
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
            task.stage_details.update({"conversion_time": conversion_time})
            self._update_task_in_db(task)
            logger.info(f"Docling转换完成，耗时: {conversion_time:.2f}秒")
            task.progress = 70
            task.current_stage = "内容提取"
            self._update_task_in_db(task)
            document = result.document
            logger.debug(f"开始提取文档内容，文档类型: {type(document).__name__}")
            task.progress = 75
            task.current_stage = "数据提取"
            self._update_task_in_db(task)
            document_title = getattr(document, 'title', Path(task.file_path).stem)
            logger.debug(f"文档标题: {document_title}")
            output_dir = os.path.join(settings.OUTPUT_DIR, task.task_id)
            os.makedirs(output_dir, exist_ok=True)
            file_stem = Path(task.file_path).stem
            export_type = task.config.get('export_type', 'markdown')
            if export_type == 'json':
                exporter = document_utils.JsonFileExporter()
                export_path, export_content = exporter.export(document, output_dir, file_stem)
                logger.info(f"文档已导出为json: {export_path}")
            else:
                exporter = document_utils.MarkdownFileExporter()
                export_path, export_content = exporter.export(document, output_dir, file_stem)
                logger.info(f"文档已导出为markdown: {export_path}")
            page_count = len(document.pages) if hasattr(document, 'pages') else 1
            file_size = Path(task.file_path).stat().st_size
            tables = document_utils.extract_tables(document)
            images = document_utils.extract_images(document)
            if export_type == 'markdown':
                content_list = document_utils.build_content_list_from_markdown(export_content)
                content_list_path = os.path.join(output_dir, f"{file_stem}_content_list.json")
                with open(content_list_path, 'w', encoding='utf-8') as f:
                    json.dump(content_list, f, ensure_ascii=False, indent=2)
            parsed_result = {
                "document_id": str(uuid.uuid4()),
                "title": document_title,
                "page_count": page_count,
                "content_length": len(export_content) if isinstance(export_content, str) else 0,
                "parsed_at": time.time(),
                "has_tables": len(tables) > 0,
                "has_images": len(images) > 0,
                "output_directory": output_dir,
                "export_type": export_type,
                "export_path": export_path,
                "tables": tables,
                "images": images
            }
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
            return str(output_dir);
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            raise
    
    def get_task_in_memory(self, task_id: str) -> Optional[Dict[str, Any]]:
        """仅查询内存缓存中的任务状态（仅供内部调度/运行时使用，非权威状态）"""
        task = self.tasks.get(task_id)
        if task:
            task_dict = task.to_dict()
            task_dict["parser_type"] = "docling"
            return task_dict
        # # 检查MinerU任务（如有需要可补充）
        # mineru_status = self.mineru_parser.get_task_status(task_id)
        # if mineru_status:
        #     mineru_status["parser_type"] = "mineru"
        #     return mineru_status
        return None

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """权威查询：从数据库获取任务状态，所有对外接口应使用本方法"""
        task = self.task_dao.get_parse_task(task_id)
        if task:
            # 兼容ParseTask对象和dict
            if hasattr(task, 'to_dict'):
                task_dict = task.to_dict()
            else:
                task_dict = dict(task)
            task_dict["parser_type"] = task_dict.get("parser_type", "docling")
            return task_dict
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
        
        # # 检查MinerU任务
        # mineru_result = self.mineru_parser.get_task_result(task_id)
        # if mineru_result:
        #     mineru_result["parser_type"] = "mineru"
        #     return mineru_result
        
        # return None
    
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
        # if self.mineru_parser.cleanup_task(task_id):
        #     return True
        
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
                'processing_logs': task.processing_logs,
                'started_at': task.started_at,
                'completed_at': task.completed_at
            }
                
            self.task_dao.update_parse_task(task.task_id, update_data)
        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.error(f"更新数据库中的任务失败: {e}")

# 全局解析器实例
document_parser = DocumentParser()