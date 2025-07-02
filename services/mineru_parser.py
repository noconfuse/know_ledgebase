# -*- coding: utf-8 -*-
import os
import re
import time
import uuid
import json
import asyncio
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from concurrent.futures import ThreadPoolExecutor
import logging
from datetime import datetime

from config import settings
from utils.logging_config import setup_logging, log_progress, get_logger
from .document_post_processor import DocumentPostProcessor
from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
from magic_pdf.pipe.UNIPipe import UNIPipe
from magic_pdf.pipe.OCRPipe import OCRPipe
from magic_pdf.pipe.TXTpipe import TXTPipe
from magic_pdf.config.enums import SupportedPdfParseMethod
from utils.document_utils import truncate_filename
from models.task_models import ParseTask
from models.parse_task import TaskStatus
from dao.task_dao import TaskDAO

# 初始化日志系统
setup_logging()
logger = get_logger(__name__)

# MinerUParseTask已被移除，现在统一使用ParseTask

class MinerUDocumentParser:
    """MinerU文档解析器"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            logger.info("开始初始化MinerUDocumentParser...")
            self.tasks: Dict[str, ParseTask] = {}
            self.task_dao = TaskDAO()
            self._initialized = True
            logger.info(f"MinerUDocumentParser初始化完成")
            logger.info(f"MinerU输出目录: {os.path.join(settings.KNOWLEDGE_BASE_DIR, 'outputs')}")
            logger.info(f"MinerU最大并发数: {settings.MINERU_MAX_WORKERS}")
    
    async def parse_document(
        self, 
        file_path: str, 
        config: Optional[Dict[str, Any]] = None,
    ) -> str:
        """解析文档并返回任务ID"""
        
        # 验证文件
        if not Path(file_path).exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_size = Path(file_path).stat().st_size
        if file_size > settings.MAX_FILE_SIZE:
            raise ValueError(f"File size {file_size} exceeds maximum {settings.MAX_FILE_SIZE}")
        
        file_ext = Path(file_path).suffix.lower()
        # MinerU主要支持PDF文件
        if file_ext != ".pdf":
            raise ValueError(f"MinerU only supports PDF files, got: {file_ext}")
        
        # 创建任务
        task_id = str(uuid.uuid4())
        task_config = config or {}
        
        # 获取文件信息
        file_path_obj = Path(file_path)
        file_stat = file_path_obj.stat()
        
        task = ParseTask(
            task_id=task_id,
            file_path=file_path,
            file_name=file_path_obj.name,
            file_size=file_stat.st_size,
            file_extension=file_path_obj.suffix,
            parser_type="mineru",
            config=task_config,
            status=TaskStatus.PENDING
        )
        self.tasks[task_id] = task
        
        # 保存到数据库
        self._save_task_to_db(task)
        
        # 异步执行解析
        asyncio.create_task(self._execute_parse_task(task))
        
        logger.info(f"Created MinerU parse task {task_id} for file: {file_path}")
        return task_id
    
    async def _execute_parse_task(self, task: ParseTask):
        """执行解析任务"""
        try:
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.utcnow()
            task.progress = 5
            task.current_stage = "任务开始"
            
            # 更新数据库状态
            self._update_task_in_db(task)
            
            # 验证文件
            task.progress = 10
            task.current_stage = "文件验证"
            self._update_task_in_db(task)
            self._validate_file(task.file_path)
            
            # 执行解析
            result = await self._sync_parse_file(task)
            
            # 任务完成
            task.result = result
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.utcnow()
            task.progress = 100
            task.current_stage = "解析完成"
            
            # 更新数据库状态和结果
            self._update_task_in_db(task)
            
            logger.info(f"MinerU解析任务 {task.task_id} 完成")
            
        except Exception as e:
            task.error = str(e)
            task.status = TaskStatus.FAILED
            task.completed_at = datetime.utcnow()
            task.progress = 0
            task.current_stage = "解析失败"
            task.error = f"解析失败: {str(e)}"
            
            # 更新数据库状态和错误信息
            self._update_task_in_db(task)
            
            logger.error(f"MinerU解析任务 {task.task_id} 失败: {e}")
            logger.error(f"错误详情: {traceback.format_exc()}")
    
    def _validate_file(self, file_path: str):
        """验证文件"""
        if not Path(file_path).exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        file_size = Path(file_path).stat().st_size
        if file_size == 0:
            raise ValueError(f"文件为空: {file_path}")
        
        if file_size > settings.MAX_FILE_SIZE:
            raise ValueError(f"文件大小 {file_size} 超过最大限制 {settings.MAX_FILE_SIZE}")
    
    async def _sync_parse_file(self, task: ParseTask) -> Dict[str, Any]:
        """同步解析文件"""
        try:
            # 准备输出目录
            task.progress = 15
            task.current_stage = "准备输出目录"
            self._update_task_in_db(task)
            output_dir = self._prepare_output_directory(task)
            
            # 开始MinerU解析
            task.progress = 20
            task.current_stage = "MinerU解析"
            self._update_task_in_db(task)
            logger.info(f"开始MinerU解析: {task.file_path}")
            
            start_time = time.time()
            
            # 调用MinerU解析器
            max_workers = task.config.get("max_workers", settings.MINERU_MAX_WORKERS)
            parse_result = await self._async_parse_pdf(
                task.file_path, 
                output_dir, 
                max_workers=max_workers
            )
            
            conversion_time = time.time() - start_time
            task.progress = 70
            task.current_stage = "解析完成"
            if not task.stage_details:
                task.stage_details = {}
            task.stage_details.update({
                "conversion_time": conversion_time
            })
            self._update_task_in_db(task)
            
            logger.info(f"MinerU解析完成，耗时: {conversion_time:.2f}秒")
            
            # 处理解析结果
            task.progress = 75
            task.current_stage = "结果处理"
            self._update_task_in_db(task)
            processed_result = self._process_parse_result(task, parse_result, output_dir)
            
            # 使用DocumentPostProcessor进行后处理
            task.progress = 80
            task.current_stage = "后处理"
            self._update_task_in_db(task)
            post_processor = DocumentPostProcessor()
            # 使用原始文件名（去掉扩展名）作为基础名
            base_name = Path(task.file_name).stem
            content_json_path = os.path.join(output_dir, f"{base_name}_content_list.json")
            if os.path.exists(content_json_path):
                enhanced_result = post_processor.process_document(content_json_path, output_dir)
            else:
                enhanced_result = None
                logger.warning(f"未找到content_list.json文件: {content_json_path}")
            
            # 提取内容和元数据
            task.progress = 85
            task.current_stage = "内容提取"
            self._update_task_in_db(task)
            final_result = self._extract_content_and_metadata(task, processed_result, enhanced_result)
            
            task.progress = 95
            task.current_stage = "结果整理"
            self._update_task_in_db(task)
            
            return final_result
            
        except Exception as e:
            logger.error(f"MinerU解析失败: {e}")
            raise
    
    def _prepare_output_directory(self, task: ParseTask) -> str:
        """准备输出目录"""
        # 使用与Docling相同的目录结构: OUTPUT_DIR / task_id
        output_dir = os.path.join(settings.KNOWLEDGE_BASE_DIR, "outputs", task.task_id)
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"MinerU输出目录: {output_dir}")
        return output_dir
    
    def _process_parse_result(self, task: ParseTask, parse_result: Dict, output_dir: str) -> Dict[str, Any]:
        """处理MinerU解析结果"""
        try:
            # MinerU返回的结果包含各种输出文件路径
            logger.info(f"MinerU解析结果: {parse_result}")
            
            # 读取markdown内容
            markdown_content = ""
            if "markdown" in parse_result and os.path.exists(parse_result["markdown"]):
                with open(parse_result["markdown"], 'r', encoding='utf-8') as f:
                    markdown_content = f.read()
            
            # 读取内容列表
            content_list = []
            if "content_list" in parse_result and os.path.exists(parse_result["content_list"]):
                with open(parse_result["content_list"], 'r', encoding='utf-8') as f:
                    content_list = json.load(f)
            
            return {
                "markdown_content": markdown_content,
                "content_list": content_list,
                "output_files": parse_result,
                "output_directory": output_dir
            }
            
        except Exception as e:
            logger.error(f"处理MinerU解析结果失败: {e}")
            raise
    
    def _extract_content_and_metadata(self, task: ParseTask, processed_result: Dict[str, Any], enhanced_result: Dict[str, Any] = None) -> Dict[str, Any]:
        """提取内容和元数据"""
        try:
            # 优先使用enhanced_result中的内容，如果没有则使用原始结果
            if enhanced_result and "single_file" in enhanced_result:
                # 读取增强后的文件内容
                enhanced_files = enhanced_result["single_file"]
                if "markdown" in enhanced_files and os.path.exists(enhanced_files["markdown"]):
                    with open(enhanced_files["markdown"], 'r', encoding='utf-8') as f:
                        markdown_content = f.read()
                else:
                    markdown_content = processed_result.get("markdown_content", "")
                
                if "content_json" in enhanced_files and os.path.exists(enhanced_files["content_json"]):
                    with open(enhanced_files["content_json"], 'r', encoding='utf-8') as f:
                        content_list = json.load(f)
                else:
                    content_list = processed_result.get("content_list", [])
            else:
                markdown_content = processed_result.get("markdown_content", "")
                content_list = processed_result.get("content_list", [])
            
            output_files = processed_result.get("output_files", {})
            
            # 提取基本信息
            document_title = Path(task.file_path).stem
            file_size = Path(task.file_path).stat().st_size
            
            # 统计内容信息
            text_blocks = [item for item in content_list if item.get("type") == "text"]
            image_blocks = [item for item in content_list if item.get("type") == "image"]
            table_blocks = [item for item in content_list if item.get("type") == "table"]
            
            # 构建精简的解析结果，去除重复字段并扁平化
            parsed_result = {
                "document_id": str(uuid.uuid4()),
                "title": document_title,
                "content_length": len(markdown_content),
                "parsed_at": time.time(),
                "has_tables": len(table_blocks) > 0,
                "has_images": len(image_blocks) > 0,
                "output_directory": processed_result.get("output_directory"),
                "output_files": output_files,
                "enhanced": enhanced_result is not None,
                "statistics": {
                    "text_blocks": len(text_blocks),
                    "image_blocks": len(image_blocks),
                    "table_blocks": len(table_blocks),
                    "total_blocks": len(content_list)
                }
            }
            
            logger.info(f"MinerU内容提取完成 - 标题: {document_title}, 内容长度: {len(markdown_content)}")
            return parsed_result
            
        except Exception as e:
            logger.error(f"提取内容和元数据失败: {e}")
            raise
    

    
    # ==================== MinerU核心解析方法 ====================
    
    async def _async_parse_pdf(self, input_path: str, output_root: str, max_workers: int = 1):
        """异步调用 parse 方法"""
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return await loop.run_in_executor(executor, self._parse_pdf, input_path, output_root, max_workers)
    
    def _parse_pdf(self, input_path: str, output_root: str, max_workers: int = 1) -> Union[Dict, List[Dict]]:
        """统一入口方法，支持处理单个文件或目录"""
        if os.path.isfile(input_path):
            return self._parse_single_pdf(input_path, output_root)
        elif os.path.isdir(input_path):
            return self._parse_batch_pdfs(input_path, output_root, max_workers)
        else:
            raise ValueError(f"Invalid input path: {input_path}")
    
    def _parse_single_pdf(self, pdf_path: str, output_root: str) -> Dict:
        """处理单个 PDF 文件"""
        file_stem = Path(pdf_path).stem
        # 使用截断函数避免文件名过长
        truncated_file_stem = truncate_filename(file_stem, max_length=60, preserve_extension=False)
        # 直接使用传入的output_root作为输出目录，不再创建子目录
        output_dir = output_root
        os.makedirs(output_dir, exist_ok=True)

        # 原 parse_pdf 方法内容
        image_dir, image_writer, md_writer = self._prepare_output_dirs(output_dir)
        pdf_bytes = self._read_pdf_content(pdf_path)

        ds = PymuDocDataset(pdf_bytes)
        parse_method = ds.classify()

        if parse_method == SupportedPdfParseMethod.OCR:
            infer_result = ds.apply(doc_analyze, ocr=True)
            pipe_result = infer_result.pipe_ocr_mode(image_writer)
        else:
            infer_result = ds.apply(doc_analyze, ocr=False)
            pipe_result = infer_result.pipe_txt_mode(image_writer)

        result_paths = self._generate_output_files(
            output_dir,
            truncated_file_stem,
            infer_result,
            pipe_result,
            md_writer,
            image_dir
        )

        return result_paths
    
    def _parse_batch_pdfs(self, dir_path: str, output_root: str, max_workers: int) -> Dict[str, Dict]:
        """批量处理目录中的 PDF 文件"""
        results = {}
        pdf_files = [os.path.join(dir_path, file_name) for file_name in os.listdir(dir_path) if
                     file_name.lower().endswith('.pdf')]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._parse_single_pdf, pdf_path, output_root): pdf_path for pdf_path in
                       pdf_files}
            for future in futures:
                pdf_path = futures[future]
                file_name = os.path.basename(pdf_path)
                try:
                    result = future.result()
                    results[file_name] = result
                except Exception as e:
                    logger.error(f"Error processing {file_name}: {str(e)}")
                    results[file_name] = {"error": str(e)}

        return results
    
    def _prepare_output_dirs(self, output_dir: str):
        """创建并初始化输出目录结构"""
        image_dir = os.path.join(output_dir, "images")
        os.makedirs(image_dir, exist_ok=True)

        return (
            os.path.basename(image_dir),  # 图片相对路径
            FileBasedDataWriter(image_dir),
            FileBasedDataWriter(output_dir),
        )
    
    def _read_pdf_content(self, pdf_path: str) -> bytes:
        """读取 PDF 二进制内容"""
        base_dir = os.path.dirname(pdf_path)
        filename = os.path.basename(pdf_path)
        return FileBasedDataReader(base_dir).read(filename)
    
    def _generate_output_files(self, output_dir, base_name, infer_result, pipe_result, md_writer, img_rel_dir):
        """生成所有输出文件并返回路径字典"""
        # 路径构造辅助方法
        def build_path(suffix): return os.path.join(output_dir, f"{base_name}{suffix}")

        # 可视化输出文件
        visual_outputs = {
            "model_visualization": build_path("_model.pdf"),
            "layout_visualization": build_path("_layout.pdf"),
            "spans_visualization": build_path("_spans.pdf"),
        }
        infer_result.draw_model(visual_outputs["model_visualization"])
        pipe_result.draw_layout(visual_outputs["layout_visualization"])
        pipe_result.draw_span(visual_outputs["spans_visualization"])

        # 核心数据输出文件 - 使用原始文档名称
        md_filename = f"{base_name}.md"
        content_list_filename = f"{base_name}_content_list.json"
        mediate_filename = f"{base_name}_mediate.json"
        
        data_outputs = {
            "markdown": os.path.join(output_dir, md_filename),
            "content_list": os.path.join(output_dir, content_list_filename),
            "intermediate_data": os.path.join(output_dir, mediate_filename),
        }
        pipe_result.dump_md(md_writer, md_filename, img_rel_dir)
        pipe_result.dump_content_list(md_writer, content_list_filename, img_rel_dir)
        pipe_result.dump_middle_json(md_writer, mediate_filename)

        return {**visual_outputs, **data_outputs}
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务状态"""
        # 优先从内存获取
        task = self.tasks.get(task_id)
        if task:
            return task.to_dict()
        
        # 从数据库获取
        db_task = self.task_dao.get_parse_task(task_id)
        if db_task:
            return db_task.to_dict()
        
        return None
    
    def get_task_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务结果"""
        # 优先从内存获取
        task = self.tasks.get(task_id)
        if task and task.status == TaskStatus.COMPLETED:
            return task.result
        
        # 从数据库获取
        db_task = self.task_dao.get_parse_task(task_id)
        if db_task and db_task.status == TaskStatus.COMPLETED:
            return db_task.result
        
        return None
    
    def cleanup_task(self, task_id: str) -> bool:
        """清理任务"""
        # 从内存清理
        if task_id in self.tasks:
            del self.tasks[task_id]
        
        # 从数据库删除
        success = self.task_dao.delete_parse_task(task_id)
        if success:
            logger.info(f"清理MinerU任务: {task_id}")
        
        return success
    
    def _save_task_to_db(self, task: ParseTask):
        """保存任务到数据库"""
        try:
            self.task_dao.create_parse_task(task)
            logger.debug(f"任务 {task.task_id} 已保存到数据库")
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
                'processing_logs': task.processing_logs,
                'started_at': task.started_at,
                'completed_at': task.completed_at,
                'result': task.result,
                'error': task.error
            }
            self.task_dao.update_parse_task(task.task_id, update_data)
            logger.debug(f"任务 {task.task_id} 状态已更新到数据库")
        except Exception as e:
            logger.error(f"更新任务状态到数据库失败: {e}")

# 创建全局实例
mineru_parser = MinerUDocumentParser()