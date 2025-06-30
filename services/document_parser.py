#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import asyncio
import mimetypes
from pathlib import Path
import time
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor
import logging

from config import settings
from models.parse_task import TaskStatus
from services.document_parse_task import DocumentParseTask
from services.document_parse_directory_task import DocumentParseDirectoryTask
from utils.logging_config import setup_logging, get_logger
from dao.task_dao import TaskDAO
from models.task_models import ParseTask


# 初始化日志系统
setup_logging()
logger = get_logger(__name__)

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
            self.executor = ThreadPoolExecutor(max_workers=settings.MAX_CONCURRENT_TASKS)
            self.task_dao = TaskDAO()
           
            
            self._initialized = True
            logger.info(f"DocumentParser初始化完成 - 最大并发任务数: {settings.MAX_CONCURRENT_TASKS}")
            logger.info(f"支持的文件格式: {settings.SUPPORTED_FORMATS}")
            logger.info(f"最大文件大小: {settings.MAX_FILE_SIZE / (1024*1024):.1f}MB")
            logger.info(f"默认解析器类型: {settings.DEFAULT_PARSER_TYPE}")
            logger.info(f"OCR启用状态: {settings.OCR_ENABLED}")
            if settings.OCR_ENABLED:
                logger.info(f"OCR语言: {settings.OCR_LANGUAGES}")
                logger.info(f"GPU使用状态: {settings.USE_GPU}")

    async def parse_directory(
        self,
        directory_path:str,
        config: Optional[Dict[str, Any]] = None,
    ) -> str:
        """解析目录下的所有文件并返回任务ID"""
        task_obj = DocumentParseDirectoryTask.create_parse_directory_task(directory_path, config)
        asyncio.create_task(self._execute_parse_directory_task(task_obj))
        return task_obj.task_id

    async def _execute_parse_directory_task(self, task_obj: ParseTask):
        parse_task = DocumentParseDirectoryTask(task_obj.task_id)
        await parse_task.execute_parse_directory_task(self.executor)

    async def parse_document(
        self, 
        file_path: str, 
        config: Optional[Dict[str, Any]] = None
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
        # 直接调用 DocumentParseTask 的类方法来创建 ParseTask 实例（会自动保存到数据库）
        parser_task = DocumentParseTask.create_parse_task(file_path, config)

        # 异步执行解析
        asyncio.create_task(self._execute_parse_task(parser_task))

        logger.info(
            f"Created Docling parse task {parser_task.task_id} for file: {file_path}"
        )
        return parser_task.task_id

    async def _execute_parse_task(self, parser_task: ParseTask):
        document_parse_task = DocumentParseTask(
            task_id=parser_task.task_id,
        )
        await asyncio.get_event_loop().run_in_executor(
            self.executor, document_parse_task.execute_parse_task
        )

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """权威查询：从数据库获取任务状态，所有对外接口应使用本方法"""
        task = self.task_dao.get_parse_task(task_id)
        if task:
            # 兼容ParseTask对象和dict
            if hasattr(task, 'to_dict'):
                task_dict = task.to_dict()
            else:
                task_dict = dict(task)
            return task_dict
        return None
    
    def get_all_tasks(self) -> List[Dict[str, Any]]:
        """获取所有任务状态"""
        # 从数据库获取所有任务
        all_tasks = self.task_dao.get_all_parse_tasks()
        return [task.to_dict() for task in all_tasks]
    
    def cleanup_expired_tasks(self):
        """清理过期任务"""
        current_time = time.time()
        # 从数据库获取所有任务
        all_tasks = self.task_dao.get_all_parse_tasks()
        
        expired_tasks = [
            task for task in all_tasks
            if current_time - task.created_at.timestamp() > settings.TASK_EXPIRE_TIME
        ]
        
        for task in expired_tasks:
            self.task_dao.delete_parse_task(task.task_id)
            logger.info(f"Cleaned up expired task: {task.task_id}")
    
    def cleanup_task(self, task_id: str) -> bool:
        """清理任务"""
        # 尝试从数据库中删除任务
        task = self.task_dao.get_parse_task(task_id)
        if task:
            self.task_dao.delete_parse_task(task_id)
            logger.info(f"清理任务: {task_id}")
            return True
        return False


# 全局解析器实例
document_parser = DocumentParser()