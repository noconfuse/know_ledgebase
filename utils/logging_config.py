#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日志配置模块 - 提供详细的日志记录功能
"""

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

from config import settings

class DoclingLogHandler(logging.Handler):
    """自定义Docling日志处理器"""
    
    def __init__(self, log_file: str):
        super().__init__()
        self.log_file = log_file
        
    def emit(self, record):
        """输出日志记录"""
        try:
            msg = self.format(record)
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(f"{msg}\n")
        except Exception:
            self.handleError(record)

class ProgressLogHandler(logging.Handler):
    """进度日志处理器"""
    
    def __init__(self, log_file: str):
        super().__init__()
        self.log_file = log_file
        
    def emit(self, record):
        """输出进度日志"""
        try:
            if hasattr(record, 'progress_data'):
                timestamp = datetime.now().isoformat()
                progress_data = record.progress_data
                log_entry = {
                    "timestamp": timestamp,
                    "task_id": progress_data.get("task_id"),
                    "progress": progress_data.get("progress"),
                    "stage": progress_data.get("stage"),
                    "message": progress_data.get("message"),
                    "details": progress_data.get("details", {})
                }
                
                import json
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    f.write(f"{json.dumps(log_entry, ensure_ascii=False)}\n")
        except Exception:
            self.handleError(record)

def setup_logging():
    """设置日志配置"""
    
    # 确保日志目录存在
    log_dir = Path(settings.LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建根日志器
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.LOG_LEVEL))
    
    # 清除现有处理器
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 创建格式化器
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    
    # 带异常堆栈的详细格式化器
    detailed_with_exc_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    root_logger.addHandler(console_handler)
    
    if settings.ENABLE_FILE_LOGGING:
        # 主应用日志文件处理器（带轮转）
        app_log_file = log_dir / "app.log"
        app_handler = logging.handlers.RotatingFileHandler(
            app_log_file,
            maxBytes=_parse_size(settings.LOG_ROTATION_SIZE),
            backupCount=settings.LOG_RETENTION_DAYS,
            encoding='utf-8'
        )
        app_handler.setLevel(getattr(logging, settings.LOG_LEVEL))
        app_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(app_handler)
        
        # 文档解析专用日志
        parse_log_file = log_dir / "document_parsing.log"
        parse_handler = logging.handlers.RotatingFileHandler(
            parse_log_file,
            maxBytes=_parse_size(settings.LOG_ROTATION_SIZE),
            backupCount=settings.LOG_RETENTION_DAYS,
            encoding='utf-8'
        )
        parse_handler.setLevel(logging.DEBUG)
        parse_handler.setFormatter(detailed_formatter)
        
        # 为文档解析器添加专用处理器
        parse_logger = logging.getLogger('services.document_parser')
        parse_logger.addHandler(parse_handler)
        parse_logger.setLevel(logging.DEBUG)
        
        # 错误日志文件处理器
        error_log_file = log_dir / "errors.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=_parse_size(settings.LOG_ROTATION_SIZE),
            backupCount=settings.LOG_RETENTION_DAYS,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_with_exc_formatter)
        root_logger.addHandler(error_handler)
        
        # 进度日志处理器
        progress_log_file = log_dir / "progress.jsonl"
        progress_handler = ProgressLogHandler(str(progress_log_file))
        progress_handler.setLevel(logging.INFO)
        
        progress_logger = logging.getLogger('progress')
        progress_logger.addHandler(progress_handler)
        progress_logger.setLevel(logging.INFO)
        progress_logger.propagate = False
    
    if settings.ENABLE_DOCLING_LOGGING:
        # Docling专用日志
        docling_log_file = log_dir / "docling_processing.log"
        docling_handler = DoclingLogHandler(str(docling_log_file))
        docling_handler.setLevel(logging.DEBUG)
        docling_handler.setFormatter(detailed_formatter)
        
        # 为docling相关模块添加处理器
        docling_loggers = [
            'docling',
            'docling.document_converter',
            'docling.backend',
            'docling.datamodel',
            'docling.pipeline'
        ]
        
        for logger_name in docling_loggers:
            logger = logging.getLogger(logger_name)
            logger.addHandler(docling_handler)
            logger.setLevel(logging.DEBUG)
    
    logging.info(f"日志系统初始化完成，日志目录: {log_dir}")

def _parse_size(size_str: str) -> int:
    """解析大小字符串为字节数"""
    size_str = size_str.upper()
    if size_str.endswith('KB'):
        return int(size_str[:-2]) * 1024
    elif size_str.endswith('MB'):
        return int(size_str[:-2]) * 1024 * 1024
    elif size_str.endswith('GB'):
        return int(size_str[:-2]) * 1024 * 1024 * 1024
    else:
        return int(size_str)

def log_progress(task_id: str, progress: int, stage: str, message: str, details: dict = None):
    """记录进度日志"""
    progress_logger = logging.getLogger('progress')
    
    # 创建进度记录
    record = logging.LogRecord(
        name='progress',
        level=logging.INFO,
        pathname='',
        lineno=0,
        msg=message,
        args=(),
        exc_info=None
    )
    
    record.progress_data = {
        "task_id": task_id,
        "progress": progress,
        "stage": stage,
        "message": message,
        "details": details or {}
    }
    
    progress_logger.handle(record)

def get_logger(name: str) -> logging.Logger:
    """获取指定名称的日志器"""
    return logging.getLogger(name)