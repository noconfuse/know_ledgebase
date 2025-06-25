from typing import Dict, Any, Optional

import mimetypes
from pathlib import Path
import time
from datetime import datetime
import uuid
import traceback
import os
from typing import Dict, Optional, Any

from bs4 import BeautifulSoup
import requests
from services import docling_parser
from utils.logging_config import setup_logging, get_logger

from config import settings
from dao.task_dao import TaskDAO
from models.parse_task import TaskStatus
from models.task_models import ParseTask
from services.docling_parser import DoclingParser
from services.document_docling_processor import DocumentDoclingProcessor
from services.document_html_processor import DocumentHTMLProcessor

setup_logging()
logger = get_logger(__name__)


class DocumentParseTask:
    """文档解析任务"""

    @staticmethod
    def validate_file(file_path: str, file_extension: str):
        """验证文件是否符合解析条件"""
        path_obj = Path(file_path)
        if not path_obj.exists():
            raise FileNotFoundError(f"文件未找到: {file_path}")

        if not path_obj.is_file():
            raise ValueError(f"路径不是文件: {file_path}")

        if path_obj.stat().st_size > settings.MAX_FILE_SIZE:
            raise ValueError(
                f"文件大小 {path_obj.stat().st_size} 超过最大限制 {settings.MAX_FILE_SIZE}"
            )

        if file_extension not in settings.SUPPORTED_FORMATS:
            raise ValueError(f"不支持的文件格式: {file_extension}")

    @staticmethod
    def create_parse_task(file_path: str, config: Optional[Dict[str, Any]] = None, parent_task: Optional[ParseTask] = None) -> ParseTask:
        """创建解析任务
        
        Args:
            file_path: 文件路径
            config: 任务配置
            parent_task: 父任务，如果提供则创建子任务
        
        Returns:
            ParseTask: 创建的解析任务
        """
        file_path_obj = Path(file_path)
        file_stat = file_path_obj.stat()
        file_extension = file_path_obj.suffix

        # 校验文件
        DocumentParseTask.validate_file(file_path, file_extension)

        parser_type = (
            config.get("parser_type") if config else settings.DEFAULT_PARSER_TYPE
        )
        # 校验解析器类型
        if parser_type not in settings.SUPPORTED_PARSER_TYPES:
            raise ValueError(f"不支持的解析器类型: {parser_type}")

        if parser_type == "mineru" and file_extension != ".pdf":
            raise ValueError(
                f"MinerU 解析器只支持 PDF 文件, 但文件类型为: {file_extension}"
            )

        # 校验文档类型
        if (
            parser_type == "docling"
            and file_extension not in settings.SUPPORTED_FORMATS
        ):
            raise ValueError(f"Docling 解析器不支持文件类型: {file_extension}")

        task_id = str(uuid.uuid4())

        parse_task = ParseTask(
            task_id=task_id,
            file_path=file_path,
            file_name=file_path_obj.name,
            file_size=file_stat.st_size,
            file_extension=file_path_obj.suffix,
            parser_type=parser_type,
            mime_type=mimetypes.guess_type(file_path)[0],
            status=TaskStatus.PENDING,
            config=config,
            parent_task_id=parent_task.id if parent_task else None
        )
        
        # 保存到数据库
        task_dao = TaskDAO()
        created_task = task_dao.create_parse_task(parse_task)
        
        # 如果有父任务，需要建立父子关系
        if parent_task and created_task:
            # 通过数据库操作建立父子关系
            task_dao.add_subtask_to_parent(parent_task.task_id, created_task.task_id)
             
        return created_task or parse_task

    def __init__(self, task_id: str) -> None:
        self.task_id = task_id
        self.task_dao = TaskDAO()
        print(task_id,'task_id')
        self.task = self.task_dao.get_parse_task(task_id)
        pass

    def _parse_file_sync(self) -> Dict[str, Any]:
        try:
            logger.info(f"使用任务配置重新设置Docling转换器: {self.task.config}")
            docling_parser = DoclingParser()
            docling_parser.setup_converter(self.task.config)

          
            self.task.progress = 30
            self.task.current_stage = "文档解析"
            self.task_dao.update_parse_task(self.task.task_id, self.task.to_dict())
            start_time = time.time()
            logger.debug(f"调用Docling转换器处理文件: {self.task.file_path}")
            result = docling_parser.parse_file(self.task.file_path) 
            conversion_time = time.time() - start_time
            self.task.progress = 60
            self.task.current_stage = "转换完成"
           
            logger.info(f"Docling转换完成，耗时: {conversion_time:.2f}秒")
            self.task_dao.update_parse_task(self.task.task_id, self.task.to_dict())

            self.task.progress = 75
            self.task.current_stage = "数据提取"
            self.task_dao.update_parse_task(self.task.task_id, self.task.to_dict())

            document = result.document
            logger.debug(f"开始提取文档内容，文档类型: {type(document).__name__}")
            document_title = getattr(document, 'title', Path(self.task.file_path).stem)
            logger.debug(f"文档标题: {document_title}")

            output_dir = os.path.join(settings.OUTPUT_DIR, self.task.task_id)
            os.makedirs(output_dir, exist_ok=True)
            file_name = self.task.file_name
            processor = DocumentDoclingProcessor()

            processed_result = processor.process_document(
                document,
                output_dir,
                file_name
            )

            # 补全其他信息
            parsed_result = {
                "title": processed_result.get("title"),
                "page_count": processed_result.get("page_count"),
            }

            self.task.output_directory = output_dir
            self.task.progress = 85
            self.task.current_stage = "解析完成"
            self.task_dao.update_parse_task(self.task.task_id, self.task.to_dict())
            return parsed_result
        except Exception as e:
            logger.error(f"解析文件 {self.task.file_path} 时出错: {e}")
            logger.error(f"错误详情: {traceback.format_exc()}")
            raise

    def execute_parse_task(self):
        """执行解析任务"""
        try:
            self.task.status = TaskStatus.RUNNING
            self.task.started_at = datetime.utcnow()
            self.task.progress = 5
            self.task.current_stage = "任务启动"

            # 更新数据库中的任务状态
            self.task_dao.update_parse_task(self.task.task_id, self.task.to_dict())

            logger.info(
                f"开始执行解析任务 {self.task.task_id}，文件: {self.task.file_path}"
            )

            # 文件预检查
            self.task.progress = 10
            self.task.current_stage = "文件检查"
            self.task_dao.update_parse_task(self.task.task_id, self.task.to_dict())

            # 再次校验文件
            self.validate_file(self.task.file_path, self.task.file_extension)

            # 在线程池中执行解析
            self.task.progress = 15
            self.task.current_stage = "准备解析"
            self.task_dao.update_parse_task(self.task.task_id, self.task.to_dict())

            # 判断文件后缀
            if self.task.file_extension in ['.html', '.htm']:
                processor = DocumentHTMLProcessor()
                result = processor.process_document(self.task)
            else:
                result = self._parse_file_sync()

            self.task.completed_at = datetime.utcnow()
            # 计算处理时间
            processing_time = (self.task.completed_at - self.task.started_at).total_seconds() if self.task.started_at and self.task.completed_at else 0

            logger.info(f"解析任务 {self.task.task_id} 完成，耗时: {processing_time:.2f}秒")
            
            self.task.result = result
            self.task.status = TaskStatus.COMPLETED
            self.task.progress = 100
            self.task.current_stage = "任务完成"
          
            self.task_dao.update_parse_task(self.task.task_id, self.task.to_dict())


        except Exception as e:
            self.task.status = TaskStatus.FAILED
            self.task.completed_at = datetime.utcnow()

            self.task.current_stage = "任务失败"
            self.task.error = f"解析失败: {str(e)}"
            
            self.task_dao.update_parse_task(self.task.task_id, self.task.to_dict())

            logger.error(f"解析任务 {self.task.task_id} 失败: {e}")
            logger.error(f"错误详情: {traceback.format_exc()}")

    