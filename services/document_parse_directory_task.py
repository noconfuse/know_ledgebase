import asyncio
from datetime import datetime
from pathlib import Path
import time
from typing import Dict, Optional,Any
import uuid
from config import settings
from dao.task_dao import TaskDAO
from models.parse_task import TaskStatus
from models.task_models import ParseTask
from utils.logging_config import setup_logging, get_logger

from services.document_parse_task import DocumentParseTask
setup_logging()
logger = get_logger(__name__)

class DocumentParseDirectoryTask(DocumentParseTask):

    @staticmethod
    def create_parse_directory_task(directory_path: str, config: Optional[Dict[str, Any]] = None) -> ParseTask:
         # 验证目录是否存在
        directory = Path(directory_path)
        if not directory.exists() or not directory.is_dir():
            raise NotADirectoryError(f"目录不存在或不是有效目录: {directory_path}")
            
        logger.info(f"开始解析目录: {directory_path}")

        parser_type = config.get("parser_type") if config else settings.DEFAULT_PARSER_TYPE
        
        task_dao = TaskDAO()
        
        # 检查数据库中是否已存在相同目录路径的任务
        existing_task = task_dao.get_parse_task_by_file_path(directory_path)
        
        if existing_task:
            logger.info(f"发现现有目录任务 {existing_task.task_id}，目录路径: {directory_path}，复用并重置状态")
            
            # 更新任务配置和解析器类型
            update_data = {
                'parser_type': parser_type,
                'config': config or {}
            }
            task_dao.update_parse_task(existing_task.task_id, update_data)
            
            # 重置任务状态
            task_dao.reset_parse_task_status(existing_task.task_id)
            
            # 重新获取更新后的任务
            updated_task = task_dao.get_parse_task(existing_task.task_id)
            
            # 获取目录下所有支持的一级文件
            supported_files = []
            for file_path in directory.glob("*"):
                if file_path.is_file() and file_path.suffix.lower() in settings.SUPPORTED_FORMATS:
                    supported_files.append(str(file_path))
            
            if not supported_files:
                raise ValueError(f"目录中未找到支持的文件格式: {directory_path}")

            logger.info(f"支持的文件: {', '.join(supported_files)}")
            
            # 为新文件创建子任务，传递主任务作为父任务
            for file_path in supported_files:
                sub_task_obj = DocumentParseDirectoryTask.create_parse_task(file_path, config, parent_task=updated_task)
            
            return updated_task or existing_task
        
        # 如果不存在现有任务，创建新的目录任务
        logger.info(f"未发现现有目录任务，为目录路径 {directory_path} 创建新任务")
        
        # 创建主任务ID
        main_task_id = str(uuid.uuid4())
        
        # 获取目录下所有支持的一级文件
        supported_files = []
        for file_path in directory.glob("*"):
            if file_path.is_file() and file_path.suffix.lower() in settings.SUPPORTED_FORMATS:
                supported_files.append(str(file_path))
        
        if not supported_files:
            raise ValueError(f"目录中未找到支持的文件格式: {directory_path}")

        logger.info(f"支持的文件: {', '.join(supported_files)}")
        
        # 创建目录解析的主任务记录
        main_task = ParseTask(
            task_id=main_task_id,
            file_path=directory_path,
            file_name=directory.name,
            parser_type=parser_type or settings.DEFAULT_PARSER_TYPE,
            status=TaskStatus.PENDING,
        )
        
        # 先保存主任务到数据库
        main_task = task_dao.create_parse_task(main_task)

        # 创建子任务，传递主任务作为父任务
        for file_path in supported_files:
            sub_task_obj = DocumentParseDirectoryTask.create_parse_task(file_path, config, parent_task=main_task)

        return main_task

    async def execute_parse_directory_task(self, executor):
        # 控制并发数量
        semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_PARSES_PER_DIRECTORY)
        # 创建子任务并跟踪
        subtasks_coros = []

        self.task.status = TaskStatus.RUNNING
        self.task.started_at = datetime.utcnow()
        self.task.progress = 5
        self.task.current_stage = "任务启动"

        # 更新数据库中的任务状态
        self.task_dao.update_parse_task(self.task.task_id, self.task.to_dict())

        logger.info(
                f"开始执行目录解析任务 {self.task.task_id}，目录: {self.task.file_path}"
            )

        for sub_task in self.task.subtasks:
            async def parse_single_document(task):
                async with semaphore:
                    try:
                        document_parse_task = DocumentParseTask(task.task_id)
                        await asyncio.get_event_loop().run_in_executor(
                            executor, document_parse_task.execute_parse_task
                        )
                        logger.debug(f"创建子任务 {task.task_id} 用于文件: {task.file_path}")
                        return {"task_id": task.task_id, "file_path": task.file_path}
                    except Exception as e:
                        logger.error(f"处理文件 {task.file_path} 时出错: {e}")
                        return None
            subtasks_coros.append(parse_single_document(sub_task))
        
        # 等待所有子任务完成
        results = await asyncio.gather(*subtasks_coros)
        subtasks = [r for r in results if r is not None]

        asyncio.create_task(self._monitor_directory_task())

    
    async def _monitor_directory_task(self):
        """异步监控目录解析任务的子任务进度和状态"""
        logger.info(f"开始监控目录解析任务: {self.task.task_id}")
        
        # 获取子任务ID列表
        subtasks = self.task.subtasks
        
        if not subtasks:
            logger.warning(f"目录解析任务 {self.task.task_id} 没有子任务，直接标记为完成。")
            self.task.status = TaskStatus.COMPLETED
            self.task.progress = 100
            self.task.completed_at = datetime.utcnow()
            self.task_dao.update_parse_task(self.task.task_id, self.task.to_dict())
            return
        
        total_subtasks = len(subtasks)
        completed_subtasks = 0
        failed_subtasks = 0

        while completed_subtasks + failed_subtasks < total_subtasks:
            await asyncio.sleep(settings.TASK_MONITOR_INTERVAL_SECONDS)  # 定期检查
            
            current_completed = 0
            current_failed = 0
            
            # 从数据库重新加载主任务以获取最新的子任务状态
            reloaded_task = self.task_dao.get_parse_task(self.task.task_id)
            if not reloaded_task:
                logger.error(f"监控任务 {self.task.task_id} 失败: 无法从数据库加载任务。")
                break # 退出循环，避免无限等待
            
            # 遍历子任务，检查其状态
            for sub_task in reloaded_task.subtasks:
                if sub_task:
                    if sub_task.status == TaskStatus.COMPLETED:
                        current_completed += 1
                    elif sub_task.status == TaskStatus.FAILED:
                        current_failed += 1
            
            # 更新已完成/失败的子任务计数
            completed_subtasks = current_completed
            failed_subtasks = current_failed

            # 更新主任务进度
            self.task.progress = int(((completed_subtasks + failed_subtasks) / total_subtasks) * 100)
            self.task.current_stage = f"处理中 ({completed_subtasks}/{total_subtasks} 完成)"
            self.task_dao.update_parse_task(self.task.task_id, self.task.to_dict())
            logger.debug(f"目录任务 {self.task.task_id} 进度: {self.task.progress}%")

        # 所有子任务处理完毕，更新主任务最终状态
        self.task.completed_at = datetime.utcnow()
        if failed_subtasks > 0:
            self.task.status = TaskStatus.FAILED
            self.task.error = f"{failed_subtasks} 个子任务失败。"
            self.task.current_stage = "任务失败"
            logger.error(f"目录任务 {self.task.task_id} 失败: {failed_subtasks} 个子任务失败。")
        else:
            self.task.status = TaskStatus.COMPLETED
            self.task.current_stage = "任务完成"
            logger.info(f"目录任务 {self.task.task_id} 成功完成。")
        
        self.task.progress = 100
        self.task_dao.update_parse_task(self.task.task_id, self.task.to_dict())
        
    
