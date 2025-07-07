#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
任务相关的数据访问对象
"""

import logging
from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session, joinedload
from sqlalchemy.exc import SQLAlchemyError
from datetime import datetime

from models.parse_task import TaskStatus
from models.task_models import ParseTask, VectorStoreTask
from models.database import get_db

logger = logging.getLogger(__name__)

class TaskDAO:
    """任务数据访问对象"""
    
    def __init__(self, db: Session = None):
        self.db = db
    
    def _get_session(self) -> Session:
        """获取数据库会话"""
        if self.db:
            return self.db
        return next(get_db())
    
    # ==================== 解析任务相关操作 ====================
    
    def create_parse_task(self, parse_task: ParseTask) -> Optional[ParseTask]:
        """创建解析任务"""
        try:
            db = self._get_session()
            
            db.add(parse_task)
            
            # 如果有子任务，也一并添加
            if hasattr(parse_task, 'subtasks') and parse_task.subtasks:
                for subtask in parse_task.subtasks:
                    db.add(subtask)
            
            db.commit()
            db.refresh(parse_task)
            
            logger.info(f"创建解析任务成功: {parse_task.task_id}")
            return parse_task
            
        except SQLAlchemyError as e:
            logger.error(f"创建解析任务失败: {e}")
            if not self.db:  # 只有在自己创建的session时才回滚
                db.rollback()
            return None
        finally:
            if not self.db:
                db.close()
    
    def get_parse_task(self, task_id: str) -> Optional[ParseTask]:
        """获取解析任务"""
        try:
            db = self._get_session()
            task = (
                db.query(ParseTask)
                .options(joinedload(ParseTask.subtasks))
                .filter(ParseTask.task_id == task_id)
                .first()
            )
            return task
        except SQLAlchemyError as e:
            logger.error(f"获取解析任务失败: {e}")
            return None
        finally:
            if not self.db:
                db.close()
    
    def update_parse_task(self, task_id: str, update_data: Dict[str, Any]) -> bool:
        """更新解析任务"""
        try:
            db = self._get_session()
            
            task = db.query(ParseTask).filter(ParseTask.task_id == task_id).first()
            if not task:
                logger.warning(f"解析任务不存在: {task_id}")
                return False
            
            # 更新字段
            for key, value in update_data.items():
                if hasattr(task, key):
                    setattr(task, key, value)
            
            task.updated_at = datetime.utcnow()
            db.commit()
            
            logger.info(f"更新解析任务成功: {task_id}")
            return True
            
        except SQLAlchemyError as e:
            import traceback
            traceback.print_exc()
            logger.error(f"更新解析任务失败: {e}")
            if not self.db:
                db.rollback()
            return False
        finally:
            if not self.db:
                db.close()
    
    def list_parse_tasks(self, limit: int = 100, offset: int = 0, status: str = None) -> List[ParseTask]:
        """列出解析任务"""
        try:
            db = self._get_session()
            
            query = db.query(ParseTask)
            if status:
                query = query.filter(ParseTask.status == status)
            
            tasks = query.order_by(ParseTask.created_at.desc()).offset(offset).limit(limit).all()
            return tasks
            
        except SQLAlchemyError as e:
            logger.error(f"列出解析任务失败: {e}")
            return []
        finally:
            if not self.db:
                db.close()
    
    def delete_parse_task(self, task_id: str) -> bool:
        """删除解析任务"""
        try:
            db = self._get_session()
            
            task = db.query(ParseTask).filter(ParseTask.task_id == task_id).first()
            if not task:
                logger.warning(f"解析任务不存在: {task_id}")
                return False
            
            db.delete(task)
            db.commit()
            
            logger.info(f"删除解析任务成功: {task_id}")
            return True
            
        except SQLAlchemyError as e:
            logger.error(f"删除解析任务失败: {e}")
            if not self.db:
                db.rollback()
            return False
        finally:
            if not self.db:
                db.close()

    def get_all_parse_tasks(self) -> List[ParseTask]:
        """获取所有解析任务"""
        try:
            db = self._get_session()
            tasks = db.query(ParseTask).all()
            return tasks
        except SQLAlchemyError as e:
            logger.error(f"获取所有解析任务失败: {e}")
            return []
        finally:
            if not self.db:
                db.close()
    
    def get_parse_task_by_file_path(self, file_path: str) -> Optional[ParseTask]:
        """根据文件路径获取解析任务"""
        try:
            db = self._get_session()
            task = (
                db.query(ParseTask)
                .options(joinedload(ParseTask.subtasks))
                .filter(ParseTask.file_path == file_path)
                .first()
            )
            return task
        except SQLAlchemyError as e:
            logger.error(f"根据文件路径获取解析任务失败: {e}")
            return None
        finally:
            if not self.db:
                db.close()
    
    def reset_parse_task_status(self, task_id: str) -> bool:
        """重置解析任务状态为PENDING"""
        try:
            db = self._get_session()
            
            task = db.query(ParseTask).filter(ParseTask.task_id == task_id).first()
            if not task:
                logger.warning(f"解析任务不存在: {task_id}")
                return False
            
            # 重置任务状态
            task.status = TaskStatus.PENDING
            task.progress = 0
            task.current_stage = None
            task.stage_details = {}
            task.started_at = None
            task.completed_at = None
            task.result = None
            task.error = None
            task.processing_logs = []
            task.updated_at = datetime.utcnow()
            
            db.commit()
            
            logger.info(f"重置解析任务状态成功: {task_id}")
            return True
            
        except SQLAlchemyError as e:
            logger.error(f"重置解析任务状态失败: {e}")
            if not self.db:
                db.rollback()
            return False
        finally:
            if not self.db:
                db.close()
    
    # ==================== 向量化任务相关操作 ====================
    
    def create_vector_store_task(self, task_data: Dict[str, Any]) -> Optional[VectorStoreTask]:
        """创建向量化任务"""
        try:
            db = self._get_session()
            
            vector_task = VectorStoreTask(
                task_id=task_data['task_id'],
                parse_task_id=task_data.get('parse_task_id'),
                status=task_data.get('status', 'PENDING'),
                progress=task_data.get('progress', 0),
                config=task_data.get('config', {}),
                processed_files=task_data.get('processed_files', []),
                total_files=task_data.get('total_files', 0)
            )
            
            db.add(vector_task)
            db.commit()
            db.refresh(vector_task)
            
            logger.info(f"创建向量化任务成功: {task_data['task_id']}")
            return vector_task
            
        except SQLAlchemyError as e:
            logger.error(f"创建向量化任务失败: {e}")
            if not self.db:
                db.rollback()
            return None
        finally:
            if not self.db:
                db.close()
    
    def get_vector_store_task(self, task_id: str) -> Optional[VectorStoreTask]:
        """获取向量化任务"""
        try:
            db = self._get_session()
            task = db.query(VectorStoreTask).filter(VectorStoreTask.task_id == task_id).first()
            return task
        except SQLAlchemyError as e:
            logger.error(f"获取向量化任务失败: {e}")
            return None
        finally:
            if not self.db:
                db.close()
    
    def update_vector_store_task(self, task_id: str, update_data: Dict[str, Any]) -> bool:
        """更新向量化任务"""
        try:
            db = self._get_session()
            
            task = db.query(VectorStoreTask).filter(VectorStoreTask.task_id == task_id).first()
            if not task:
                logger.warning(f"向量化任务不存在: {task_id}")
                return False
            
            # 更新字段
            for key, value in update_data.items():
                if hasattr(task, key):
                    setattr(task, key, value)
            
            task.updated_at = datetime.utcnow()
            db.commit()
            
            logger.info(f"更新向量化任务成功: {task_id}")
            return True
            
        except SQLAlchemyError as e:
            logger.error(f"更新向量化任务失败: {e}")
            if not self.db:
                db.rollback()
            return False
        finally:
            if not self.db:
                db.close()
    
    def list_vector_store_tasks(self, limit: int = 100, offset: int = 0, status: str = None, parse_task_id: str = None) -> List[VectorStoreTask]:
        """列出向量化任务"""
        try:
            db = self._get_session()
            
            query = db.query(VectorStoreTask)
            if status:
                query = query.filter(VectorStoreTask.status == status)
            if parse_task_id:
                query = query.filter(VectorStoreTask.parse_task_id == parse_task_id)
            
            tasks = query.order_by(VectorStoreTask.created_at.desc()).offset(offset).limit(limit).all()
            return tasks
            
        except SQLAlchemyError as e:
            logger.error(f"列出向量化任务失败: {e}")
            return []
        finally:
            if not self.db:
                db.close()
    
    def delete_vector_store_task(self, task_id: str) -> bool:
        """删除向量化任务"""
        try:
            db = self._get_session()
            
            task = db.query(VectorStoreTask).filter(VectorStoreTask.task_id == task_id).first()
            if not task:
                logger.warning(f"向量化任务不存在: {task_id}")
                return False
            
            db.delete(task)
            db.commit()
            
            logger.info(f"删除向量化任务成功: {task_id}")
            return True
            
        except SQLAlchemyError as e:
            logger.error(f"删除向量化任务失败: {e}")
            if not self.db:
                db.rollback()
            return False
        finally:
            if not self.db:
                db.close()
    
    def get_subtasks_by_parent(self, parent_task_id: str) -> List[ParseTask]:
        """根据父任务ID获取所有子任务"""
        try:
            db = self._get_session()
            
            # 先获取父任务
            parent_task = db.query(ParseTask).filter(ParseTask.task_id == parent_task_id).first()
            if not parent_task:
                logger.warning(f"父任务不存在: {parent_task_id}")
                return []
            
            # 获取所有子任务
            subtasks = db.query(ParseTask).filter(ParseTask.parent_task_id == parent_task.id).all()
            return subtasks
            
        except SQLAlchemyError as e:
            logger.error(f"获取子任务失败: {e}")
            return []
        finally:
            if not self.db:
                db.close()
    
    def add_subtask_to_parent(self, parent_task_id: str, subtask_id: str) -> bool:
        """将子任务添加到父任务"""
        try:
            db = self._get_session()
            
            # 获取父任务和子任务
            parent_task = db.query(ParseTask).filter(ParseTask.task_id == parent_task_id).first()
            subtask = db.query(ParseTask).filter(ParseTask.task_id == subtask_id).first()
            
            if not parent_task:
                logger.error(f"父任务不存在: {parent_task_id}")
                return False
                
            if not subtask:
                logger.error(f"子任务不存在: {subtask_id}")
                return False
            
            # 设置父子关系
            subtask.parent_task_id = parent_task.id
            
            db.commit()
            logger.info(f"成功建立父子任务关系: {parent_task_id} -> {subtask_id}")
            return True
            
        except SQLAlchemyError as e:
            logger.error(f"建立父子任务关系失败: {e}")
            if not self.db:
                db.rollback()
            return False
        finally:
            if not self.db:
                db.close()
    
    # ==================== 关联查询 ====================
    
    def get_vector_tasks_by_parse_task(self, parse_task_id: str) -> List[VectorStoreTask]:
        """根据解析任务ID获取相关的向量化任务"""
        try:
            db = self._get_session()
            tasks = db.query(VectorStoreTask).filter(VectorStoreTask.parse_task_id == parse_task_id).all()
            return tasks
        except SQLAlchemyError as e:
            logger.error(f"获取关联向量化任务失败: {e}")
            return []
        finally:
            if not self.db:
                db.close()
    
    # ==================== 管理界面需要的方法 ====================
    
    def get_parse_tasks_count(self, status: str = None, task_type: str = None) -> int:
        """获取解析任务总数"""
        try:
            db = self._get_session()
            query = db.query(ParseTask)
            if status:
                query = query.filter(ParseTask.status == status)
            
            # 根据任务类型筛选
            if task_type == 'parent':
                query = query.filter(ParseTask.parent_task_id.is_(None))
            elif task_type == 'child':
                query = query.filter(ParseTask.parent_task_id.isnot(None))
            # task_type == 'all' 或 None 时不添加筛选条件
            
            count = query.count()
            return count
        except SQLAlchemyError as e:
            logger.error(f"获取解析任务总数失败: {e}")
            return 0
        finally:
            if not self.db:
                db.close()
    
    def get_vector_tasks_count(self, status: str = None) -> int:
        """获取向量化任务总数"""
        try:
            db = self._get_session()
            query = db.query(VectorStoreTask)
            if status:
                query = query.filter(VectorStoreTask.status == status)
            count = query.count()
            return count
        except SQLAlchemyError as e:
            logger.error(f"获取向量化任务总数失败: {e}")
            return 0
        finally:
            if not self.db:
                db.close()
    
    def get_all_vector_store_tasks(self, limit: int = 100, offset: int = 0) -> List[VectorStoreTask]:
        """获取所有向量化任务（重定向到 list_vector_store_tasks 以避免重复实现）"""
        return self.list_vector_store_tasks(limit=limit, offset=offset)
    
    def list_parse_tasks_with_parent_info(self, limit: int = 100, offset: int = 0, status: str = None, task_type: str = None) -> List[ParseTask]:
        """列出解析任务（带父任务信息）"""
        try:
            db = self._get_session()
            
            # 创建主查询
            query = db.query(ParseTask)
            
            # 添加状态筛选
            if status:
                query = query.filter(ParseTask.status == status)
            
            # 根据任务类型筛选
            if task_type == 'parent':
                query = query.filter(ParseTask.parent_task_id.is_(None))
            elif task_type == 'child':
                query = query.filter(ParseTask.parent_task_id.isnot(None))
            # task_type == 'all' 或 None 时不添加筛选条件
            
            # 执行查询
            tasks = query.order_by(ParseTask.created_at.desc()).offset(offset).limit(limit).all()
            
            # 为每个任务添加父任务信息和是否为父任务的标识
            for task in tasks:
                # 添加父任务信息
                if task.parent_task_id:
                    parent_task = db.query(ParseTask).filter(ParseTask.id == task.parent_task_id).first()
                    if parent_task:
                        task.parent_task_info = {
                            'task_id': parent_task.task_id,
                            'file_path': parent_task.file_path,
                            'status': parent_task.status.value if hasattr(parent_task.status, 'value') else str(parent_task.status)
                        }
                    else:
                        task.parent_task_info = None
                else:
                    task.parent_task_info = None
                
                # 检查是否为父任务（是否有子任务）
                child_count = db.query(ParseTask).filter(ParseTask.parent_task_id == task.id).count()
                task.is_parent = child_count > 0
            
            return tasks
            
        except SQLAlchemyError as e:
            logger.error(f"列出解析任务（带父任务信息）失败: {e}")
            return []
        finally:
            if not self.db:
                db.close()