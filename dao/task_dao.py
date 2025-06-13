#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
任务相关的数据访问对象
"""

import logging
from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from datetime import datetime

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
    
    def create_parse_task(self, task_data: Dict[str, Any]) -> Optional[ParseTask]:
        """创建解析任务"""
        try:
            db = self._get_session()
            
            parse_task = ParseTask(
                task_id=task_data['task_id'],
                file_path=task_data['file_path'],
                file_name=task_data.get('file_name', ''),
                file_size=task_data.get('file_size'),
                file_extension=task_data.get('file_extension'),
                mime_type=task_data.get('mime_type'),
                parser_type=task_data.get('parser_type'),
                status=task_data.get('status', 'PENDING'),
                progress=task_data.get('progress', 0),
                current_stage=task_data.get('current_stage'),
                stage_details=task_data.get('stage_details', {}),
                config=task_data.get('config', {}),
                processing_logs=task_data.get('processing_logs', [])
            )
            
            db.add(parse_task)
            db.commit()
            db.refresh(parse_task)
            
            logger.info(f"创建解析任务成功: {task_data['task_id']}")
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
            task = db.query(ParseTask).filter(ParseTask.task_id == task_id).first()
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