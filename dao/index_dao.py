#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
索引信息数据访问对象
"""

import logging
from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session
from models.index_models import IndexInfo

logger = logging.getLogger(__name__)

class IndexDAO:
    """索引信息数据访问对象"""
    
    @staticmethod
    def create_index(
        db: Session, 
        index_id: str, 
        index_description: Optional[str] = None,
        file_md5: Optional[str] = None,
        file_path: Optional[str] = None,
        file_name: Optional[str] = None,
        file_size: Optional[int] = None,
        file_extension: Optional[str] = None,
        mime_type: Optional[str] = None,
        document_count: Optional[int] = None,
        node_count: Optional[int] = None,
        vector_dimension: Optional[int] = None,
        processing_config: Optional[Dict[str, Any]] = None
    ) -> IndexInfo:
        """创建索引信息"""
        try:
            index_info = IndexInfo(
                index_id=index_id,
                index_description=index_description,
                file_md5=file_md5,
                file_path=file_path,
                file_name=file_name,
                file_size=file_size,
                file_extension=file_extension,
                mime_type=mime_type,
                document_count=document_count,
                node_count=node_count,
                vector_dimension=vector_dimension,
                processing_config=processing_config
            )
            db.add(index_info)
            db.commit()
            db.refresh(index_info)
            logger.info(f"Created index info: {index_id}")
            return index_info
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to create index info: {e}")
            raise
    
    @staticmethod
    def get_index_by_id(db: Session, index_id: str) -> Optional[IndexInfo]:
        """根据索引ID获取索引信息"""
        try:
            return db.query(IndexInfo).filter(IndexInfo.index_id == index_id).first()
        except Exception as e:
            logger.error(f"Failed to get index info: {e}")
            return None
    
    @staticmethod
    def get_index_by_file_md5(db: Session, file_md5: str) -> Optional[IndexInfo]:
        """根据文件MD5获取索引信息"""
        try:
            return db.query(IndexInfo).filter(IndexInfo.file_md5 == file_md5).first()
        except Exception as e:
            logger.error(f"Failed to get index info by file MD5: {e}")
            return None
    
    @staticmethod
    def get_all_indexes(db: Session) -> List[IndexInfo]:
        """获取所有索引信息"""
        try:
            return db.query(IndexInfo).all()
        except Exception as e:
            logger.error(f"Failed to get all index info: {e}")
            return []
    
    @staticmethod
    def update_index_description(db: Session, index_id: str, index_description: str) -> Optional[IndexInfo]:
        """更新索引描述"""
        try:
            index_info = db.query(IndexInfo).filter(IndexInfo.index_id == index_id).first()
            if index_info:
                index_info.index_description = index_description
                db.commit()
                db.refresh(index_info)
                logger.info(f"Updated index description for: {index_id}")
                return index_info
            return None
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to update index description: {e}")
            return None
    
    @staticmethod
    def delete_index(db: Session, index_id: str) -> bool:
        """删除索引信息"""
        try:
            index_info = db.query(IndexInfo).filter(IndexInfo.index_id == index_id).first()
            if index_info:
                db.delete(index_info)
                db.commit()
                logger.info(f"Deleted index info: {index_id}")
                return True
            return False
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to delete index info: {e}")
            return False