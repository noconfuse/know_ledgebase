#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
索引信息的数据库模型
"""

import uuid
from datetime import datetime
from typing import Optional, Dict, Any
from sqlalchemy import Column, String, DateTime, Text, UUID, BigInteger, Integer
from sqlalchemy.dialects.postgresql import JSON
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class IndexInfo(Base):
    """索引信息模型"""
    __tablename__ = 'indexes'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    index_id = Column(String(255), unique=True, nullable=False, index=True)
    index_description = Column(Text, nullable=True)  # 索引描述

    origin_file_path = Column(Text, nullable=True, index=True)  # 原始文件路径, 可能是目录或文件

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # 文档和节点统计
    document_count = Column(Integer, nullable=True)  # 文档数量
    node_count = Column(Integer, nullable=True)  # 节点数量
    vector_dimension = Column(Integer, nullable=True)  # 向量维度
    
    # 处理配置
    processing_config = Column(JSON, nullable=True)  # 处理配置信息
    
    def __repr__(self):
        return f"<IndexInfo(index_id='{self.index_id}', created_at='{self.created_at}')>"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": str(self.id),
            "index_id": self.index_id,
            "index_description": self.index_description,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "file_md5": self.file_md5,
            "file_path": self.file_path,
            "file_name": self.file_name,
            "file_size": self.file_size,
            "file_extension": self.file_extension,
            "mime_type": self.mime_type,
            "document_count": self.document_count,
            "node_count": self.node_count,
            "vector_dimension": self.vector_dimension,
            "processing_config": self.processing_config
        }