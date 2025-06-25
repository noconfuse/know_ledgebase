#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
任务相关的数据库模型
"""

import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List
from sqlalchemy import Column, String, DateTime, Text, Integer, JSON, Boolean, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship, backref

Base = declarative_base()

class ParseTask(Base):
    """解析任务模型"""
    __tablename__ = 'parse_tasks'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    task_id = Column(String(255), unique=True, nullable=False, index=True)
    file_path = Column(Text, nullable=False)  # 原始文件路径
    file_name = Column(String(255), nullable=False)  # 文件名
    file_size = Column(Integer, nullable=True)  # 文件大小
    file_extension = Column(String(50), nullable=True)  # 文件扩展名
    mime_type = Column(String(255), nullable=True)  # MIME类型
    parser_type = Column(String(100), nullable=True)  # 解析器类型
    
    # 任务状态
    status = Column(String(50), nullable=False, default='PENDING')  # PENDING, RUNNING, COMPLETED, FAILED
    progress = Column(Integer, default=0)  # 进度百分比
    current_stage = Column(String(255), nullable=True)  # 当前阶段
    stage_details = Column(JSON, default=dict)  # 阶段详情
    
    # 时间戳
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # 结果和错误信息
    result = Column(JSON, nullable=True)  # 解析结果
    error = Column(Text, nullable=True)  # 错误信息
    processing_logs = Column(JSON, default=list)  # 处理日志
    
    # 输出信息
    output_directory = Column(Text, nullable=True)  # 输出目录
    output_files = Column(JSON, default=list)  # 输出文件列表
    
    # 配置信息
    config = Column(JSON, default=dict)  # 任务配置
    
    # 关联的向量化任务
    vector_tasks = relationship("VectorStoreTask", back_populates="parse_task")

    # 子任务列表，通过parent_task_id关联
    subtasks = relationship("ParseTask", 
                          backref=backref("parent_task", remote_side=[id]),
                          foreign_keys="ParseTask.parent_task_id",
                          cascade="all, delete-orphan")
    
    # 父任务ID，如果为空则表示这是顶层任务
    parent_task_id = Column(UUID(as_uuid=True), 
                           ForeignKey('parse_tasks.id', ondelete='CASCADE'), 
                           nullable=True,
                           index=True)
    
    def __repr__(self):
        return f"<ParseTask(task_id='{self.task_id}', status='{self.status}', file_name='{self.file_name}')>"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": str(self.id),
            "task_id": self.task_id,
            "file_path": self.file_path,
            "file_name": self.file_name,
            "file_size": self.file_size,
            "file_extension": self.file_extension,
            "mime_type": self.mime_type,
            "parser_type": self.parser_type,
            "status": self.status,
            "progress": self.progress,
            "current_stage": self.current_stage,
            "stage_details": self.stage_details,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "updated_at": self.updated_at,
            "result": self.result,
            "error": self.error,
            "processing_logs": self.processing_logs,
            "output_directory": self.output_directory,
            "output_files": self.output_files,
            "config": self.config
        }

class VectorStoreTask(Base):
    """向量数据库构建任务模型"""
    __tablename__ = 'vector_store_tasks'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    task_id = Column(String(255), unique=True, nullable=False, index=True)
    
    # 关联的解析任务
    parse_task_id = Column(String(255), ForeignKey('parse_tasks.task_id'), nullable=True, index=True)
    parse_task = relationship("ParseTask", back_populates="vector_tasks")
    
    # 输入信息
    
    # 任务状态
    status = Column(String(50), nullable=False, default='PENDING')  # PENDING, RUNNING, COMPLETED, FAILED
    progress = Column(Integer, default=0)  # 进度百分比
    
    # 时间戳
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # 结果和错误信息
    result = Column(JSON, nullable=True)  # 构建结果
    error = Column(Text, nullable=True)  # 错误信息
    
    # 处理统计
    processed_files = Column(JSON, default=list)  # 已处理文件列表
    total_files = Column(Integer, default=0)  # 总文件数
    total_documents = Column(Integer, default=0)  # 总文档数
    total_nodes = Column(Integer, default=0)  # 总节点数
    
    # 向量数据库信息
    index_id = Column(String(255), nullable=True)  # 生成的索引ID
    vector_store_path = Column(Text, nullable=True)  # 向量数据库存储路径
    
    # 配置信息
    config = Column(JSON, default=dict)  # 任务配置
    
    def __repr__(self):
        return f"<VectorStoreTask(task_id='{self.task_id}', status='{self.status}', index_id='{self.index_id}')>"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": str(self.id),
            "task_id": self.task_id,
            "parse_task_id": self.parse_task_id,

            "status": self.status,
            "progress": self.progress,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "updated_at": self.updated_at,
            "result": self.result,
            "error": self.error,
            "processed_files": self.processed_files,
            "total_files": self.total_files,
            "total_documents": self.total_documents,
            "total_nodes": self.total_nodes,
            "index_id": self.index_id,
            "vector_store_path": self.vector_store_path,
            "config": self.config
        }