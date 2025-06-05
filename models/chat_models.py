#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
聊天会话和消息的数据库模型
"""

import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any
from sqlalchemy import Column, String, DateTime, Text, Integer, JSON, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID

Base = declarative_base()

class ChatSession(Base):
    """聊天会话模型"""
    __tablename__ = 'chat_sessions'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(String(255), unique=True, nullable=False, index=True)
    user_id = Column(UUID(as_uuid=True), nullable=False, index=True)  # 关联用户ID
    index_ids = Column(JSON, nullable=False)  # 存储索引ID列表
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_activity = Column(DateTime, default=datetime.utcnow, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    session_metadata = Column(JSON, default=dict)  # 存储额外的会话元数据
    
    # 关联聊天消息
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<ChatSession(session_id='{self.session_id}', created_at='{self.created_at}')>"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": str(self.id),
            "session_id": self.session_id,
            "user_id": str(self.user_id),
            "index_ids": self.index_ids,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_activity": self.last_activity.isoformat() if self.last_activity else None,
            "is_active": self.is_active,
            "session_metadata": self.session_metadata,
            "message_count": len(self.messages) if self.messages else 0
        }

class ChatMessage(Base):
    """聊天消息模型"""
    __tablename__ = 'chat_messages'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(String(255), ForeignKey('chat_sessions.session_id'), nullable=False, index=True)
    role = Column(String(50), nullable=False)  # 'user' 或 'assistant'
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    message_metadata = Column(JSON, nullable=True)  # 存储额外的元数据信息
    
    # 关联聊天会话
    session = relationship("ChatSession", back_populates="messages")
    
    def __repr__(self):
        return f"<ChatMessage(session_id='{self.session_id}', role='{self.role}', timestamp='{self.timestamp}')>"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": str(self.id),
            "session_id": self.session_id,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "message_metadata": self.message_metadata
        }