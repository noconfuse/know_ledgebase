#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
聊天会话和消息的数据访问对象
"""

import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import desc, and_

from models.chat_models import ChatSession, ChatMessage
from models.database import get_db

logger = logging.getLogger(__name__)

class ChatDAO:
    """聊天数据访问对象"""
    
    @staticmethod
    def create_session(session_id: str, index_ids: List[str], metadata: Dict[str, Any] = None) -> Optional[ChatSession]:
        """创建聊天会话"""
        try:
            with get_db() as db:
                # 检查会话是否已存在
                existing_session = db.query(ChatSession).filter(
                    ChatSession.session_id == session_id
                ).first()
                
                if existing_session:
                    # 更新现有会话
                    existing_session.index_ids = index_ids
                    existing_session.last_activity = datetime.utcnow()
                    existing_session.is_active = True
                    if metadata:
                        existing_session.session_metadata.update(metadata)
                    db.commit()
                    db.refresh(existing_session)
                    # 从会话中分离对象，避免Session绑定问题
                    db.expunge(existing_session)
                    return existing_session
                
                # 创建新会话
                chat_session = ChatSession(
                session_id=session_id,
                index_ids=index_ids,
                session_metadata=metadata or {}
            )
                
                db.add(chat_session)
                db.commit()
                db.refresh(chat_session)
                # 从会话中分离对象，避免Session绑定问题
                db.expunge(chat_session)
                
                logger.info(f"Created chat session: {session_id}")
                return chat_session
                
        except Exception as e:
            logger.error(f"Failed to create chat session {session_id}: {e}")
            return None
    
    @staticmethod
    def get_session(session_id: str) -> Optional[ChatSession]:
        """获取聊天会话"""
        try:
            with get_db() as db:
                session = db.query(ChatSession).filter(
                    and_(
                        ChatSession.session_id == session_id,
                        ChatSession.is_active == True
                    )
                ).first()
                if session:
                    # 从会话中分离对象，避免Session绑定问题
                    db.expunge(session)
                return session
        except Exception as e:
            logger.error(f"Failed to get chat session {session_id}: {e}")
            return None
    
    @staticmethod
    def update_session_activity(session_id: str) -> bool:
        """更新会话活动时间"""
        try:
            with get_db() as db:
                session = db.query(ChatSession).filter(
                    ChatSession.session_id == session_id
                ).first()
                
                if session:
                    session.last_activity = datetime.utcnow()
                    db.commit()
                    return True
                return False
        except Exception as e:
            logger.error(f"Failed to update session activity {session_id}: {e}")
            return False
    
    @staticmethod
    def deactivate_session(session_id: str) -> bool:
        """停用会话"""
        try:
            with get_db() as db:
                session = db.query(ChatSession).filter(
                    ChatSession.session_id == session_id
                ).first()
                
                if session:
                    session.is_active = False
                    session.last_activity = datetime.utcnow()
                    db.commit()
                    logger.info(f"Deactivated chat session: {session_id}")
                    return True
                return False
        except Exception as e:
            logger.error(f"Failed to deactivate session {session_id}: {e}")
            return False
    
    @staticmethod
    def list_active_sessions() -> List[ChatSession]:
        """列出所有活跃会话"""
        try:
            with get_db() as db:
                sessions = db.query(ChatSession).filter(
                    ChatSession.is_active == True
                ).order_by(desc(ChatSession.last_activity)).all()
                # 从会话中分离所有对象，避免Session绑定问题
                for session in sessions:
                    db.expunge(session)
                return sessions
        except Exception as e:
            logger.error(f"Failed to list active sessions: {e}")
            return []
    
    @staticmethod
    def add_message(session_id: str, role: str, content: str, metadata: Dict[str, Any] = None) -> Optional[ChatMessage]:
        """添加聊天消息"""
        try:
            with get_db() as db:
                # 验证会话存在
                session = db.query(ChatSession).filter(
                    ChatSession.session_id == session_id
                ).first()
                
                if not session:
                    logger.error(f"Session {session_id} not found")
                    return None
                
                # 创建消息
                message = ChatMessage(
            session_id=session_id,
            role=role,
            content=content,
            message_metadata=metadata or {}
        )
                
                db.add(message)
                
                # 更新会话活动时间
                session.last_activity = datetime.utcnow()
                
                db.commit()
                db.refresh(message)
                # 从会话中分离对象，避免Session绑定问题
                db.expunge(message)
                
                logger.debug(f"Added message to session {session_id}: {role}")
                return message
                
        except Exception as e:
            logger.error(f"Failed to add message to session {session_id}: {e}")
            return None
    
    @staticmethod
    def get_messages(session_id: str, limit: int = None, offset: int = 0) -> List[ChatMessage]:
        """获取会话消息"""
        try:
            with get_db() as db:
                query = db.query(ChatMessage).filter(
                    ChatMessage.session_id == session_id
                ).order_by(ChatMessage.timestamp)
                
                if offset > 0:
                    query = query.offset(offset)
                
                if limit and limit > 0:
                    query = query.limit(limit)
                
                messages = query.all()
                # 从会话中分离所有对象，避免Session绑定问题
                for message in messages:
                    db.expunge(message)
                return messages
        except Exception as e:
            logger.error(f"Failed to get messages for session {session_id}: {e}")
            return []
    
    @staticmethod
    def get_recent_messages(session_id: str, limit: int = 10) -> List[ChatMessage]:
        """获取最近的消息"""
        try:
            with get_db() as db:
                messages = db.query(ChatMessage).filter(
                    ChatMessage.session_id == session_id
                ).order_by(desc(ChatMessage.timestamp)).limit(limit).all()
                
                # 从会话中分离所有对象，避免Session绑定问题
                for message in messages:
                    db.expunge(message)
                
                # 按时间正序返回
                return list(reversed(messages))
        except Exception as e:
            logger.error(f"Failed to get recent messages for session {session_id}: {e}")
            return []
    
    @staticmethod
    def get_message_count(session_id: str) -> int:
        """获取会话消息数量"""
        try:
            with get_db() as db:
                count = db.query(ChatMessage).filter(
                    ChatMessage.session_id == session_id
                ).count()
                return count
        except Exception as e:
            logger.error(f"Failed to get message count for session {session_id}: {e}")
            return 0
    
    @staticmethod
    def delete_session_messages(session_id: str) -> bool:
        """删除会话的所有消息"""
        try:
            with get_db() as db:
                deleted_count = db.query(ChatMessage).filter(
                    ChatMessage.session_id == session_id
                ).delete()
                
                db.commit()
                logger.info(f"Deleted {deleted_count} messages from session {session_id}")
                return True
        except Exception as e:
            logger.error(f"Failed to delete messages for session {session_id}: {e}")
            return False
    
    @staticmethod
    def cleanup_expired_sessions(expire_hours: int = 24) -> int:
        """清理过期会话"""
        try:
            from datetime import timedelta
            
            cutoff_time = datetime.utcnow() - timedelta(hours=expire_hours)
            
            with get_db() as db:
                # 获取过期会话
                expired_sessions = db.query(ChatSession).filter(
                    and_(
                        ChatSession.last_activity < cutoff_time,
                        ChatSession.is_active == True
                    )
                ).all()
                
                count = 0
                for session in expired_sessions:
                    # 删除会话消息
                    db.query(ChatMessage).filter(
                        ChatMessage.session_id == session.session_id
                    ).delete()
                    
                    # 停用会话
                    session.is_active = False
                    count += 1
                
                db.commit()
                logger.info(f"Cleaned up {count} expired sessions")
                return count
                
        except Exception as e:
            logger.error(f"Failed to cleanup expired sessions: {e}")
            return 0