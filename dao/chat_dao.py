#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
聊天会话和消息的数据访问对象
"""

import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import desc, and_, asc
from config import settings
from datetime import timedelta
from models.chat_models import ChatSession, ChatMessage
from models.database import get_db

logger = logging.getLogger(__name__)

class ChatDAO:
    @staticmethod
    def clear_chat_history(session_id: str, user_id: str) -> bool:
        """清除会话的聊天记录"""
        db_gen = None
        db = None
        try:
            db_gen = get_db()
            db = next(db_gen)
            # 首先验证会话是否属于该用户
            session = db.query(ChatSession).filter(
                and_(
                    ChatSession.session_id == session_id,
                    ChatSession.user_id == user_id
                )
            ).first()

            if not session:
                logger.warning(f"Attempt to clear history for non-existent or unauthorized session {session_id} by user {user_id}")
                return False

            # 删除与会话关联的所有消息
            num_deleted = db.query(ChatMessage).filter(
                ChatMessage.session_id == session_id
            ).delete(synchronize_session=False)
            
            db.commit()
            logger.info(f"Cleared {num_deleted} messages for session {session_id}")
            return True
        except Exception as e:
            if db:
                db.rollback()
            logger.error(f"Error clearing chat history for session {session_id}: {e}")
            return False
        finally:
            if db_gen:
                try:
                    next(db_gen)
                except StopIteration:
                    pass
    """聊天数据访问对象"""
    
    @staticmethod
    def create_session(session_id: str, user_id: str, index_ids: List[str], metadata: Dict[str, Any] = None) -> Optional[ChatSession]:
        """创建聊天会话"""
        db_gen = None
        db = None
        try:
            db_gen = get_db()
            db = next(db_gen)
            # 检查会话是否已存在
            existing_session = db.query(ChatSession).filter(
                ChatSession.session_id == session_id
            ).first()
            
            if existing_session:
                # 更新现有会话
                existing_session.user_id = user_id
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
                 user_id=user_id,
                 index_ids=index_ids,
                 session_metadata=metadata or {}
             )
            
            db.add(chat_session)
            db.commit()
            db.refresh(chat_session)
            # 从会话中分离对象，避免Session绑定问题
            db.expunge(chat_session)
            
            logger.info(f"Created chat session: {session_id} for user: {user_id}")
            return chat_session
                
        except Exception as e:
            if db:
                db.rollback()
            import traceback
            traceback.print_exc()
            logger.error(f"Failed to create chat session {session_id}: {e}")
            return None
        finally:
            if db_gen:
                try:
                    next(db_gen)
                except StopIteration:
                    pass
    
    @staticmethod
    def get_session(session_id: str) -> Optional[ChatSession]:
        """获取聊天会话"""
        db_gen = None
        db = None
        try:
            db_gen = get_db()
            db = next(db_gen)
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
            if db:
                db.rollback()
            logger.error(f"Failed to get chat session {session_id}: {e}")
            return None
        finally:
            if db_gen:
                try:
                    next(db_gen)
                except StopIteration:
                    pass
    
    @staticmethod
    def update_session_activity(session_id: str) -> bool:
        """更新会话活动时间"""
        db_gen = None
        db = None
        try:
            db_gen = get_db()
            db = next(db_gen)
            session = db.query(ChatSession).filter(
                ChatSession.session_id == session_id
            ).first()
            
            if session:
                session.last_activity = datetime.utcnow()
                db.commit()
                return True
            return False
        except Exception as e:
            if db:
                db.rollback()
            logger.error(f"Failed to update session activity {session_id}: {e}")
            return False
        finally:
            if db_gen:
                try:
                    next(db_gen)
                except StopIteration:
                    pass
    
    @staticmethod
    def get_session_by_user(session_id: str, user_id: str) -> Optional[ChatSession]:
        """根据会话ID和用户ID获取会话（确保用户只能访问自己的会话）"""
        db_gen = None
        db = None
        try:
            db_gen = get_db()
            db = next(db_gen)
            session = db.query(ChatSession).filter(
                and_(
                    ChatSession.session_id == session_id,
                    ChatSession.user_id == user_id,
                    ChatSession.is_active == True
                )
            ).first()
            if session:
                # 从会话中分离对象，避免Session绑定问题
                db.expunge(session)
            return session
        except Exception as e:
            if db:
                db.rollback()
            logger.error(f"Error getting session by user: {e}")
            return None
        finally:
            if db_gen:
                try:
                    next(db_gen)
                except StopIteration:
                    pass
    
    @staticmethod
    def delete_user_session(session_id: str, user_id: str) -> bool:
        """删除用户的会话"""
        db_gen = None
        db = None
        try:
            db_gen = get_db()
            db = next(db_gen)
            session = db.query(ChatSession).filter(
                and_(
                    ChatSession.session_id == session_id,
                    ChatSession.user_id == user_id
                )
            ).first()
                
            if session:
                session.is_active = False
                db.commit()
                logger.info(f"Deleted session {session_id} for user {user_id}")
                return True
            return False
        except Exception as e:
            if db:
                db.rollback()
            logger.error(f"Error deleting user session: {e}")
            return False
        finally:
            if db_gen:
                try:
                    next(db_gen)
                except StopIteration:
                    pass
    
    @staticmethod
    def deactivate_session(session_id: str) -> bool:
        """停用会话"""
        db_gen = None
        db = None
        try:
            db_gen = get_db()
            db = next(db_gen)
            session = db.query(ChatSession).filter(
                ChatSession.session_id == session_id
            ).first()
            
            if session:
                session.is_active = False
                db.commit()
                logger.info(f"Deactivated session: {session_id}")
                return True
            return False
        except Exception as e:
            if db:
                db.rollback()
            logger.error(f"Failed to deactivate session {session_id}: {e}")
            return False
        finally:
            if db_gen:
                try:
                    next(db_gen)
                except StopIteration:
                    pass
    
    @staticmethod
    def list_active_sessions() -> List[ChatSession]:
        """列出所有活跃会话"""
        db_gen = None
        db = None
        try:
            db_gen = get_db()
            db = next(db_gen)
            sessions = db.query(ChatSession).filter(
                ChatSession.is_active == True
            ).all()
            # 从会话中分离对象，避免Session绑定问题
            for session in sessions:
                db.expunge(session)
            return sessions
        except Exception as e:
            if db:
                db.rollback()
            logger.error(f"Failed to list active sessions: {e}")
            return []
        finally:
            if db_gen:
                try:
                    next(db_gen)
                except StopIteration:
                    pass
    
    @staticmethod
    def get_user_sessions(user_id: str = None, limit: int = None) -> List[ChatSession]:
        """获取用户会话"""
        db_gen = None
        db = None
        try:
            db_gen = get_db()
            db = next(db_gen)
            query = db.query(ChatSession).filter(
                and_(
                    ChatSession.is_active == True,
                    ChatSession.soft_deleted_at.is_(None)
                )
            )
            
            # 如果指定了用户ID，则只获取该用户的会话
            if user_id:
                query = query.filter(ChatSession.user_id == user_id)
            
            if limit:
                query = query.limit(limit)
                
            sessions = query.order_by(desc(ChatSession.last_activity)).all()
            # 从会话中分离所有对象，避免Session绑定问题
            for session in sessions:
                db.expunge(session)
            return sessions
        except Exception as e:
            if db:
                db.rollback()
            logger.error(f"Error getting user sessions: {e}")
            return []
        finally:
            if db_gen:
                try:
                    next(db_gen)
                except StopIteration:
                    pass
    
    @staticmethod
    def add_message(session_id: str, role: str, content: str, metadata: Dict[str, Any] = None) -> Optional[ChatMessage]:
        """添加聊天消息"""
        db_gen = None
        db = None
        try:
            db_gen = get_db()
            db = next(db_gen)
            # 验证会话存在
            session = db.query(ChatSession).filter(
                and_(
                    ChatSession.session_id == session_id,
                    ChatSession.is_active == True
                )
            ).first()
            
            if not session:
                logger.error(f"Session {session_id} not found or inactive")
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
            
            logger.info(f"Added message to session {session_id}")
            return message
            
        except Exception as e:
            if db:
                db.rollback()
            logger.error(f"Failed to add message to session {session_id}: {e}")
            return None
        finally:
            if db_gen:
                try:
                    next(db_gen)
                except StopIteration:
                    pass
    
    @staticmethod
    def get_messages(session_id: str, limit: int = None, offset: int = 0) -> List[ChatMessage]:
        """获取会话消息"""
        db_gen = None
        db = None
        try:
            db_gen = get_db()
            db = next(db_gen)
            query = db.query(ChatMessage).filter(
                ChatMessage.session_id == session_id
            ).order_by(asc(ChatMessage.timestamp))
            
            if offset > 0:
                query = query.offset(offset)
            
            if limit:
                query = query.limit(limit)
            
            messages = query.all()
            # 从会话中分离对象，避免Session绑定问题
            for message in messages:
                db.expunge(message)
            return messages
        except Exception as e:
            if db:
                db.rollback()
            logger.error(f"Failed to get messages for session {session_id}: {e}")
            return []
        finally:
            if db_gen:
                try:
                    next(db_gen)
                except StopIteration:
                    pass
    
    @staticmethod
    def get_recent_messages(session_id: str, limit: int = 10) -> List[ChatMessage]:
        """获取最近的消息"""
        db_gen = None
        db = None
        try:
            db_gen = get_db()
            db = next(db_gen)
            messages = db.query(ChatMessage).filter(
                ChatMessage.session_id == session_id
            ).order_by(desc(ChatMessage.timestamp)).limit(limit).all()
            
            # 从会话中分离对象，避免Session绑定问题
            for message in messages:
                db.expunge(message)
            return messages
        except Exception as e:
            if db:
                db.rollback()
            logger.error(f"Failed to get recent messages for session {session_id}: {e}")
            return []
        finally:
            if db_gen:
                try:
                    next(db_gen)
                except StopIteration:
                    pass
    
    @staticmethod
    def get_message_count(session_id: str) -> int:
        """获取会话消息数量"""
        db_gen = None
        db = None
        try:
            db_gen = get_db()
            db = next(db_gen)
            count = db.query(ChatMessage).filter(
                ChatMessage.session_id == session_id
            ).count()
            return count
        except Exception as e:
            if db:
                db.rollback()
            logger.error(f"Failed to get message count for session {session_id}: {e}")
            return 0
        finally:
            if db_gen:
                try:
                    next(db_gen)
                except StopIteration:
                    pass
    
    @staticmethod
    def delete_session_messages(session_id: str) -> bool:
        """删除会话的所有消息"""
        db_gen = None
        db = None
        try:
            db_gen = get_db()
            db = next(db_gen)
            deleted_count = db.query(ChatMessage).filter(
                ChatMessage.session_id == session_id
            ).delete()
            
            db.commit()
            logger.info(f"Deleted {deleted_count} messages from session {session_id}")
            return True
        except Exception as e:
            if db:
                db.rollback()
            logger.error(f"Failed to delete messages for session {session_id}: {e}")
            return False
        finally:
            if db_gen:
                try:
                    next(db_gen)
                except StopIteration:
                    pass
    
    @staticmethod
    def cleanup_expired_sessions() -> dict:
        """清理过期会话：30天软删除，180天硬删除"""
       
        
        db_gen = None
        db = None
        result = {"soft_deleted": 0, "hard_deleted": 0}
        
        try:
            current_time = datetime.utcnow()
            soft_delete_cutoff = current_time - timedelta(days=settings.SESSION_SOFT_DELETE_DAYS)
            hard_delete_cutoff = current_time - timedelta(days=settings.SESSION_HARD_DELETE_DAYS)
            
            db_gen = get_db()
            db = next(db_gen)
            
            # 1. 软删除：30天未活动且未被软删除的会话
            sessions_to_soft_delete = db.query(ChatSession).filter(
                and_(
                    ChatSession.last_activity < soft_delete_cutoff,
                    ChatSession.soft_deleted_at.is_(None)
                )
            ).all()
            
            for session in sessions_to_soft_delete:
                session.soft_deleted_at = current_time
                result["soft_deleted"] += 1
            
            # 2. 硬删除：180天前软删除的会话
            sessions_to_hard_delete = db.query(ChatSession).filter(
                and_(
                    ChatSession.soft_deleted_at < hard_delete_cutoff,
                    ChatSession.soft_deleted_at.isnot(None)
                )
            ).all()
            
            for session in sessions_to_hard_delete:
                # 删除会话消息
                db.query(ChatMessage).filter(
                    ChatMessage.session_id == session.session_id
                ).delete()
                
                # 删除会话
                db.delete(session)
                result["hard_deleted"] += 1
            
            db.commit()
            
            if result["soft_deleted"] > 0:
                logger.info(f"Soft deleted {result['soft_deleted']} expired sessions")
            if result["hard_deleted"] > 0:
                logger.info(f"Hard deleted {result['hard_deleted']} sessions")
                
            return result
                
        except Exception as e:
            if db:
                db.rollback()
            logger.error(f"Failed to cleanup expired sessions: {e}")
            return result
        finally:
            if db_gen:
                try:
                    next(db_gen)
                except StopIteration:
                    pass