# -*- coding: utf-8 -*-
"""
用户数据访问对象
"""

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_
import jwt
from models.user_models import User, UserSession
from config import settings

logger = logging.getLogger(__name__)

class UserDAO:
    """用户数据访问对象"""
    
    @staticmethod
    def create_user(db: Session, user_data) -> Optional[User]:
        """创建新用户"""
        try:
            # 检查用户名是否已存在
            existing_username = db.query(User).filter(User.username == user_data.username).first()
            if existing_username:
                logger.warning(f"User with username '{user_data.username}' already exists")
                return None
            
            # 只有当提供了邮箱时才检查邮箱是否已存在
            if user_data.email:
                existing_email = db.query(User).filter(User.email == user_data.email).first()
                if existing_email:
                    logger.warning(f"User with email '{user_data.email}' already exists")
                    return None
            
            # 创建新用户
            user = User(
                username=user_data.username,
                email=user_data.email,
                hashed_password=User.hash_password(user_data.password),
                full_name=user_data.full_name,
                is_superuser=getattr(user_data, 'is_superuser', False)
            )
            db.add(user)
            db.flush()  # 刷新以获取ID，但不提交
            db.commit()  # 新增：提交事务，确保写入数据库
            
            logger.info(f"Created new user: {user_data.username}")
            return user
            
        except Exception as e:
            logger.error(f"Error creating user: {e}", exc_info=True)
            db.rollback()
            return None
    
    @staticmethod
    def get_user_by_username(db: Session, username: str) -> Optional[User]:
        """根据用户名获取用户"""
        try:
            user = db.query(User).filter(User.username == username).first()
            return user
        except Exception as e:
            logger.error(f"Error getting user by username: {e}", exc_info=True)
            return None
    
    @staticmethod
    def get_user_by_id(db: Session, user_id: str) -> Optional[User]:
        """根据用户ID获取用户"""
        try:
            user = db.query(User).filter(User.id == user_id).first()
            return user
        except Exception as e:
            logger.error(f"Error getting user by ID: {e}", exc_info=True)
            return None
    
    @staticmethod
    def get_user_by_email(db: Session, email: str) -> Optional[User]:
        """根据邮箱获取用户"""
        try:
            user = db.query(User).filter(User.email == email).first()
            return user
        except Exception as e:
            logger.error(f"Error getting user by email: {e}", exc_info=True)
            return None
    
    @staticmethod
    def authenticate_user(db: Session, username: str, password: str) -> Optional[User]:
        """验证用户登录"""
        try:
            user = UserDAO.get_user_by_username(db, username)
            if user and user.verify_password(password) and user.is_active:
                # 更新最后登录时间
                user.last_login = datetime.utcnow()
                return user
            return None
        except Exception as e:
            logger.error(f"Error authenticating user: {e}")
            return None
    
    @staticmethod
    def verify_token(token: str) -> Optional[Dict[str, Any]]:
        """验证JWT令牌"""
        try:
            payload = jwt.decode(
                token, 
                settings.JWT_SECRET_KEY, 
                algorithms=[settings.JWT_ALGORITHM]
            )
            user_id = payload.get("user_id")
            if user_id is None:
                return None
            # 检查用户是否存在且活跃
            # 需要获取数据库会话
            from models.database import get_db
            db = next(get_db())
            try:
                user = UserDAO.get_user_by_id(db, user_id)
                if user is None or not user.is_active:
                    return None
            finally:
                db.close()
            return {
                "user_id": user_id,
                "username": payload.get("username"),
                "exp": payload.get("exp")
            }
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            return None
        except jwt.PyJWTError as e:
            logger.warning(f"JWT decode error: {e}")
            return None
        except Exception as e:
            logger.error(f"Error verifying token: {e}")
            return None
    
    @staticmethod
    def update_last_login(db: Session, user_id: str) -> bool:
        """更新用户最后登录时间"""
        try:
            user = db.query(User).filter(User.id == user_id).first()
            if not user:
                return False
            
            user.last_login = datetime.utcnow()
            logger.info(f"Updated last login for user: {user.username}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating last login: {e}")
            db.rollback()
            return False
    
    @staticmethod
    def update_user(db: Session, user_id: str, user_update) -> Optional[User]:
        """更新用户信息"""
        try:
            user = db.query(User).filter(User.id == user_id).first()
            if not user:
                return None
            
            # 更新允许的字段
            update_data = user_update.dict(exclude_unset=True)
            for field, value in update_data.items():
                if field == 'password':
                    user.hashed_password = User.hash_password(value)
                elif hasattr(user, field):
                    setattr(user, field, value)
            
            logger.info(f"Updated user: {user.username}")
            return user
            
        except Exception as e:
            logger.error(f"Error updating user: {e}")
            db.rollback()
            return None
    
    @staticmethod
    def delete_user(db: Session, user_id: str) -> bool:
        """删除用户（软删除）"""
        try:
            user = db.query(User).filter(User.id == user_id).first()
            if not user:
                return False
            
            user.is_active = False
            logger.info(f"Deleted user: {user.username}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting user: {e}")
            db.rollback()
            return False
    
    @staticmethod
    def list_users(db: Session, skip: int = 0, limit: int = 100) -> List[User]:
        """获取用户列表"""
        try:
            users = db.query(User).offset(skip).limit(limit).all()
            return users
        except Exception as e:
            logger.error(f"Error listing users: {e}")
            return []
    
    @staticmethod
    def create_user_session(db: Session, user_id: str, refresh_token: str) -> bool:
        """创建用户会话记录"""
        try:
            # 解析token获取过期时间
            import jwt
            payload = jwt.decode(refresh_token, options={"verify_signature": False})
            expires_at = datetime.fromtimestamp(payload.get('exp', 0))
            
            session = UserSession(
                user_id=user_id,
                token_jti=refresh_token,
                expires_at=expires_at
            )
            db.add(session)
            return True
        except Exception as e:
            logger.error(f"Error creating user session: {e}")
            db.rollback()
            return False
    
    @staticmethod
    def revoke_session_token(db: Session, token_jti: str) -> bool:
        """撤销会话令牌"""
        try:
            session = db.query(UserSession).filter(
                UserSession.token_jti == token_jti
            ).first()
            
            if session:
                session.is_revoked = True
                return True
            return False
        except Exception as e:
            logger.error(f"Error revoking session token: {e}")
            db.rollback()
            return False
    
    @staticmethod
    def revoke_token(db: Session, token_jti: str) -> bool:
        """撤销令牌"""
        try:
            session = db.query(UserSession).filter(
                UserSession.token_jti == token_jti
            ).first()
            
            if session:
                session.is_revoked = True
                return True
            return False
        except Exception as e:
            logger.error(f"Error revoking token: {e}")
            return False
    
    @staticmethod
    def is_token_revoked(db: Session, token_jti: str) -> bool:
        """检查令牌是否被撤销"""
        try:
            session = db.query(UserSession).filter(
                and_(
                    UserSession.token_jti == token_jti,
                    UserSession.is_revoked == True
                )
            ).first()
            return session is not None
        except Exception as e:
            logger.error(f"Error checking token revocation: {e}")
            return False