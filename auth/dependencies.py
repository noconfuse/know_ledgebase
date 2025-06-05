# -*- coding: utf-8 -*-
"""
认证依赖项和中间件
"""

import logging
from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from models.database import get_db
from dao.user_dao import UserDAO
from models.user_models import User

logger = logging.getLogger(__name__)

# HTTP Bearer 认证方案
security = HTTPBearer()

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db = Depends(get_db)
) -> User:
    """获取当前认证用户"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        # 验证令牌
        token_data = UserDAO.verify_token(credentials.credentials)
        if token_data is None:
            raise credentials_exception
        
        # 检查令牌是否被撤销
        if UserDAO.is_token_revoked(db, token_data.get("jti", "")):
            raise credentials_exception
        
        # 获取用户信息
        user = UserDAO.get_user_by_id(db, token_data["user_id"])
        if user is None:
            raise credentials_exception
            
        return user
        
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        raise credentials_exception

async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """获取当前活跃用户"""
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Inactive user"
        )
    return current_user.to_dict()

async def get_current_superuser(current_user: User = Depends(get_current_user)) -> User:
    """获取当前超级用户"""
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    return current_user.to_dict()

async def get_optional_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False)),
    db = Depends(get_db)
) -> Optional[User]:
    """获取可选的当前用户（用于可选认证的接口）"""
    if credentials is None:
        return None
    
    try:
        # 验证令牌
        token_data = UserDAO.verify_token(credentials.credentials)
        if token_data is None:
            return None
        
        # 检查令牌是否被撤销
        if UserDAO.is_token_revoked(db, token_data.get("jti", "")):
            return None
        
        # 获取用户信息
        user = UserDAO.get_user_by_id(db, token_data["user_id"])
        return user if user and user.is_active else None
        
    except Exception as e:
        logger.error(f"Optional authentication error: {e}")
        return None