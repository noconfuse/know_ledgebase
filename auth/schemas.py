# -*- coding: utf-8 -*-
"""
认证相关的Pydantic模型
"""

from typing import Optional
from pydantic import BaseModel, Field, EmailStr, validator
from datetime import datetime
import uuid

class UserCreate(BaseModel):
    """创建用户请求"""
    username: str = Field(..., min_length=3, max_length=50, description="用户名")
    password: str = Field(..., min_length=6, max_length=100, description="密码")
    email: Optional[EmailStr] = Field(None, description="邮箱（可选）")
    full_name: Optional[str] = Field(None, max_length=100, description="全名（可选）")

class UserLogin(BaseModel):
    """用户登录请求"""
    username: str = Field(..., description="用户名")
    password: str = Field(..., description="密码")

class UserUpdate(BaseModel):
    """更新用户请求"""
    email: Optional[EmailStr] = Field(None, description="邮箱")
    full_name: Optional[str] = Field(None, max_length=100, description="全名")
    password: Optional[str] = Field(None, min_length=6, max_length=100, description="新密码")

class UserResponse(BaseModel):
    """用户响应"""
    id: str
    username: str
    email: Optional[str]
    full_name: Optional[str]
    is_active: bool
    is_superuser: bool
    created_at: Optional[str]
    last_login: Optional[str]
    
    @validator('id', pre=True)
    def convert_uuid_to_str(cls, v):
        if isinstance(v, uuid.UUID):
            return str(v)
        return v
    
    @validator('created_at', 'last_login', pre=True)
    def convert_datetime_to_str(cls, v):
        if isinstance(v, datetime):
            return v.isoformat()
        return v
    
    class Config:
        from_attributes = True

class Token(BaseModel):
    """令牌响应"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int  # 过期时间（秒）
    user: UserResponse

class TokenData(BaseModel):
    """令牌数据"""
    user_id: Optional[str] = None
    username: Optional[str] = None