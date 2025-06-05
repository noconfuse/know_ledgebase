#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据库连接和会话管理
"""

import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.ext.declarative import declarative_base
from contextlib import contextmanager
from typing import Generator

from config import settings
from models.chat_models import Base as ChatBase
from models.user_models import Base as UserBase
from models.index_models import Base as IndexBase
from sqlalchemy.ext.declarative import declarative_base

# 合并所有模型的Base
Base = declarative_base()

# 导入所有模型以确保它们被注册
from models.chat_models import ChatSession, ChatMessage
from models.user_models import User, UserSession
from models.index_models import IndexInfo

# 将所有表添加到统一的Base中
for table in ChatBase.metadata.tables.values():
    table.tometadata(Base.metadata)
    
for table in UserBase.metadata.tables.values():
    table.tometadata(Base.metadata)
    
for table in IndexBase.metadata.tables.values():
    table.tometadata(Base.metadata)

logger = logging.getLogger(__name__)

# 创建数据库连接URL
DATABASE_URL = f"postgresql://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DATABASE}"

# 创建数据库引擎
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,  # 连接池预检查
    pool_recycle=3600,   # 连接回收时间
    echo=False           # 是否打印SQL语句
)

# 创建会话工厂
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 创建线程安全的会话
db_session = scoped_session(SessionLocal)

def get_db() -> Generator:
    """获取数据库会话的依赖注入函数"""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Database error: {e}")
        raise
    finally:
        session.close()

def init_db() -> None:
    """初始化数据库表结构"""
    try:
        # 创建所有表
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise