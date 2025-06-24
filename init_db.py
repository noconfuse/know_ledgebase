#!/usr/bin/env python3
"""
数据库初始化脚本
创建所有表并添加默认管理员用户
"""

import sys
import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
from passlib.context import CryptContext
import psycopg2

# 添加项目根目录到sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.database import engine, Base, get_db, DATABASE_URL
from dao.user_dao import UserDAO
from auth.schemas import UserCreate
from config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def create_database_if_not_exists():
    """创建数据库（如果不存在）"""
    db_name = settings.POSTGRES_DATABASE
    # 连接到默认的 postgres 数据库来创建新的数据库
    default_db_url = f"postgresql://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/postgres"
    try:
        # 直接用 psycopg2 连接，避免 SQLAlchemy 代理问题
        conn = psycopg2.connect(
            dbname='postgres',
            user=settings.POSTGRES_USER,
            password=settings.POSTGRES_PASSWORD,
            host=settings.POSTGRES_HOST,
            port=settings.POSTGRES_PORT
        )
        conn.autocommit = True
        cursor = conn.cursor()
        cursor.execute(f"SELECT 1 FROM pg_database WHERE datname = '{db_name}'")
        if not cursor.fetchone():
            print(f"Database '{db_name}' does not exist. Creating...")
            cursor.execute(f"CREATE DATABASE {db_name}")
            print(f"Database '{db_name}' created successfully.")
        else:
            print(f"Database '{db_name}' already exists.")
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"Error connecting to default database or creating '{db_name}': {e}")
        sys.exit(1)

def init_database():
    """初始化数据库"""
    create_database_if_not_exists()
    print("Creating database tables...")
    
    # 创建所有表
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully.")
    
    # 创建默认管理员用户和免费用户
    from sqlalchemy.orm import sessionmaker
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()
    try:
        # 检查是否已存在管理员用户
        existing_admin = UserDAO.get_user_by_username(db, "admin")
        if not existing_admin:
            from auth.schemas import UserCreate
            admin_data = UserCreate(
                username="admin",
                password="admin123",
                email="admin@example.com",
                full_name="Administrator"
            )
            admin_user = UserDAO.create_user(db, admin_data)
            if admin_user:
                print(f"Admin user created successfully:")
                print(f"  Username: {admin_user.username}")
                print(f"  Email: {admin_user.email}")
                print(f"  Password: admin123")
                print(f"  Please change the default password after first login!")
            else:
                print("Failed to create admin user.")
        else:
            print("管理员用户已存在")
        # 检查是否已存在免费用户
        existing_free = UserDAO.get_user_by_username(db, "free_user")
        if not existing_free:
            from auth.schemas import UserCreate
            import uuid
            import random, string
            free_data = UserCreate(
                username="free_user",
                password=''.join(random.choices(string.ascii_letters + string.digits, k=16)),
                email=None,
                full_name="Free User"
            )
            free_user = UserDAO.create_user(db, free_data)
            print(free_data)
            if free_user:
                print("Free user created successfully.")
            else:
                print("Failed to create free user.")
        else:
            print("免费用户已存在")
    except Exception as e:
        print(f"Error creating admin or free user: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    init_database()