#!/usr/bin/env python3
"""
数据库初始化脚本
创建所有表并添加默认管理员用户
"""

import sys
import os
from sqlalchemy.orm import Session
from passlib.context import CryptContext

# 添加项目根目录到sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.database import engine, Base, get_db
from dao.user_dao import UserDAO
from auth.schemas import UserCreate

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def init_database():
    """初始化数据库"""
    print("Creating database tables...")
    
    # 创建所有表
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully.")
    
    # 创建默认管理员用户
    from sqlalchemy.orm import sessionmaker
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()
    
    try:
        # 检查是否已存在管理员用户
        existing_admin = UserDAO.get_user_by_username(db, "admin")
        if existing_admin:
            print("管理员用户已存在")
            return
        
        # 创建默认管理员用户
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
            
    except Exception as e:
        print(f"Error creating admin user: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    init_database()