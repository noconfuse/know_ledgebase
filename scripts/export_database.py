#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PostgreSQL数据库导出脚本

此脚本用于导出PostgreSQL数据库的数据和结构，包括：
1. 数据库结构（表、索引、约束等）
2. 数据内容
3. 可选择性导出特定表或全库

使用方法：
    python export_database.py [选项]

选项：
    --host: 数据库主机地址（默认：localhost）
    --port: 数据库端口（默认：5432）
    --database: 数据库名称（默认：knowledge_base）
    --user: 数据库用户名（默认：postgres）
    --password: 数据库密码（默认：postgres）
    --output-dir: 导出文件保存目录（默认：./database_export）
    --tables: 指定要导出的表名，多个表用逗号分隔（可选，不指定则导出全库）
    --data-only: 仅导出数据，不导出结构
    --schema-only: 仅导出结构，不导出数据
    --compress: 压缩导出文件
"""

import os
import sys
import argparse
import subprocess
import datetime
from pathlib import Path
from typing import Optional, List

# 尝试从配置文件加载默认值
try:
    from config import Settings
    settings = Settings()
    DEFAULT_HOST = settings.POSTGRES_HOST
    DEFAULT_PORT = settings.POSTGRES_PORT
    DEFAULT_DATABASE = settings.POSTGRES_DATABASE
    DEFAULT_USER = settings.POSTGRES_USER
    DEFAULT_PASSWORD = settings.POSTGRES_PASSWORD
except ImportError:
    # 如果无法导入配置，使用默认值
    DEFAULT_HOST = "localhost"
    DEFAULT_PORT = 5432
    DEFAULT_DATABASE = "knowledge_base"
    DEFAULT_USER = "postgres"
    DEFAULT_PASSWORD = "postgres"

class DatabaseExporter:
    def __init__(self, host: str, port: int, database: str, user: str, password: str, output_dir: str):
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.output_dir = Path(output_dir)
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置环境变量以避免密码提示
        os.environ['PGPASSWORD'] = password
    
    def check_pg_dump_available(self) -> bool:
        """检查pg_dump是否可用"""
        try:
            result = subprocess.run(['pg_dump', '--version'], 
                                  capture_output=True, text=True, check=True)
            print(f"使用 pg_dump 版本: {result.stdout.strip()}")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("错误: 未找到 pg_dump 工具，请确保已安装 PostgreSQL 客户端工具")
            return False
    
    def test_connection(self) -> bool:
        """测试数据库连接"""
        try:
            cmd = [
                'psql',
                '-h', self.host,
                '-p', str(self.port),
                '-U', self.user,
                '-d', self.database,
                '-c', 'SELECT 1;'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print("数据库连接测试成功")
            return True
        except subprocess.CalledProcessError as e:
            print(f"数据库连接失败: {e.stderr}")
            return False
    
    def export_full_database(self, compress: bool = False) -> str:
        """导出完整数据库（结构+数据）"""
        filename = f"{self.database}_full_{self.timestamp}.sql"
        if compress:
            filename += ".gz"
        
        filepath = self.output_dir / filename
        
        cmd = [
            'pg_dump',
            '-h', self.host,
            '-p', str(self.port),
            '-U', self.user,
            '-d', self.database,
            '--verbose',
            '--no-password'
        ]
        
        if compress:
            cmd.extend(['-Z', '9'])  # 最高压缩级别
        
        cmd.extend(['-f', str(filepath)])
        
        print(f"开始导出完整数据库到: {filepath}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"完整数据库导出成功: {filepath}")
            return str(filepath)
        except subprocess.CalledProcessError as e:
            print(f"导出失败: {e.stderr}")
            raise
    
    def export_schema_only(self) -> str:
        """仅导出数据库结构"""
        filename = f"{self.database}_schema_{self.timestamp}.sql"
        filepath = self.output_dir / filename
        
        cmd = [
            'pg_dump',
            '-h', self.host,
            '-p', str(self.port),
            '-U', self.user,
            '-d', self.database,
            '--schema-only',
            '--verbose',
            '--no-password',
            '-f', str(filepath)
        ]
        
        print(f"开始导出数据库结构到: {filepath}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"数据库结构导出成功: {filepath}")
            return str(filepath)
        except subprocess.CalledProcessError as e:
            print(f"导出失败: {e.stderr}")
            raise
    
    def export_data_only(self, tables: Optional[List[str]] = None) -> str:
        """仅导出数据"""
        if tables:
            filename = f"{self.database}_data_{'_'.join(tables)}_{self.timestamp}.sql"
        else:
            filename = f"{self.database}_data_{self.timestamp}.sql"
        
        filepath = self.output_dir / filename
        
        cmd = [
            'pg_dump',
            '-h', self.host,
            '-p', str(self.port),
            '-U', self.user,
            '-d', self.database,
            '--data-only',
            '--verbose',
            '--no-password'
        ]
        
        if tables:
            for table in tables:
                cmd.extend(['-t', table])
        
        cmd.extend(['-f', str(filepath)])
        
        print(f"开始导出数据到: {filepath}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"数据导出成功: {filepath}")
            return str(filepath)
        except subprocess.CalledProcessError as e:
            print(f"导出失败: {e.stderr}")
            raise
    
    def export_specific_tables(self, tables: List[str], compress: bool = False) -> str:
        """导出指定表（结构+数据）"""
        filename = f"{self.database}_tables_{'_'.join(tables)}_{self.timestamp}.sql"
        if compress:
            filename += ".gz"
        
        filepath = self.output_dir / filename
        
        cmd = [
            'pg_dump',
            '-h', self.host,
            '-p', str(self.port),
            '-U', self.user,
            '-d', self.database,
            '--verbose',
            '--no-password'
        ]
        
        for table in tables:
            cmd.extend(['-t', table])
        
        if compress:
            cmd.extend(['-Z', '9'])
        
        cmd.extend(['-f', str(filepath)])
        
        print(f"开始导出指定表到: {filepath}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"指定表导出成功: {filepath}")
            return str(filepath)
        except subprocess.CalledProcessError as e:
            print(f"导出失败: {e.stderr}")
            raise
    
    def get_database_info(self):
        """获取数据库信息"""
        try:
            cmd = [
                'psql',
                '-h', self.host,
                '-p', str(self.port),
                '-U', self.user,
                '-d', self.database,
                '-c', "SELECT schemaname, tablename FROM pg_tables WHERE schemaname = 'public';"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print("数据库表信息:")
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"获取数据库信息失败: {e.stderr}")

def main():
    parser = argparse.ArgumentParser(description='PostgreSQL数据库导出工具')
    parser.add_argument('--host', default=DEFAULT_HOST, help=f'数据库主机地址 (默认: {DEFAULT_HOST})')
    parser.add_argument('--port', type=int, default=DEFAULT_PORT, help=f'数据库端口 (默认: {DEFAULT_PORT})')
    parser.add_argument('--database', default=DEFAULT_DATABASE, help=f'数据库名称 (默认: {DEFAULT_DATABASE})')
    parser.add_argument('--user', default=DEFAULT_USER, help=f'数据库用户名 (默认: {DEFAULT_USER})')
    parser.add_argument('--password', default=DEFAULT_PASSWORD, help='数据库密码')
    parser.add_argument('--output-dir', default='./database_export', help='导出文件保存目录 (默认: ./database_export)')
    parser.add_argument('--tables', help='指定要导出的表名，多个表用逗号分隔')
    parser.add_argument('--data-only', action='store_true', help='仅导出数据，不导出结构')
    parser.add_argument('--schema-only', action='store_true', help='仅导出结构，不导出数据')
    parser.add_argument('--compress', action='store_true', help='压缩导出文件')
    parser.add_argument('--info', action='store_true', help='显示数据库信息')
    
    args = parser.parse_args()
    
    # 创建导出器
    exporter = DatabaseExporter(
        host=args.host,
        port=args.port,
        database=args.database,
        user=args.user,
        password=args.password,
        output_dir=args.output_dir
    )
    
    # 检查工具可用性
    if not exporter.check_pg_dump_available():
        sys.exit(1)
    
    # 测试连接
    if not exporter.test_connection():
        sys.exit(1)
    
    # 显示数据库信息
    if args.info:
        exporter.get_database_info()
        return
    
    try:
        exported_files = []
        
        if args.schema_only:
            # 仅导出结构
            file_path = exporter.export_schema_only()
            exported_files.append(file_path)
        elif args.data_only:
            # 仅导出数据
            tables = args.tables.split(',') if args.tables else None
            file_path = exporter.export_data_only(tables)
            exported_files.append(file_path)
        elif args.tables:
            # 导出指定表
            tables = args.tables.split(',')
            file_path = exporter.export_specific_tables(tables, args.compress)
            exported_files.append(file_path)
        else:
            # 导出完整数据库
            file_path = exporter.export_full_database(args.compress)
            exported_files.append(file_path)
        
        print("\n=== 导出完成 ===")
        for file_path in exported_files:
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            print(f"文件: {file_path}")
            print(f"大小: {file_size:.2f} MB")
        
        print(f"\n导出文件保存在: {args.output_dir}")
        
    except Exception as e:
        print(f"导出过程中发生错误: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()