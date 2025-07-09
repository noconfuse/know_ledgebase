#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PostgreSQL数据库导入脚本

此脚本用于将导出的PostgreSQL数据库文件导入到新的数据库中，包括：
1. 创建数据库（如果不存在）
2. 导入数据库结构
3. 导入数据内容
4. 验证导入结果

使用方法：
    python import_database.py [选项]

选项：
    --host: 数据库主机地址（默认：localhost）
    --port: 数据库端口（默认：5432）
    --database: 目标数据库名称（默认：knowledge_base）
    --user: 数据库用户名（默认：postgres）
    --password: 数据库密码（默认：postgres）
    --sql-file: 要导入的SQL文件路径（必需）
    --create-db: 如果数据库不存在则创建
    --drop-existing: 删除现有数据库后重新创建
    --dry-run: 仅验证文件和连接，不执行实际导入
    --verbose: 显示详细输出
"""

import os
import sys
import argparse
import subprocess
import datetime
import gzip
from pathlib import Path
from typing import Optional

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

class DatabaseImporter:
    def __init__(self, host: str, port: int, database: str, user: str, password: str, verbose: bool = False):
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.verbose = verbose
        
        # 设置环境变量以避免密码提示
        os.environ['PGPASSWORD'] = password
    
    def check_tools_available(self) -> bool:
        """检查必要的PostgreSQL工具是否可用"""
        tools = ['psql', 'createdb', 'dropdb']
        missing_tools = []
        
        for tool in tools:
            try:
                result = subprocess.run([tool, '--version'], 
                                      capture_output=True, text=True, check=True)
                if self.verbose:
                    print(f"找到 {tool}: {result.stdout.strip()}")
            except (subprocess.CalledProcessError, FileNotFoundError):
                missing_tools.append(tool)
        
        if missing_tools:
            print(f"错误: 未找到以下工具: {', '.join(missing_tools)}")
            print("请确保已安装 PostgreSQL 客户端工具")
            return False
        
        return True
    
    def test_connection(self, database: Optional[str] = None) -> bool:
        """测试数据库连接"""
        test_db = database or 'postgres'  # 使用postgres数据库进行连接测试
        
        try:
            cmd = [
                'psql',
                '-h', self.host,
                '-p', str(self.port),
                '-U', self.user,
                '-d', test_db,
                '-c', 'SELECT 1;'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            if self.verbose:
                print(f"数据库连接测试成功 (连接到: {test_db})")
            return True
        except subprocess.CalledProcessError as e:
            print(f"数据库连接失败: {e.stderr}")
            return False
    
    def database_exists(self) -> bool:
        """检查数据库是否存在"""
        try:
            cmd = [
                'psql',
                '-h', self.host,
                '-p', str(self.port),
                '-U', self.user,
                '-d', 'postgres',
                '-t',  # 仅输出元组
                '-c', f"SELECT 1 FROM pg_database WHERE datname = '{self.database}';"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            exists = result.stdout.strip() == '1'
            
            if self.verbose:
                print(f"数据库 '{self.database}' {'存在' if exists else '不存在'}")
            
            return exists
        except subprocess.CalledProcessError as e:
            print(f"检查数据库是否存在时出错: {e.stderr}")
            return False
    
    def create_database(self) -> bool:
        """创建数据库"""
        try:
            cmd = [
                'createdb',
                '-h', self.host,
                '-p', str(self.port),
                '-U', self.user,
                self.database
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"数据库 '{self.database}' 创建成功")
            return True
        except subprocess.CalledProcessError as e:
            print(f"创建数据库失败: {e.stderr}")
            return False
    
    def drop_database(self) -> bool:
        """删除数据库"""
        try:
            cmd = [
                'dropdb',
                '-h', self.host,
                '-p', str(self.port),
                '-U', self.user,
                self.database
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"数据库 '{self.database}' 删除成功")
            return True
        except subprocess.CalledProcessError as e:
            print(f"删除数据库失败: {e.stderr}")
            return False
    
    def validate_sql_file(self, sql_file: str) -> bool:
        """验证SQL文件"""
        file_path = Path(sql_file)
        
        if not file_path.exists():
            print(f"错误: SQL文件不存在: {sql_file}")
            return False
        
        if file_path.stat().st_size == 0:
            print(f"错误: SQL文件为空: {sql_file}")
            return False
        
        # 检查文件是否为压缩文件
        is_compressed = sql_file.endswith('.gz')
        
        if self.verbose:
            file_size = file_path.stat().st_size / (1024 * 1024)  # MB
            print(f"SQL文件: {sql_file}")
            print(f"文件大小: {file_size:.2f} MB")
            print(f"压缩文件: {'是' if is_compressed else '否'}")
        
        return True
    
    def import_sql_file(self, sql_file: str) -> bool:
        """导入SQL文件"""
        file_path = Path(sql_file)
        is_compressed = sql_file.endswith('.gz')
        
        print(f"开始导入SQL文件: {sql_file}")
        
        try:
            if is_compressed:
                # 处理压缩文件
                with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                    cmd = [
                        'psql',
                        '-h', self.host,
                        '-p', str(self.port),
                        '-U', self.user,
                        '-d', self.database
                    ]
                    
                    if self.verbose:
                        cmd.append('-v')
                        cmd.append('ON_ERROR_STOP=1')
                    
                    result = subprocess.run(cmd, input=f.read(), text=True, 
                                          capture_output=True, check=True)
            else:
                # 处理普通文件
                cmd = [
                    'psql',
                    '-h', self.host,
                    '-p', str(self.port),
                    '-U', self.user,
                    '-d', self.database,
                    '-f', str(file_path)
                ]
                
                if self.verbose:
                    cmd.append('-v')
                    cmd.append('ON_ERROR_STOP=1')
                
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            print("SQL文件导入成功")
            
            if self.verbose and result.stdout:
                print("导入输出:")
                print(result.stdout)
            
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"导入SQL文件失败: {e.stderr}")
            if self.verbose and e.stdout:
                print("标准输出:")
                print(e.stdout)
            return False
        except Exception as e:
            print(f"导入过程中发生错误: {e}")
            return False
    
    def verify_import(self) -> bool:
        """验证导入结果"""
        try:
            # 检查表数量
            cmd = [
                'psql',
                '-h', self.host,
                '-p', str(self.port),
                '-U', self.user,
                '-d', self.database,
                '-t',
                '-c', "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public';"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            table_count = int(result.stdout.strip())
            
            print(f"导入验证: 找到 {table_count} 个表")
            
            if table_count > 0:
                # 显示表列表
                cmd = [
                    'psql',
                    '-h', self.host,
                    '-p', str(self.port),
                    '-U', self.user,
                    '-d', self.database,
                    '-c', "SELECT schemaname, tablename FROM pg_tables WHERE schemaname = 'public' ORDER BY tablename;"
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                print("导入的表:")
                print(result.stdout)
                
                return True
            else:
                print("警告: 没有找到任何表")
                return False
                
        except subprocess.CalledProcessError as e:
            print(f"验证导入结果时出错: {e.stderr}")
            return False
    
    def get_database_size(self) -> Optional[str]:
        """获取数据库大小"""
        try:
            cmd = [
                'psql',
                '-h', self.host,
                '-p', str(self.port),
                '-U', self.user,
                '-d', self.database,
                '-t',
                '-c', f"SELECT pg_size_pretty(pg_database_size('{self.database}'));"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return None

def main():
    parser = argparse.ArgumentParser(description='PostgreSQL数据库导入工具')
    parser.add_argument('--host', default=DEFAULT_HOST, help=f'数据库主机地址 (默认: {DEFAULT_HOST})')
    parser.add_argument('--port', type=int, default=DEFAULT_PORT, help=f'数据库端口 (默认: {DEFAULT_PORT})')
    parser.add_argument('--database', default=DEFAULT_DATABASE, help=f'目标数据库名称 (默认: {DEFAULT_DATABASE})')
    parser.add_argument('--user', default=DEFAULT_USER, help=f'数据库用户名 (默认: {DEFAULT_USER})')
    parser.add_argument('--password', default=DEFAULT_PASSWORD, help='数据库密码')
    parser.add_argument('--sql-file', required=True, help='要导入的SQL文件路径')
    parser.add_argument('--create-db', action='store_true', help='如果数据库不存在则创建')
    parser.add_argument('--drop-existing', action='store_true', help='删除现有数据库后重新创建')
    parser.add_argument('--dry-run', action='store_true', help='仅验证文件和连接，不执行实际导入')
    parser.add_argument('--verbose', action='store_true', help='显示详细输出')
    
    args = parser.parse_args()
    
    # 创建导入器
    importer = DatabaseImporter(
        host=args.host,
        port=args.port,
        database=args.database,
        user=args.user,
        password=args.password,
        verbose=args.verbose
    )
    
    # 检查工具可用性
    if not importer.check_tools_available():
        sys.exit(1)
    
    # 验证SQL文件
    if not importer.validate_sql_file(args.sql_file):
        sys.exit(1)
    
    # 测试连接（连接到postgres数据库）
    if not importer.test_connection():
        sys.exit(1)
    
    if args.dry_run:
        print("干运行模式: 验证完成，未执行实际导入")
        return
    
    try:
        # 处理数据库创建/删除
        db_exists = importer.database_exists()
        
        if args.drop_existing and db_exists:
            print(f"删除现有数据库: {args.database}")
            if not importer.drop_database():
                sys.exit(1)
            db_exists = False
        
        if not db_exists:
            if args.create_db:
                print(f"创建数据库: {args.database}")
                if not importer.create_database():
                    sys.exit(1)
            else:
                print(f"错误: 数据库 '{args.database}' 不存在，请使用 --create-db 选项创建")
                sys.exit(1)
        
        # 测试目标数据库连接
        if not importer.test_connection(args.database):
            sys.exit(1)
        
        # 导入SQL文件
        start_time = datetime.datetime.now()
        
        if not importer.import_sql_file(args.sql_file):
            sys.exit(1)
        
        end_time = datetime.datetime.now()
        duration = end_time - start_time
        
        print(f"\n=== 导入完成 ===")
        print(f"导入耗时: {duration.total_seconds():.2f} 秒")
        
        # 验证导入结果
        if importer.verify_import():
            db_size = importer.get_database_size()
            if db_size:
                print(f"数据库大小: {db_size}")
            print("\n导入验证成功！")
        else:
            print("\n导入验证失败，请检查导入结果")
            sys.exit(1)
        
    except KeyboardInterrupt:
        print("\n导入被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"导入过程中发生错误: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()