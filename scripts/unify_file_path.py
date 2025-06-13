#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件路径统一脚本
用于修改PostgreSQL数据库中向量存储表的file_path字段
"""

import os
import sys
import psycopg2
import argparse
from pathlib import Path
from typing import Optional, List, Tuple

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
from utils.logging_config import get_logger

logger = get_logger(__name__)

class FilePathUnifier:
    """文件路径统一处理器"""
    
    def __init__(self):
        """初始化数据库连接"""
        self.connection = None
        self.cursor = None
        
    def connect_database(self) -> bool:
        """连接到PostgreSQL数据库"""
        try:
            self.connection = psycopg2.connect(
                host=settings.POSTGRES_HOST,
                port=settings.POSTGRES_PORT,
                database=settings.POSTGRES_DATABASE,
                user=settings.POSTGRES_USER,
                password=settings.POSTGRES_PASSWORD
            )
            self.cursor = self.connection.cursor()
            logger.info(f"成功连接到数据库: {settings.POSTGRES_DATABASE}")
            return True
        except Exception as e:
            logger.error(f"数据库连接失败: {e}")
            return False
    
    def close_connection(self):
        """关闭数据库连接"""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
        logger.info("数据库连接已关闭")
    
    def get_table_info(self, table_name: str) -> Optional[List[str]]:
        """获取表的列信息"""
        try:
            self.cursor.execute("""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = %s
                ORDER BY ordinal_position
            """, (table_name,))
            
            columns = self.cursor.fetchall()
            if not columns:
                logger.warning(f"表 {table_name} 不存在或无列信息")
                return None
            
            logger.info(f"表 {table_name} 的列信息:")
            for col_name, col_type in columns:
                logger.info(f"  - {col_name}: {col_type}")
            
            return [col[0] for col in columns]
        except Exception as e:
            logger.error(f"获取表信息失败: {e}")
            return None
    
    def get_file_path_records(self, table_name: str) -> Optional[List[Tuple]]:
        """获取所有包含file_path的记录"""
        try:
            # 检查表是否有file_path字段
            columns = self.get_table_info(table_name)
            if not columns or 'metadata_' not in columns:
                logger.error(f"表 {table_name} 中没有找到 file_path 字段")
                return None
            
            # 查询所有记录的file_path
            self.cursor.execute(f"""
                SELECT id, metadata_->>'file_path' as file_path 
                FROM {table_name} 
                WHERE metadata_->>'file_path' IS NOT NULL
                ORDER BY id
            """)
            
            records = self.cursor.fetchall()
            logger.info(f"从表 {table_name} 中找到 {len(records)} 条包含file_path的记录")
            
            return records
        except Exception as e:
            logger.error(f"查询file_path记录失败: {e}")
            return None
    
    def normalize_file_path(self, file_path: str, base_path: Optional[str] = None) -> str:
        """标准化文件路径"""
        if not file_path:
            return file_path
        
        # 转换为Path对象进行处理
        path = Path(file_path)
        
        # 如果是相对路径且提供了基础路径，则转换为绝对路径
        if not path.is_absolute() and base_path:
            path = Path(base_path) / path
        
        # 标准化路径（解析.和..，统一分隔符）
        normalized = path.resolve()
        
        return str(normalized)
    
    def update_file_paths(self, table_name: str, dry_run: bool = True, 
                         base_path: Optional[str] = None) -> bool:
        """更新表中的file_path字段"""
        try:
            # 获取所有记录
            records = self.get_file_path_records(table_name)
            if not records:
                logger.warning("没有找到需要更新的记录")
                return True
            
            updated_count = 0
            
            for record_id, current_path in records:
                # 标准化路径
                normalized_path = self.normalize_file_path(current_path, base_path)
                
                # 如果路径发生了变化
                if normalized_path != current_path:
                    logger.info(f"记录 {record_id}:")
                    logger.info(f"  原路径: {current_path}")
                    logger.info(f"  新路径: {normalized_path}")
                    
                    if not dry_run:
                        # 执行更新
                        self.cursor.execute(f"""
                            UPDATE {table_name} 
                            SET metadata_ = jsonb_set(metadata_::jsonb, ARRAY['file_path'], to_jsonb(%s::text))
                            WHERE id = %s
                        """, (normalized_path, record_id))
                        updated_count += 1
                    else:
                        logger.info("  [DRY RUN] 将会更新此记录")
                        updated_count += 1
            
            if not dry_run and updated_count > 0:
                self.connection.commit()
                logger.info(f"成功更新了 {updated_count} 条记录")
            elif dry_run:
                logger.info(f"[DRY RUN] 将会更新 {updated_count} 条记录")
            else:
                logger.info("没有记录需要更新")
            
            return True
            
        except Exception as e:
            logger.error(f"更新file_path失败: {e}")
            if self.connection:
                self.connection.rollback()
            return False
    
    def batch_update_specific_paths(self, table_name: str, path_mappings: dict, 
                                  dry_run: bool = True) -> bool:
        """批量更新特定的路径映射"""
        try:
            updated_count = 0
            
            for old_path, new_path in path_mappings.items():
                # 查找匹配的记录
                self.cursor.execute(f"""
                    SELECT id, metadata_->>'file_path' as file_path 
                    FROM {table_name} 
                    WHERE metadata_->>'file_path' LIKE %s
                """, (f"%{old_path}%",))
                
                matching_records = self.cursor.fetchall()
                
                for record_id, current_path in matching_records:
                    # 替换路径
                    updated_path = current_path.replace(old_path, new_path)
                    
                    logger.info(f"记录 {record_id}:")
                    logger.info(f"  原路径: {current_path}")
                    logger.info(f"  新路径: {updated_path}")
                    
                    if not dry_run:
                        self.cursor.execute(f"""
                            UPDATE {table_name} 
                            SET metadata_ = jsonb_set(metadata_::jsonb, ARRAY['file_path'], to_jsonb(%s::text))
                            WHERE id = %s
                        """, (updated_path, record_id))
                        updated_count += 1
                    else:
                        logger.info("  [DRY RUN] 将会更新此记录")
                        updated_count += 1
            
            if not dry_run and updated_count > 0:
                self.connection.commit()
                logger.info(f"成功更新了 {updated_count} 条记录")
            elif dry_run:
                logger.info(f"[DRY RUN] 将会更新 {updated_count} 条记录")
            else:
                logger.info("没有记录需要更新")
            
            return True
            
        except Exception as e:
            logger.error(f"批量更新路径失败: {e}")
            if self.connection:
                self.connection.rollback()
            return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="统一数据库中的文件路径")
    parser.add_argument("--table", 
                       default="data_vector_store_28367261_f5fa_4db0_9baa_9f440789c329",
                       help="要处理的表名")
    parser.add_argument("--dry-run", action="store_true", 
                       help="只显示将要进行的更改，不实际执行")
    parser.add_argument("--base-path", 
                       help="相对路径的基础路径")
    parser.add_argument("--mode", choices=["normalize", "replace"], 
                       default="normalize",
                       help="处理模式: normalize(标准化路径) 或 replace(替换特定路径)")
    parser.add_argument("--old-path", 
                       help="要替换的旧路径（仅在replace模式下使用）")
    parser.add_argument("--new-path", 
                       help="替换后的新路径（仅在replace模式下使用）")
    
    args = parser.parse_args()
    
    # 创建处理器实例
    unifier = FilePathUnifier()
    
    try:
        # 连接数据库
        if not unifier.connect_database():
            logger.error("无法连接到数据库，程序退出")
            return 1
        
        # 显示表信息
        logger.info(f"正在处理表: {args.table}")
        columns = unifier.get_table_info(args.table)
        if not columns:
            logger.error(f"表 {args.table} 不存在")
            return 1
        
        # 根据模式执行不同的操作
        if args.mode == "normalize":
            # 标准化路径模式
            success = unifier.update_file_paths(
                table_name=args.table,
                dry_run=args.dry_run,
                base_path=args.base_path
            )
        elif args.mode == "replace":
            # 路径替换模式
            if not args.old_path or not args.new_path:
                logger.error("replace模式需要指定 --old-path 和 --new-path 参数")
                return 1
            
            path_mappings = {args.old_path: args.new_path}
            success = unifier.batch_update_specific_paths(
                table_name=args.table,
                path_mappings=path_mappings,
                dry_run=args.dry_run
            )
        
        if success:
            logger.info("文件路径统一处理完成")
            return 0
        else:
            logger.error("文件路径统一处理失败")
            return 1
            
    except KeyboardInterrupt:
        logger.info("用户中断操作")
        return 1
    except Exception as e:
        logger.error(f"程序执行出错: {e}")
        return 1
    finally:
        # 关闭数据库连接
        unifier.close_connection()

if __name__ == "__main__":
    exit(main())





