# -*- coding: utf-8 -*-
"""
元数据提取失败记录管理器
用于记录和管理元数据提取失败的情况
"""

import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)

class MetadataFailureManager:
    """元数据提取失败记录管理器"""
    
    def __init__(self, failure_log_dir: str = "cache/metadata_failures"):
        """
        初始化失败记录管理器
        
        Args:
            failure_log_dir: 失败记录存储目录
        """
        self.failure_log_dir = Path(failure_log_dir)
        self.failure_log_dir.mkdir(parents=True, exist_ok=True)
        
        # 文档级失败记录文件
        self.doc_failure_file = self.failure_log_dir / "document_failures.json"
        # chunk级失败记录文件
        self.chunk_failure_file = self.failure_log_dir / "chunk_failures.json"
        
        # 初始化失败记录文件
        self._init_failure_files()
    
    def _init_failure_files(self):
        """初始化失败记录文件"""
        for file_path in [self.doc_failure_file, self.chunk_failure_file]:
            if not file_path.exists():
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump({}, f, ensure_ascii=False, indent=2)
    
    def record_document_failure(self, doc_id: str, text: str, error_reason: str):
        """
        记录文档级元数据提取失败
        
        Args:
            doc_id: 文档ID
            text: 文档文本内容
            error_reason: 失败原因
        """
        try:
            # 读取现有失败记录
            with open(self.doc_failure_file, 'r', encoding='utf-8') as f:
                failures = json.load(f)
            
            # 添加新的失败记录
            failures[doc_id] = {
                "doc_id": doc_id,
                "text": text[:1000] + "..." if len(text) > 1000 else text,  # 限制文本长度
                "full_text_length": len(text),
                "error_reason": error_reason,
                "failure_time": datetime.now().isoformat(),
                "retry_count": failures.get(doc_id, {}).get("retry_count", 0) + 1
            }
            
            # 保存失败记录
            with open(self.doc_failure_file, 'w', encoding='utf-8') as f:
                json.dump(failures, f, ensure_ascii=False, indent=2)
            
            logger.warning(f"📝 Recorded document failure for {doc_id}: {error_reason}")
            
        except Exception as e:
            logger.error(f"❌ Failed to record document failure: {e}")
    
    def record_chunk_failure(self, chunk_id: str, text: str, error_reason: str, doc_id: str = None):
        """
        记录chunk级元数据提取失败
        
        Args:
            chunk_id: chunk ID
            text: chunk文本内容
            error_reason: 失败原因
            doc_id: 所属文档ID（可选）
        """
        try:
            # 读取现有失败记录
            with open(self.chunk_failure_file, 'r', encoding='utf-8') as f:
                failures = json.load(f)
            
            # 添加新的失败记录
            failures[chunk_id] = {
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "text": text[:500] + "..." if len(text) > 500 else text,  # 限制文本长度
                "full_text_length": len(text),
                "error_reason": error_reason,
                "failure_time": datetime.now().isoformat(),
                "retry_count": failures.get(chunk_id, {}).get("retry_count", 0) + 1
            }
            
            # 保存失败记录
            with open(self.chunk_failure_file, 'w', encoding='utf-8') as f:
                json.dump(failures, f, ensure_ascii=False, indent=2)
            
            logger.warning(f"📝 Recorded chunk failure for {chunk_id}: {error_reason}")
            
        except Exception as e:
            logger.error(f"❌ Failed to record chunk failure: {e}")
    
    def remove_document_failure(self, doc_id: str):
        """
        删除文档级失败记录（成功提取后调用）
        
        Args:
            doc_id: 文档ID
        """
        try:
            with open(self.doc_failure_file, 'r', encoding='utf-8') as f:
                failures = json.load(f)
            
            if doc_id in failures:
                del failures[doc_id]
                
                with open(self.doc_failure_file, 'w', encoding='utf-8') as f:
                    json.dump(failures, f, ensure_ascii=False, indent=2)
                
                logger.info(f"✅ Removed document failure record for {doc_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to remove document failure record: {e}")
    
    def remove_chunk_failure(self, chunk_id: str):
        """
        删除chunk级失败记录（成功提取后调用）
        
        Args:
            chunk_id: chunk ID
        """
        try:
            with open(self.chunk_failure_file, 'r', encoding='utf-8') as f:
                failures = json.load(f)
            
            if chunk_id in failures:
                del failures[chunk_id]
                
                with open(self.chunk_failure_file, 'w', encoding='utf-8') as f:
                    json.dump(failures, f, ensure_ascii=False, indent=2)
                
                logger.info(f"✅ Removed chunk failure record for {chunk_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to remove chunk failure record: {e}")
    
    def get_document_failures(self) -> Dict[str, Any]:
        """
        获取所有文档级失败记录
        
        Returns:
            Dict[str, Any]: 失败记录字典
        """
        try:
            with open(self.doc_failure_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"❌ Failed to read document failures: {e}")
            return {}
    
    def get_chunk_failures(self) -> Dict[str, Any]:
        """
        获取所有chunk级失败记录
        
        Returns:
            Dict[str, Any]: 失败记录字典
        """
        try:
            with open(self.chunk_failure_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"❌ Failed to read chunk failures: {e}")
            return {}
    
    def get_failure_summary(self) -> Dict[str, Any]:
        """
        获取失败记录摘要统计
        
        Returns:
            Dict[str, Any]: 失败记录统计信息
        """
        doc_failures = self.get_document_failures()
        chunk_failures = self.get_chunk_failures()
        
        return {
            "document_failures_count": len(doc_failures),
            "chunk_failures_count": len(chunk_failures),
            "total_failures": len(doc_failures) + len(chunk_failures),
            "document_failure_ids": list(doc_failures.keys()),
            "chunk_failure_ids": list(chunk_failures.keys())
        }
    
    def cleanup_old_failures(self, days_old: int = 30):
        """
        清理旧的失败记录
        
        Args:
            days_old: 清理多少天前的记录
        """
        from datetime import timedelta
        
        cutoff_date = datetime.now() - timedelta(days=days_old)
        
        # 清理文档级失败记录
        try:
            doc_failures = self.get_document_failures()
            cleaned_doc_failures = {}
            
            for doc_id, failure_info in doc_failures.items():
                failure_time = datetime.fromisoformat(failure_info.get("failure_time", ""))
                if failure_time > cutoff_date:
                    cleaned_doc_failures[doc_id] = failure_info
            
            with open(self.doc_failure_file, 'w', encoding='utf-8') as f:
                json.dump(cleaned_doc_failures, f, ensure_ascii=False, indent=2)
            
            logger.info(f"🧹 Cleaned {len(doc_failures) - len(cleaned_doc_failures)} old document failure records")
            
        except Exception as e:
            logger.error(f"❌ Failed to cleanup old document failures: {e}")
        
        # 清理chunk级失败记录
        try:
            chunk_failures = self.get_chunk_failures()
            cleaned_chunk_failures = {}
            
            for chunk_id, failure_info in chunk_failures.items():
                failure_time = datetime.fromisoformat(failure_info.get("failure_time", ""))
                if failure_time > cutoff_date:
                    cleaned_chunk_failures[chunk_id] = failure_info
            
            with open(self.chunk_failure_file, 'w', encoding='utf-8') as f:
                json.dump(cleaned_chunk_failures, f, ensure_ascii=False, indent=2)
            
            logger.info(f"🧹 Cleaned {len(chunk_failures) - len(cleaned_chunk_failures)} old chunk failure records")
            
        except Exception as e:
            logger.error(f"❌ Failed to cleanup old chunk failures: {e}")