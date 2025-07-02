import os
import json
import hashlib
import logging
import re
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class MetadataCacheManager:
    """元数据缓存管理器 - 基于文件名目录结构的持久化缓存
    
    新的缓存结构:
    cache/metadata/
    ├── document_name_1/
    │   ├── document_metadata.json  # 文档级元数据
    │   └── chunks/                 # chunk级元数据目录
    │       ├── chunk_0.json
    │       ├── chunk_1.json
    │       └── ...
    ├── document_name_2/
    │   ├── document_metadata.json
    │   └── chunks/
    └── ...
    """

    def __init__(self, cache_dir: str = "cache/metadata"):
        """初始化缓存管理器

        Args:
            cache_dir: 缓存目录路径
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"MetadataCacheManager initialized with cache_dir: {self.cache_dir}"
        )

    def _sanitize_filename(self, filename: str) -> str:
        """清理文件名，移除或替换不安全的字符
        
        Args:
            filename: 原始文件名
            
        Returns:
            清理后的安全文件名
        """
        # 移除路径分隔符和其他不安全字符
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
        # 移除连续的下划线
        sanitized = re.sub(r'_+', '_', sanitized)
        # 移除开头和结尾的下划线和点
        sanitized = sanitized.strip('_.')
        # 限制长度
        if len(sanitized) > 100:
            sanitized = sanitized[:100]
        # 如果清理后为空，使用默认名称
        if not sanitized:
            sanitized = "unknown_document"
        return sanitized
    
    def _get_file_hash(self, content: str) -> str:
        """计算文件的哈希值，用于检测文件变化

        Args:
            content: 文件内容

        Returns:
            文件内容的MD5哈希值
        """
        return hashlib.md5(content.encode("utf-8")).hexdigest()
    
    def _get_document_dir(self, doc_id: str) -> Path:
        """获取文档的缓存目录路径
        
        Args:
            doc_id: 文档ID
            
        Returns:
            文档缓存目录路径
        """
        # 从doc_id中提取文件名
        filename = os.path.basename(doc_id)
        # 清理文件名
        safe_filename = self._sanitize_filename(filename)
        # 为了避免重名，添加doc_id的hash前缀
        doc_hash = hashlib.md5(doc_id.encode("utf-8")).hexdigest()[:8]
        dir_name = f"{doc_hash}_{safe_filename}"
        return self.cache_dir / dir_name
    
    def _get_document_metadata_file(self, doc_id: str) -> Path:
        """获取文档级元数据文件路径
        
        Args:
            doc_id: 文档ID
            
        Returns:
            文档级元数据文件路径
        """
        doc_dir = self._get_document_dir(doc_id)
        return doc_dir / "document_metadata.json"
    
    def _get_chunks_dir(self, doc_id: str) -> Path:
        """获取文档的chunks目录路径
        
        Args:
            doc_id: 文档ID
            
        Returns:
            chunks目录路径
        """
        doc_dir = self._get_document_dir(doc_id)
        return doc_dir / "chunks"
    
    def _get_chunk_file_path(self, doc_id: str, chunk_id: str) -> Path:
        """获取chunk缓存文件路径
        
        Args:
            doc_id: 文档ID
            chunk_id: chunk的唯一ID
            
        Returns:
            chunk缓存文件路径
        """
        chunks_dir = self._get_chunks_dir(doc_id)
        return chunks_dir / f"{chunk_id}.json"
    
    def _generate_chunk_id(self, chunk_text: str, chunk_index: int = None) -> str:
        """生成chunk的唯一ID，基于内容哈希
        
        Args:
            chunk_text: chunk文本内容
            chunk_index: chunk索引（可选，用于调试）
            
        Returns:
            chunk的唯一ID
        """
        # 使用chunk文本内容生成哈希作为唯一ID
        content_hash = hashlib.sha256(chunk_text.encode('utf-8')).hexdigest()[:16]
        
        # 如果提供了索引，可以在ID中包含索引信息用于调试
        if chunk_index is not None:
            return f"chunk_{chunk_index}_{content_hash}"
        else:
            return f"chunk_{content_hash}"

    def get_cached_metadata(
        self, doc_id: str, content: str = None
    ) -> Optional[Dict[str, Any]]:
        """获取缓存的文档级元数据

        Args:
            doc_id: 文档ID
            content: 文件内容（可选，如果提供则用于验证文件是否变化）

        Returns:
            缓存的元数据，如果缓存无效或不存在则返回None
        """
        metadata_file = self._get_document_metadata_file(doc_id)

        if not metadata_file.exists():
            logger.debug(f"No document metadata cache found for doc_id: {doc_id}")
            return None

        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                cache_data = json.load(f)

            # 验证缓存版本
            if cache_data.get("version") != "2.0":
                logger.warning(
                    f"Cache version mismatch for {doc_id}, invalidating cache"
                )
                return None

            # 如果提供了内容，验证文件是否发生变化
            if content:
                current_hash = self._get_file_hash(content)
                cached_hash = cache_data.get("file_hash", "")

                if current_hash != cached_hash:
                    logger.info(f"File content changed for {doc_id}, cache invalidated")
                    return None

            logger.info(f"Document metadata cache hit for doc_id: {doc_id}")
            return cache_data.get("metadata")

        except Exception as e:
            logger.error(f"Error reading document metadata cache for {doc_id}: {e}")
            return None

    def save_metadata_to_cache(
        self, doc_id: str, metadata: Dict[str, Any], content: str = None
    ) -> bool:
        """保存文档级元数据到缓存

        Args:
            doc_id: 文档ID
            metadata: 要缓存的元数据
            content: 文件内容（可选，如果提供则计算文件哈希）

        Returns:
            是否保存成功
        """
        try:
            # 确保文档目录存在
            doc_dir = self._get_document_dir(doc_id)
            doc_dir.mkdir(parents=True, exist_ok=True)
            
            metadata_file = self._get_document_metadata_file(doc_id)

            cache_data = {
                "version": "2.0",
                "timestamp": datetime.now().isoformat(),
                "doc_id": doc_id,
                "metadata": metadata,
            }

            # 如果提供了内容，计算并保存文件哈希
            if content:
                cache_data["file_hash"] = self._get_file_hash(content)

            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)

            logger.info(f"Document metadata cached for doc_id: {doc_id}")
            return True

        except Exception as e:
            logger.error(f"Error saving document metadata to cache for {doc_id}: {e}")
            return False

    def clear_cache(self, doc_id: str = None):
        """清除缓存

        Args:
            doc_id: 如果指定，只清除特定文档的缓存；否则清除所有缓存
        """
        try:
            if doc_id:
                doc_dir = self._get_document_dir(doc_id)
                if doc_dir.exists():
                    import shutil
                    shutil.rmtree(doc_dir)
                    logger.info(f"Cache cleared for doc_id: {doc_id}")
            else:
                if self.cache_dir.exists():
                    import shutil
                    shutil.rmtree(self.cache_dir)
                    self.cache_dir.mkdir(parents=True, exist_ok=True)
                    logger.info("All cache cleared")

        except Exception as e:
            logger.error(f"Error clearing cache: {e}")

    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息

        Returns:
            包含缓存统计信息的字典
        """
        try:
            if not self.cache_dir.exists():
                return {
                    "total_documents": 0,
                    "total_chunks": 0,
                    "total_size_mb": 0.0,
                    "cache_dir": str(self.cache_dir),
                }

            total_documents = 0
            total_chunks = 0
            total_size = 0
            
            # 遍历所有文档目录
            for doc_dir in self.cache_dir.iterdir():
                if doc_dir.is_dir():
                    total_documents += 1
                    
                    # 统计文档级元数据文件
                    doc_metadata_file = doc_dir / "document_metadata.json"
                    if doc_metadata_file.exists():
                        total_size += doc_metadata_file.stat().st_size
                    
                    # 统计chunk级元数据文件
                    chunks_dir = doc_dir / "chunks"
                    if chunks_dir.exists():
                        chunk_files = list(chunks_dir.glob("*.json"))
                        total_chunks += len(chunk_files)
                        total_size += sum(f.stat().st_size for f in chunk_files)

            return {
                "total_documents": total_documents,
                "total_chunks": total_chunks,
                "total_size_mb": total_size / (1024 * 1024),
                "cache_dir": str(self.cache_dir),
            }

        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {
                "total_documents": 0,
                "total_chunks": 0,
                "total_size_mb": 0.0,
                "cache_dir": str(self.cache_dir),
                "error": str(e),
            }

    def get_chunk_metadata_from_cache(self, doc_id: str, chunk_text: str, chunk_index: int = None) -> Optional[Dict[str, Any]]:
        """从缓存中获取chunk元数据
        
        Args:
            doc_id: 文档ID
            chunk_text: chunk文本内容
            chunk_index: chunk索引（可选）
            
        Returns:
            chunk元数据，如果不存在则返回None
        """
        try:
            # 生成基于内容的chunk ID
            chunk_id = self._generate_chunk_id(chunk_text, chunk_index)
            chunk_file_path = self._get_chunk_file_path(doc_id, chunk_id)
            
            if not chunk_file_path.exists():
                return None
                
            with open(chunk_file_path, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
                
            # 检查缓存版本
            if cached_data.get('version') != "2.0":
                logger.warning(f"Cache version mismatch for chunk {chunk_id} in doc {doc_id}")
                return None
                
            # 验证chunk内容是否匹配（额外的安全检查）
            cached_chunk_text = cached_data.get('chunk_text', '')
            if cached_chunk_text != chunk_text:
                logger.warning(f"Chunk content mismatch for {chunk_id} in doc {doc_id}, cache may be stale")
                return None
                
            return cached_data.get('metadata')
            
        except Exception as e:
            logger.error(f"Error reading chunk metadata from cache: {e}")
            return None

    def save_chunk_metadata_to_cache(
        self, doc_id: str, chunk_text: str, metadata: Dict[str, Any], chunk_index: int = None
    ) -> bool:
        """保存chunk级别的元数据到缓存

        Args:
            doc_id: 文档ID
            chunk_text: chunk文本内容
            metadata: chunk的元数据
            chunk_index: chunk索引（可选）

        Returns:
            是否保存成功
        """
        try:
            # 生成基于内容的chunk ID
            chunk_id = self._generate_chunk_id(chunk_text, chunk_index)
            
            # 确保chunk目录存在
            chunks_dir = self._get_chunks_dir(doc_id)
            chunks_dir.mkdir(parents=True, exist_ok=True)

            chunk_file = self._get_chunk_file_path(doc_id, chunk_id)

            cache_data = {
                "version": "2.0",
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "chunk_text": chunk_text,  # 保存chunk文本用于验证
                "metadata": metadata,
                "timestamp": datetime.now().isoformat(),
            }

            with open(chunk_file, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)

            logger.debug(f"Chunk metadata cached for {doc_id}:{chunk_id}")
            return True

        except Exception as e:
            logger.error(f"Error saving chunk metadata to cache: {e}")
            return False

    def clear_chunk_cache(self, doc_id: str = None):
        """清除chunk缓存
        
        Args:
            doc_id: 如果指定，只清除特定文档的chunk缓存；否则清除所有chunk缓存
        """
        try:
            if doc_id:
                chunks_dir = self._get_chunks_dir(doc_id)
                if chunks_dir.exists():
                    import shutil
                    shutil.rmtree(chunks_dir)
                    logger.info(f"Chunk cache cleared for doc_id: {doc_id}")
            else:
                # 清除所有文档的chunk缓存
                for doc_dir in self.cache_dir.iterdir():
                    if doc_dir.is_dir():
                        chunks_dir = doc_dir / "chunks"
                        if chunks_dir.exists():
                            import shutil
                            shutil.rmtree(chunks_dir)
                logger.info("All chunk cache cleared")
        except Exception as e:
            logger.error(f"Error clearing chunk cache: {e}")

    def get_chunk_cache_stats(self) -> Dict[str, Any]:
        """获取chunk缓存统计信息

        Returns:
            包含chunk缓存统计信息的字典
        """
        try:
            if not self.cache_dir.exists():
                return {
                    "total_chunks": 0,
                    "total_documents_with_chunks": 0,
                    "total_size_mb": 0.0,
                    "cache_dir": str(self.cache_dir),
                }

            total_chunks = 0
            total_documents_with_chunks = 0
            total_size = 0
            
            # 遍历所有文档目录
            for doc_dir in self.cache_dir.iterdir():
                if doc_dir.is_dir():
                    chunks_dir = doc_dir / "chunks"
                    if chunks_dir.exists():
                        chunk_files = list(chunks_dir.glob("*.json"))
                        if chunk_files:
                            total_documents_with_chunks += 1
                            total_chunks += len(chunk_files)
                            total_size += sum(f.stat().st_size for f in chunk_files)

            return {
                "total_chunks": total_chunks,
                "total_documents_with_chunks": total_documents_with_chunks,
                "total_size_mb": total_size / (1024 * 1024),
                "cache_dir": str(self.cache_dir),
            }

        except Exception as e:
            logger.error(f"Error getting chunk cache stats: {e}")
            return {
                "total_chunks": 0,
                "total_documents_with_chunks": 0,
                "total_size_mb": 0.0,
                "cache_dir": str(self.cache_dir),
                "error": str(e),
            }
