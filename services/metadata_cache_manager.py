import os
import json
import hashlib
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class MetadataCacheManager:
    """元数据缓存管理器 - 提供文件级别和chunk级别的持久化缓存"""

    def __init__(self, cache_dir: str = "cache/metadata"):
        """初始化缓存管理器

        Args:
            cache_dir: 缓存目录路径
        """
        self.cache_dir = Path(cache_dir)
        self.chunk_cache_dir = self.cache_dir / "chunks"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"MetadataCacheManager initialized with cache_dir: {self.cache_dir}"
        )

    def _get_file_hash(self, file_path: str, content: str = None) -> str:
        """计算文件的哈希值，用于检测文件变化

        Args:
            file_path: 文件路径
            content: 文件内容（可选，如果提供则直接使用）

        Returns:
            文件内容的MD5哈希值
        """
        if content is not None:
            # 使用提供的内容计算哈希
            return hashlib.md5(content.encode("utf-8")).hexdigest()

        # 从文件读取内容计算哈希
        try:
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                return hashlib.md5(content.encode("utf-8")).hexdigest()
        except Exception as e:
            logger.warning(f"Failed to calculate hash for {file_path}: {e}")

        return ""

    def _get_chunk_cache_key(
        self, chunk_text: str, extract_config: Dict[str, Any]
    ) -> str:
        """生成chunk缓存的唯一key

        Args:
            chunk_text: chunk的文本内容
            extract_config: 提取配置参数

        Returns:
            chunk缓存的唯一key
        """
        # 将chunk文本和配置参数组合生成唯一key
        config_str = json.dumps(extract_config, sort_keys=True)
        combined_content = f"{chunk_text}||{config_str}"
        return hashlib.md5(combined_content.encode("utf-8")).hexdigest()

    def _get_chunk_cache_file_path(self, cache_key: str) -> Path:
        """获取chunk缓存文件路径

        Args:
            cache_key: chunk缓存key

        Returns:
            chunk缓存文件路径
        """
        return self.chunk_cache_dir / f"{cache_key}.json"

    def _get_cache_file_path(self, doc_id: str) -> Path:
        """获取缓存文件路径

        Args:
            doc_id: 文档ID

        Returns:
            缓存文件的完整路径
        """
        # 使用doc_id的哈希值作为文件名，避免特殊字符问题
        cache_filename = hashlib.md5(doc_id.encode("utf-8")).hexdigest() + ".json"
        return self.cache_dir / cache_filename

    def get_cached_metadata(
        self, doc_id: str, file_path: str = None, content: str = None
    ) -> Optional[Dict[str, Any]]:
        """获取缓存的元数据

        Args:
            doc_id: 文档ID
            file_path: 原始文件路径（用于验证文件是否变化）
            content: 文件内容（可选，如果提供则用于验证）

        Returns:
            缓存的元数据，如果缓存无效或不存在则返回None
        """
        cache_file = self._get_cache_file_path(doc_id)

        if not cache_file.exists():
            logger.debug(f"No cache found for doc_id: {doc_id}")
            return None

        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cache_data = json.load(f)

            # 验证缓存版本
            if cache_data.get("version") != "1.0":
                logger.warning(
                    f"Cache version mismatch for {doc_id}, invalidating cache"
                )
                return None

            # 如果提供了文件路径或内容，验证文件是否发生变化
            if file_path or content:
                current_hash = self._get_file_hash(file_path, content)
                cached_hash = cache_data.get("file_hash", "")

                if current_hash != cached_hash:
                    logger.info(f"File content changed for {doc_id}, cache invalidated")
                    return None

            logger.info(f"Cache hit for doc_id: {doc_id}")
            metadata = cache_data.get("metadata")
            
            # 修复可能存在的嵌套metadata问题
            if isinstance(metadata, dict) and "metadata" in metadata:
                logger.warning(f"Detected nested metadata structure in cache for {doc_id}, fixing...")
                # 如果metadata中包含嵌套的metadata，返回正确的结构
                fixed_metadata = metadata.get("metadata", {})
                chunk_template = metadata.get("chunk_template", "")
                return {
                    "metadata": fixed_metadata,
                    "chunk_template": chunk_template
                }
            
            return metadata

        except Exception as e:
            logger.error(f"Error reading cache for {doc_id}: {e}")
            return None

    def save_metadata_to_cache(
        self,
        doc_id: str,
        metadata: Dict[str, Any],
        file_path: str = None,
        content: str = None,
    ):
        """保存元数据到缓存

        Args:
            doc_id: 文档ID
            metadata: 要缓存的元数据
            file_path: 原始文件路径
            content: 文件内容（可选）
        """
        cache_file = self._get_cache_file_path(doc_id)

        try:
            # 计算文件哈希
            file_hash = (
                self._get_file_hash(file_path, content)
                if (file_path or content)
                else ""
            )

            cache_data = {
                "version": "1.0",
                "doc_id": doc_id,
                "file_path": file_path,
                "file_hash": file_hash,
                "cached_at": datetime.utcnow().isoformat(),
                "metadata": metadata,
            }

            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)

            logger.info(f"Metadata cached for doc_id: {doc_id}")

        except Exception as e:
            logger.error(f"Error saving cache for {doc_id}: {e}")

    def clear_cache(self, doc_id: str = None):
        """清除缓存

        Args:
            doc_id: 要清除的文档ID，如果为None则清除所有缓存
        """
        if doc_id:
            cache_file = self._get_cache_file_path(doc_id)
            if cache_file.exists():
                cache_file.unlink()
                logger.info(f"Cache cleared for doc_id: {doc_id}")
        else:
            # 清除所有缓存文件
            for cache_file in self.cache_dir.glob("*.json"):
                cache_file.unlink()
            logger.info("All metadata cache cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息

        Returns:
            缓存统计信息
        """
        cache_files = list(self.cache_dir.glob("*.json"))
        total_size = sum(f.stat().st_size for f in cache_files)

        return {
            "cache_dir": str(self.cache_dir),
            "total_files": len(cache_files),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
        }

    def get_chunk_metadata_from_cache(
        self, chunk_text: str, extract_config: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """从缓存获取chunk元数据

        Args:
            chunk_text: chunk的文本内容
            extract_config: 提取配置参数

        Returns:
            缓存的chunk元数据，如果不存在则返回None
        """
        cache_key = self._get_chunk_cache_key(chunk_text, extract_config)
        cache_file = self._get_chunk_cache_file_path(cache_key)

        if not cache_file.exists():
            return None

        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cache_data = json.load(f)

            # 验证缓存数据的完整性
            if not all([
                "chunk_metadata" in cache_data,
                "cached_at" in cache_data,
            ]):
                return None

            # 验证元数据是否是无用数据
            if cache_data["chunk_metadata"].get("summary") == "元数据提取失败 - API响应异常":
                return None
            
            chunk_metadata = cache_data["chunk_metadata"]
            
            # 修复可能存在的嵌套metadata问题
            if isinstance(chunk_metadata, dict) and "metadata" in chunk_metadata:
                logger.warning(f"Detected nested metadata structure in chunk cache for key: {cache_key[:8]}..., fixing...")
                # 如果chunk_metadata中包含嵌套的metadata，提取正确的内容
                if isinstance(chunk_metadata["metadata"], dict):
                    # 将嵌套的metadata内容合并到顶层
                    fixed_metadata = {**chunk_metadata}
                    nested_metadata = fixed_metadata.pop("metadata", {})
                    fixed_metadata.update(nested_metadata)
                    chunk_metadata = fixed_metadata
                
            logger.debug(f"Chunk cache hit for key: {cache_key[:8]}...")
            return chunk_metadata

        except Exception as e:
            logger.warning(f"Error reading chunk cache {cache_key}: {e}")

        return None

    def save_chunk_metadata_to_cache(
        self,
        chunk_text: str,
        extract_config: Dict[str, Any],
        chunk_metadata: Dict[str, Any],
    ):
        """保存chunk元数据到缓存

        Args:
            chunk_text: chunk的文本内容
            extract_config: 提取配置参数
            chunk_metadata: chunk元数据
        """
        cache_key = self._get_chunk_cache_key(chunk_text, extract_config)
        cache_file = self._get_chunk_cache_file_path(cache_key)

        try:
            cache_data = {
                "version": "1.0",
                "cache_key": cache_key,
                "chunk_text_hash": hashlib.md5(chunk_text.encode("utf-8")).hexdigest(),
                "extract_config": extract_config,
                "cached_at": datetime.utcnow().isoformat(),
                "chunk_metadata": chunk_metadata,
            }

            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)

            logger.debug(f"Chunk metadata cached with key: {cache_key[:8]}...")

        except Exception as e:
            logger.error(f"Error saving chunk cache {cache_key}: {e}")

    def clear_chunk_cache(self):
        """清除所有chunk缓存"""
        try:
            for cache_file in self.chunk_cache_dir.glob("*.json"):
                cache_file.unlink()
            logger.info("All chunk cache cleared")
        except Exception as e:
            logger.error(f"Error clearing chunk cache: {e}")

    def get_chunk_cache_stats(self) -> Dict[str, Any]:
        """获取chunk缓存统计信息

        Returns:
            chunk缓存统计信息
        """
        cache_files = list(self.chunk_cache_dir.glob("*.json"))
        total_size = sum(f.stat().st_size for f in cache_files)

        return {
            "chunk_cache_files": len(cache_files),
            "chunk_cache_size_mb": total_size / (1024 * 1024),
        }
