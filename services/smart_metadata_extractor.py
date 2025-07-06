# -*- coding: utf-8 -*-
"""
智能元数据提取器
根据用户反馈重新设计的元数据提取系统：
1. 区分文档级元数据（一次性提取）和chunk级元数据（每个chunk提取）
2. 合并原有的统一提取器和增强提取器
3. 根据chunk大小智能决定提取哪些元数据
4. 文档级元数据会自动继承到所有chunk中
"""

import logging
import asyncio
import re
from typing import List, Dict, Sequence, Any, Optional
from tenacity import retry, stop_after_attempt, wait_fixed
from llama_index.core.llms.llm import LLM
from pydantic import BaseModel, Field, ValidationError
from llama_index.core.extractors import BaseExtractor
from llama_index.core.program import LLMTextCompletionProgram
from llama_index.core.schema import BaseNode
from llama_index.core.output_parsers import ChainableOutputParser, PydanticOutputParser

from models.metadata_models import DocumentLevelMetadata, ChunkLevelMetadata
from services.metadata_cache_manager import MetadataCacheManager
from config import settings

logger = logging.getLogger(__name__)


# 动态模板生成函数
def _create_legal_document_template() -> str:
    """为法律文档创建基于模型字段的文档级提取模板"""
    from models.metadata_models import LegalDocumentMetadata
    
    fields_desc = []
    for field_name, model_field in LegalDocumentMetadata.model_fields.items():
        fields_desc.append(f"- **{field_name}**: {model_field.description}")
    
    return f"""
请根据以下法律文档内容，提取文档级的元数据信息。

文档内容:
----------------
{{context_str}}
----------------

请提取以下法律文档级信息：

{chr(10).join(fields_desc)}

请根据给定的结构化格式返回结果。
"""

def _create_legal_chunk_template() -> str:
    """为法律文档创建基于模型字段的chunk级提取模板"""
    from models.metadata_models import ChunkLevelMetadata
    
    fields_desc = []
    for field_name, model_field in ChunkLevelMetadata.model_fields.items():
        fields_desc.append(f"- **{field_name}**: {model_field.description}")
    
    return f"""
请根据以下法律文档片段内容，提取chunk级的元数据信息。

文本内容:
----------------
{{context_str}}
----------------

文本长度: {{text_length}} 字符

请提取以下chunk级信息：

{chr(10).join(fields_desc)}

{{optional_fields}}

请根据给定的结构化格式返回结果。
"""

def _create_policy_news_document_template() -> str:
    """为政策新闻文档创建基于模型字段的文档级提取模板"""
    from models.metadata_models import PolicyDocumentMetadata
    
    fields_desc = []
    for field_name, model_field in PolicyDocumentMetadata.model_fields.items():
        fields_desc.append(f"- **{field_name}**: {model_field.description}")
    
    return f"""
请根据以下政策新闻文档内容，提取文档级的元数据信息。

文档内容:
----------------
{{context_str}}
----------------

请提取以下政策文档级信息：

{chr(10).join(fields_desc)}

请根据给定的结构化格式返回结果。
"""

def _create_policy_news_chunk_template() -> str:
    """为政策新闻文档创建基于模型字段的chunk级提取模板"""
    from models.metadata_models import ChunkLevelMetadata
    
    fields_desc = []
    for field_name, model_field in ChunkLevelMetadata.model_fields.items():
        fields_desc.append(f"- **{field_name}**: {model_field.description}")
    
    return f"""
请根据以下政策新闻文档片段内容，提取chunk级的元数据信息。

文本内容:
----------------
{{context_str}}
----------------

文本长度: {{text_length}} 字符

请提取以下chunk级信息：

{chr(10).join(fields_desc)}

{{optional_fields}}

请根据给定的结构化格式返回结果。
"""

CLASSIFICATION_PROMPT_TEMPLATE = """
Based on the content of the document below, classify it into one of the following categories: 'legal_document' or 'policy_news'.
Return only the category name.

---
{context_str}
---

Category: """

class SmartMetadataExtractor(BaseExtractor):
    """智能元数据提取器"""
    
    llm: LLM = Field(description="语言模型实例")
    min_chunk_size_for_extraction: int = Field(default=20, description="进行元数据提取的最小chunk大小，低于此长度的chunk将跳过提取")
    min_chunk_size_for_summary: int = Field(default=512, description="生成摘要的最小chunk大小")
    min_chunk_size_for_qa: int = Field(default=1024, description="生成问答对的最小chunk大小")
    max_keywords: int = Field(default=5, description="要提取的最大关键词数")
    # 移除内存缓存，改为使用基于文件名的本地缓存
    persistent_cache_manager: Optional[MetadataCacheManager] = Field(default=None, description="持久化缓存管理器")

    def __init__(self, llm: Any, min_chunk_size_for_extraction: int = None, min_chunk_size_for_summary: int = None, min_chunk_size_for_qa: int = None, max_keywords: int = None, enable_persistent_cache: bool = True, cache_dir: str = "cache/metadata", **kwargs):
        # 使用配置文件中的默认值
        if min_chunk_size_for_extraction is None:
            min_chunk_size_for_extraction = settings.MIN_CHUNK_SIZE_FOR_EXTRACTION
        if min_chunk_size_for_summary is None:
            min_chunk_size_for_summary = settings.MIN_CHUNK_SIZE_FOR_SUMMARY
        if min_chunk_size_for_qa is None:
            min_chunk_size_for_qa = settings.MIN_CHUNK_SIZE_FOR_QA
        if max_keywords is None:
            max_keywords = settings.MAX_KEYWORDS
        # 初始化持久化缓存管理器
        persistent_cache_manager = MetadataCacheManager(cache_dir) if enable_persistent_cache else None
        
        super().__init__(
            llm=llm, 
            min_chunk_size_for_extraction=min_chunk_size_for_extraction,
            min_chunk_size_for_summary=min_chunk_size_for_summary, 
            min_chunk_size_for_qa=min_chunk_size_for_qa, 
            max_keywords=max_keywords, 
            persistent_cache_manager=persistent_cache_manager,
            **kwargs
        )

    def _create_default_chunk_template(self) -> str:
        """为chunk级元数据创建默认的通用提取模板"""
        
        # 从Pydantic模型中动态生成字段描述
        fields_desc = []
        for field_name, model_field in ChunkLevelMetadata.model_fields.items():
            # 排除可选字段，它们将由 _create_chunk_program 动态添加
            if model_field.is_required():
                fields_desc.append(f"- **{field_name}**: {model_field.description}")
        
        return f"""
请根据以下文本片段内容，提取chunk级的元数据信息。

文本内容:
----------------
{{context_str}}
----------------

文本长度: {{text_length}} 字符

请提取以下核心信息：

{chr(10).join(fields_desc)}

{{optional_fields}}

请根据给定的结构化格式返回结果。
"""

    def _create_default_document_template(self) -> str:
        """为文档级元数据创建默认的通用提取模板"""
        
        # 从Pydantic模型中动态生成字段描述
        fields_desc = []
        for field_name, model_field in DocumentLevelMetadata.model_fields.items():
            fields_desc.append(f"- **{field_name}**: {model_field.description}")
        
        return f"""
请根据以下完整文档内容，提取文档级的元数据信息。这些信息将应用于整个文档的所有部分。

文档内容:
----------------
{{context_str}}
----------------

请提取以下文档级信息：

{chr(10).join(fields_desc)}

请根据给定的结构化格式返回结果。
"""
    
    def _create_fallback_metadata(self, title: str, doc_id: str) -> Dict[str, Any]:
        """创建回退元数据"""
        # 尝试从持久化缓存获取文档级元数据
        doc_metadata = {}
        if self.persistent_cache_manager:
            cached_data = self.persistent_cache_manager.get_cached_metadata(doc_id=doc_id)
            if cached_data:
                doc_metadata = cached_data.get("metadata", {})
        
        fallback_metadata = {
            "title": title,
            "summary": "",
            "keywords": [],
            "qa_pairs": [],
            "is_fallback": True
        }
        
        # 使用智能合并逻辑
        return self._merge_document_and_chunk_metadata(doc_metadata, fallback_metadata)
    
    def _optimize_document_content_for_extraction(self, text_content: str, doc_id: str) -> str:
        """优化文档内容以提高元数据提取效率
        
        对于超大文档，采用以下优化策略：
        1. 对于超过配置阈值的文档，提取关键章节
        2. 对于超大文档，使用文档摘要或关键段落
        3. 保留文档结构信息（标题、章节等）
        
        Args:
            text_content: 原始文档内容
            doc_id: 文档ID
            
        Returns:
            优化后的文档内容
        """
        from config import settings
        
        content_length = len(text_content)
        
        # 小文档直接返回
        if content_length <= settings.DOC_CONTENT_OPTIMIZATION_THRESHOLD:
            return text_content
            
        logger.info(f"📊 [CONTENT OPTIMIZER] Optimizing large document: {doc_id} ({content_length} chars)")
        
        try:
            # 策略1: 提取文档结构和关键部分
            if content_length <= settings.DOC_LARGE_CONTENT_THRESHOLD:
                return self._extract_key_sections(text_content)
            
            # 策略2: 超大文档使用智能摘要
            else:
                return self._extract_document_summary_sections(text_content)
                
        except Exception as e:
            logger.warning(f"⚠️ [CONTENT OPTIMIZER] Optimization failed for {doc_id}: {e}, using truncated content")
            # 降级策略：截取前配置长度的字符
            fallback_length = settings.DOC_MAX_OPTIMIZED_LENGTH
            return text_content[:fallback_length] + "\n\n[文档内容已截断以优化处理效率]" if content_length > fallback_length else text_content
    
    def _extract_key_sections(self, text_content: str) -> str:
        """提取文档的关键章节（适用于中等长度文档）"""
        from config import settings
        
        lines = text_content.split('\n')
        key_sections = []
        current_section = []
        
        # 定义关键章节标识符
        section_patterns = [
            r'^(第[一二三四五六七八九十\d]+章|Chapter\s+\d+)',  # 章节
            r'^(第[一二三四五六七八九十\d]+条|Article\s+\d+)',  # 条文
            r'^(第[一二三四五六七八九十\d]+节|Section\s+\d+)',  # 节
            r'^(\d+\.|\d+、|\([一二三四五六七八九十\d]+\))',  # 编号列表
            r'^(#{1,6}\s+)',  # Markdown标题
            r'^([一二三四五六七八九十]、|[A-Z]\.|\d+\.)\s*[^\s]',  # 中文序号
        ]
        
        import re
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # 检查是否是关键章节开始
            is_section_start = any(re.match(pattern, line) for pattern in section_patterns)
            
            if is_section_start:
                # 保存之前的章节（如果有内容且不超过配置的最大章节长度）
                if current_section and len('\n'.join(current_section)) <= settings.DOC_MAX_SECTION_LENGTH:
                    key_sections.extend(current_section)
                    key_sections.append('')  # 添加分隔符
                
                current_section = [line]
            else:
                current_section.append(line)
                
                # 如果当前章节过长，截断
                if len('\n'.join(current_section)) > settings.DOC_MAX_SECTION_LENGTH:
                    key_sections.extend(current_section[:20])  # 只保留前20行
                    key_sections.append('[章节内容已截断]')
                    key_sections.append('')
                    current_section = []
        
        # 处理最后一个章节
        if current_section and len('\n'.join(current_section)) <= settings.DOC_MAX_SECTION_LENGTH:
            key_sections.extend(current_section)
        
        result = '\n'.join(key_sections)
        
        # 如果提取的内容仍然过长，进一步截断
        if len(result) > settings.DOC_MAX_OPTIMIZED_LENGTH:
            result = result[:settings.DOC_MAX_OPTIMIZED_LENGTH] + "\n\n[内容已优化截断]" 
            
        return result
    
    def _extract_document_summary_sections(self, text_content: str) -> str:
        """提取超大文档的摘要部分（适用于超长文档）"""
        from config import settings
        
        lines = text_content.split('\n')
        
        # 提取策略：开头、关键标题、结尾
        summary_parts = []
        
        # 1. 文档开头（使用配置的章节长度）
        beginning = '\n'.join(lines[:50])  # 前50行
        if len(beginning) > settings.DOC_MAX_SECTION_LENGTH:
            beginning = beginning[:settings.DOC_MAX_SECTION_LENGTH]
        summary_parts.append("=== 文档开头 ===")
        summary_parts.append(beginning)
        summary_parts.append("")
        
        # 2. 提取主要标题和章节（中间部分）
        import re
        title_patterns = [
            r'^(第[一二三四五六七八九十\d]+章.*)',
            r'^(第[一二三四五六七八九十\d]+条.*)',
            r'^(#{1,3}\s+.*)',  # Markdown主标题
            r'^([一二三四五六七八九十]、.*)',
        ]
        
        summary_parts.append("=== 主要章节标题 ===")
        title_count = 0
        for line in lines[50:-50]:  # 跳过开头和结尾
            line = line.strip()
            if any(re.match(pattern, line) for pattern in title_patterns):
                summary_parts.append(line)
                title_count += 1
                if title_count >= 20:  # 最多20个标题
                    break
        summary_parts.append("")
        
        # 3. 文档结尾（使用配置的章节长度）
        ending = '\n'.join(lines[-20:])  # 后20行
        if len(ending) > settings.DOC_MAX_SECTION_LENGTH:
            ending = ending[-settings.DOC_MAX_SECTION_LENGTH:]
        summary_parts.append("=== 文档结尾 ===")
        summary_parts.append(ending)
        
        result = '\n'.join(summary_parts)
        
        # 确保结果不超过配置的摘要最大长度
        if len(result) > settings.DOC_SUMMARY_MAX_LENGTH:
            result = result[:settings.DOC_SUMMARY_MAX_LENGTH] + "\n\n[超大文档已智能摘要]" 
            
        return result

    def _merge_document_and_chunk_metadata(self, doc_metadata: Dict[str, Any], chunk_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """分层级存储文档级和chunk级元数据
        
        Args:
            doc_metadata: 文档级元数据
            chunk_metadata: chunk级元数据
            
        Returns:
            分层级存储的元数据字典
        """
        # 分层级存储：chunk级元数据为主体，文档级元数据单独存储
        layered_metadata = {
            # chunk级元数据作为主体
            **chunk_metadata,
            
            # 文档级元数据单独存储在document_metadata字段中
            'document_metadata': doc_metadata,
        }
        
        return layered_metadata

    
    async def _classify_and_extract(self, document_text: str, doc_id: str) -> Dict[str, Any]:
        """分类并提取文档级元数据"""
        # 检查持久化缓存
        if self.persistent_cache_manager:
            logger.debug(f"Checking persistent cache for doc_id: {doc_id}")
            cached_metadata = self.persistent_cache_manager.get_cached_metadata(
                doc_id=doc_id, content=document_text
            )
            if cached_metadata:
                logger.info(f"Persistent cache hit for doc_id: {doc_id}")
                return cached_metadata

        # 1. 分类文档（添加错误处理）
        try:
            classification_prompt = CLASSIFICATION_PROMPT_TEMPLATE.format(
                context_str=document_text[:4000]  # 使用片段进行分类
            )
            
            response = await self.llm.acomplete(classification_prompt)
            category = response.text.strip().lower()
        except Exception as e:
            logger.warning(f"Document classification failed for {doc_id}: {e}")
            logger.warning("Using default 'generic' category due to LLM API error")
            category = "generic"

        # 2. 选择模板和元数据模型
        if "legal" in category:
            from models.metadata_models import LegalDocumentMetadata
            doc_template = _create_legal_document_template()
            chunk_template = _create_legal_chunk_template()
            category_name = "legal_document"
            metadata_model = LegalDocumentMetadata
        elif "policy" in category:
            from models.metadata_models import PolicyDocumentMetadata
            doc_template = _create_policy_news_document_template()
            chunk_template = _create_policy_news_chunk_template()
            category_name = "policy_news"
            metadata_model = PolicyDocumentMetadata
        else:
            from models.metadata_models import DocumentLevelMetadata
            doc_template = self._create_default_document_template()
            chunk_template = self._create_default_chunk_template()
            category_name = "generic"
            metadata_model = DocumentLevelMetadata

        # 3. 提取文档级元数据
        try:
            document_program = LLMTextCompletionProgram.from_defaults(
                output_cls=metadata_model,
                prompt_template_str=doc_template,
                llm=self.llm,
                verbose=True,
            )
            # 添加API请求间隔控制，避免频率限制
            await asyncio.sleep(settings.llm_model_settings.API_REQUEST_INTERVAL)
            
            result = await document_program.acall(context_str=document_text)
            doc_metadata = result.dict()
            doc_metadata["document_category"] = category_name
            
            # 缓存结果，包括模板
            cache_data = {
                "metadata": doc_metadata,
                "chunk_template": chunk_template,
            }
            
            # 保存到持久化缓存
            if self.persistent_cache_manager:
                logger.info(f"Saving metadata to persistent cache for doc_id: {doc_id}")
                self.persistent_cache_manager.save_metadata_to_cache(
                    doc_id=doc_id, metadata=cache_data, content=document_text
                )
            
            logger.info(f"Document-level metadata extracted and cached (memory + persistent) for {doc_id}")
            return cache_data

        except Exception as e:
            logger.error(f"Error extracting document metadata for {doc_id}: {e}")
            
            # 记录详细的失败日志
            import traceback
            traceback_info = traceback.format_exc()
            self._log_failed_document(
                doc_id=doc_id,
                document_text=document_text,
                category=category_name,
                error=e,
                traceback_info=traceback_info
            )
            
            # 检查是否是API响应格式错误
            if "choices" in str(e) or "KeyError" in str(e):
                logger.warning(f"LLM API response format error detected: {e}")
                logger.warning("This is likely due to API timeout or malformed response")
            
            # 返回包含基本信息的默认元数据
            fallback_metadata = {
                "document_id": doc_id,
                "document_summary": "元数据提取失败 - API响应异常",
                "document_category": category_name,
                "extraction_failed": True,
                "error_message": str(e)
            }
            
            cache_data = {
                "metadata": fallback_metadata,
                "chunk_template": chunk_template,
            }
            
            logger.warning(f"Using fallback metadata for {doc_id} due to extraction failure")
            
            return cache_data
    
    def clear_cache(self, doc_id: str = None, include_chunks: bool = True, include_failed_logs: bool = False):
        """清除缓存
        
        Args:
            doc_id: 要清除的文档ID，如果为None则清除所有缓存
            include_chunks: 是否同时清除chunk级别的缓存
            include_failed_logs: 是否同时清除失败日志
        """
        if doc_id:
            # 清除指定文档的缓存
            if self.persistent_cache_manager:
                self.persistent_cache_manager.clear_cache(doc_id)
            
            # 清除指定文档的失败日志
            if include_failed_logs:
                self._clear_failed_document_logs(doc_id)
        else:
            # 清除所有缓存
            if self.persistent_cache_manager:
                self.persistent_cache_manager.clear_cache()
                if include_chunks:
                    self.persistent_cache_manager.clear_chunk_cache()
            
            # 清除所有失败日志
            if include_failed_logs:
                self._clear_all_failed_logs(include_chunks)
            
            cache_type = "document and chunk caches" if include_chunks else "document cache"
            log_type = " and failed logs" if include_failed_logs else ""
            logger.info(f"All {cache_type}{log_type} cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息
        
        Returns:
            缓存统计信息
        """
        stats = {}
        
        if self.persistent_cache_manager:
            stats['persistent_cache'] = self.persistent_cache_manager.get_cache_stats()
            stats['chunk_cache'] = self.persistent_cache_manager.get_chunk_cache_stats()
        
        # 添加失败统计信息
        stats['failed_documents'] = self.get_failed_documents_summary()
        stats['failed_chunks'] = self.get_failed_chunks_summary()
        
        return stats
    
    def _should_extract_summary(self, text_length: int) -> bool:
        """判断是否应该提取摘要"""
        return text_length >= self.min_chunk_size_for_summary
    
    def _should_extract_qa(self, text_length: int) -> bool:
        """判断是否应该提取问答对"""
        return text_length >= self.min_chunk_size_for_qa
    
    def _generate_failed_chunk_identifier(self, doc_id: str, chunk_index: int, text_content: str) -> str:
        """生成失败chunk的唯一标识符，用于匹配和删除失败记录
        
        Args:
            doc_id: 文档ID
            chunk_index: chunk索引
            text_content: chunk文本内容
            
        Returns:
            唯一标识符字符串
        """
        import hashlib
        import os
        
        # 使用文档ID、chunk索引和文本内容的hash生成唯一标识
        content_hash = hashlib.md5(text_content.encode('utf-8')).hexdigest()[:8]
        safe_doc_name = os.path.basename(doc_id).replace("/", "_").replace("\\", "_")
        return f"chunk_{chunk_index}_{safe_doc_name[:30]}_{content_hash}"
    
    def _log_failed_document(self, doc_id: str, document_text: str, category: str,
                           error: Exception, traceback_info: str) -> None:
        """记录失败的文档级元数据提取到本地日志目录
        
        Args:
            doc_id: 文档ID
            document_text: 文档文本内容
            category: 文档分类
            error: 错误信息
            traceback_info: 错误堆栈信息
        """
        try:
            import json
            import os
            from datetime import datetime
            import hashlib
            
            # 创建失败日志目录
            failed_docs_dir = "logs/failed_documents"
            os.makedirs(failed_docs_dir, exist_ok=True)
            
            # 生成时间戳和唯一标识符
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            content_hash = hashlib.md5(document_text.encode('utf-8')).hexdigest()[:8]
            safe_doc_name = os.path.basename(doc_id).replace("/", "_").replace("\\", "_")
            doc_identifier = f"doc_{safe_doc_name[:30]}_{content_hash}"
            
            # 构建失败记录
            failed_record = {
                "timestamp": datetime.now().isoformat(),
                "doc_id": doc_id,
                "doc_identifier": doc_identifier,
                "document_category": category,
                "text_length": len(document_text),
                "error_type": type(error).__name__,
                "error_message": str(error),
                "traceback": traceback_info,
                "text_content": document_text[:5000] + "..." if len(document_text) > 5000 else document_text,  # 保存更多内容用于调试
                "text_preview": document_text[:500] + "..." if len(document_text) > 500 else document_text
            }
            
            # 生成文件名
            filename = f"{timestamp}_{doc_identifier}.json"
            filepath = os.path.join(failed_docs_dir, filename)
            
            # 写入失败记录
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(failed_record, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Failed document logged to: {filepath}")
            
        except Exception as log_error:
            logger.error(f"Failed to log failed document: {log_error}")
    
    def _log_failed_chunk(self, doc_id: str, chunk_index: int, text_content: str,
                         text_length: int, extract_config: Dict[str, Any], 
                         error: Exception, traceback_info: str) -> None:
        """记录失败的chunk到本地日志目录
        
        Args:
            doc_id: 文档ID
            chunk_index: chunk索引
            text_content: chunk文本内容
            text_length: 文本长度
            extract_config: 提取配置
            error: 错误信息
            traceback_info: 错误堆栈信息
        """
        try:
            import json
            import os
            from datetime import datetime
            
            # 创建失败日志目录
            failed_chunks_dir = "logs/failed_chunks"
            os.makedirs(failed_chunks_dir, exist_ok=True)
            
            # 生成时间戳和唯一标识符
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            chunk_identifier = self._generate_failed_chunk_identifier(doc_id, chunk_index, text_content)
            
            # 构建失败记录
            failed_record = {
                "timestamp": datetime.now().isoformat(),
                "doc_id": doc_id,
                "chunk_index": chunk_index,
                "chunk_identifier": chunk_identifier,  # 添加唯一标识符
                "text_length": text_length,
                "extract_config": extract_config,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "traceback": traceback_info,
                "text_content": text_content[:2000] + "..." if len(text_content) > 2000 else text_content,  # 限制长度避免文件过大
                "text_preview": text_content[:200] + "..." if len(text_content) > 200 else text_content  # 提供预览
            }
            
            # 生成文件名（使用唯一标识符）
            filename = f"{timestamp}_{chunk_identifier}.json"
            filepath = os.path.join(failed_chunks_dir, filename)
            
            # 写入失败记录
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(failed_record, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Failed chunk logged to: {filepath}")
            
        except Exception as log_error:
            logger.error(f"Failed to log failed chunk: {log_error}")
    
    def _remove_failed_chunk_record(self, doc_id: str, chunk_index: int, text_content: str) -> None:
        """删除成功提取后的失败记录
        
        Args:
            doc_id: 文档ID
            chunk_index: chunk索引
            text_content: chunk文本内容
        """
        try:
            import os
            import json
            import glob
            
            failed_chunks_dir = "logs/failed_chunks"
            if not os.path.exists(failed_chunks_dir):
                return
            
            # 生成要查找的chunk标识符
            chunk_identifier = self._generate_failed_chunk_identifier(doc_id, chunk_index, text_content)
            
            # 查找匹配的失败记录文件
            pattern = os.path.join(failed_chunks_dir, f"*{chunk_identifier}.json")
            matching_files = glob.glob(pattern)
            
            # 删除找到的失败记录文件
            for filepath in matching_files:
                try:
                    # 验证文件内容确保是正确的记录
                    with open(filepath, 'r', encoding='utf-8') as f:
                        record = json.load(f)
                        if (record.get('doc_id') == doc_id and 
                            record.get('chunk_index') == chunk_index and
                            record.get('chunk_identifier') == chunk_identifier):
                            os.remove(filepath)
                            logger.info(f"Removed failed chunk record: {filepath}")
                        else:
                            logger.warning(f"Failed chunk record mismatch, skipping: {filepath}")
                except Exception as file_error:
                    logger.error(f"Error processing failed chunk record {filepath}: {file_error}")
            
            if not matching_files:
                logger.debug(f"No failed chunk record found for {chunk_identifier}")
                
        except Exception as e:
            logger.error(f"Error removing failed chunk record: {e}")
    
    def _clear_failed_document_logs(self, doc_id: str) -> None:
        """清除指定文档的失败日志
        
        Args:
            doc_id: 文档ID
        """
        try:
            import os
            import json
            import glob
            
            failed_docs_dir = "logs/failed_documents"
            if not os.path.exists(failed_docs_dir):
                return
            
            # 查找匹配的失败记录文件
            pattern = os.path.join(failed_docs_dir, "*.json")
            matching_files = glob.glob(pattern)
            
            removed_count = 0
            for filepath in matching_files:
                try:
                    # 验证文件内容确保是正确的记录
                    with open(filepath, 'r', encoding='utf-8') as f:
                        record = json.load(f)
                        if record.get('doc_id') == doc_id:
                            os.remove(filepath)
                            removed_count += 1
                            logger.info(f"Removed failed document record: {filepath}")
                except Exception as file_error:
                    logger.error(f"Error processing failed document record {filepath}: {file_error}")
            
            if removed_count == 0:
                logger.debug(f"No failed document records found for {doc_id}")
            else:
                logger.info(f"Removed {removed_count} failed document records for {doc_id}")
                
        except Exception as e:
            logger.error(f"Error clearing failed document logs: {e}")
    
    def _clear_all_failed_logs(self, include_chunks: bool = True) -> None:
        """清除所有失败日志
        
        Args:
            include_chunks: 是否同时清除chunk失败日志
        """
        try:
            import os
            import shutil
            
            # 清除文档失败日志
            failed_docs_dir = "logs/failed_documents"
            if os.path.exists(failed_docs_dir):
                shutil.rmtree(failed_docs_dir)
                logger.info(f"Cleared all failed document logs from {failed_docs_dir}")
            
            # 清除chunk失败日志
            if include_chunks:
                failed_chunks_dir = "logs/failed_chunks"
                if os.path.exists(failed_chunks_dir):
                    shutil.rmtree(failed_chunks_dir)
                    logger.info(f"Cleared all failed chunk logs from {failed_chunks_dir}")
                    
        except Exception as e:
            logger.error(f"Error clearing all failed logs: {e}")
    
    def get_failed_documents_summary(self) -> Dict[str, Any]:
        """获取失败文档的汇总信息
        
        Returns:
            包含失败文档统计信息的字典
        """
        try:
            import os
            import json
            import glob
            from collections import defaultdict
            
            failed_docs_dir = "logs/failed_documents"
            if not os.path.exists(failed_docs_dir):
                return {
                    "total_failed_documents": 0,
                    "failed_by_error_type": {},
                    "failed_by_category": {},
                    "all_extraction_successful": True,
                    "message": "No failed documents directory found - all extractions successful"
                }
            
            # 查找所有失败记录文件
            pattern = os.path.join(failed_docs_dir, "*.json")
            failed_files = glob.glob(pattern)
            
            if not failed_files:
                return {
                    "total_failed_documents": 0,
                    "failed_by_error_type": {},
                    "failed_by_category": {},
                    "all_extraction_successful": True,
                    "message": "All document metadata extractions successful - no failed documents found"
                }
            
            # 统计失败信息
            failed_by_error_type = defaultdict(int)
            failed_by_category = defaultdict(int)
            failed_documents = []
            
            for filepath in failed_files:
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        record = json.load(f)
                        
                        error_type = record.get('error_type', 'Unknown')
                        category = record.get('document_category', 'Unknown')
                        
                        failed_by_error_type[error_type] += 1
                        failed_by_category[category] += 1
                        
                        failed_documents.append({
                            'doc_id': record.get('doc_id'),
                            'timestamp': record.get('timestamp'),
                            'error_type': error_type,
                            'category': category,
                            'text_length': record.get('text_length', 0),
                            'error_message': record.get('error_message', '')[:200] + '...' if len(record.get('error_message', '')) > 200 else record.get('error_message', ''),
                            'filepath': filepath
                        })
                        
                except Exception as file_error:
                    logger.error(f"Error reading failed document record {filepath}: {file_error}")
            
            return {
                "total_failed_documents": len(failed_documents),
                "failed_by_error_type": dict(failed_by_error_type),
                "failed_by_category": dict(failed_by_category),
                "failed_documents": failed_documents,
                "all_extraction_successful": False,
                "message": f"Found {len(failed_documents)} failed document metadata extractions"
            }
            
        except Exception as e:
            logger.error(f"Error getting failed documents summary: {e}")
            return {
                "total_failed_documents": 0,
                "failed_by_error_type": {},
                "failed_by_category": {},
                "error": str(e),
                "message": "Error occurred while retrieving failed documents summary"
            }
    
    def get_failed_chunks_summary(self) -> Dict[str, Any]:
        """获取失败chunk的汇总信息
        
        Returns:
            包含失败chunk统计信息的字典
        """
        try:
            import os
            import json
            import glob
            from collections import defaultdict
            
            failed_chunks_dir = "logs/failed_chunks"
            if not os.path.exists(failed_chunks_dir):
                return {
                    "total_failed_chunks": 0,
                    "failed_by_document": {},
                    "failed_by_error_type": {},
                    "all_extraction_successful": True,
                    "message": "No failed chunks directory found - all extractions successful"
                }
            
            # 查找所有失败记录文件
            pattern = os.path.join(failed_chunks_dir, "*.json")
            failed_files = glob.glob(pattern)
            
            if not failed_files:
                return {
                    "total_failed_chunks": 0,
                    "failed_by_document": {},
                    "failed_by_error_type": {},
                    "all_extraction_successful": True,
                    "message": "All metadata extractions successful - no failed chunks found"
                }
            
            # 统计失败信息
            failed_by_document = defaultdict(list)
            failed_by_error_type = defaultdict(int)
            total_failed = 0
            
            for filepath in failed_files:
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        record = json.load(f)
                        doc_id = record.get('doc_id', 'unknown')
                        chunk_index = record.get('chunk_index', 'unknown')
                        error_type = record.get('error_type', 'Unknown')
                        timestamp = record.get('timestamp', 'unknown')
                        
                        failed_by_document[doc_id].append({
                            'chunk_index': chunk_index,
                            'error_type': error_type,
                            'timestamp': timestamp,
                            'file_path': filepath
                        })
                        failed_by_error_type[error_type] += 1
                        total_failed += 1
                        
                except Exception as file_error:
                    logger.error(f"Error reading failed chunk record {filepath}: {file_error}")
            
            return {
                "total_failed_chunks": total_failed,
                "failed_by_document": dict(failed_by_document),
                "failed_by_error_type": dict(failed_by_error_type),
                "all_extraction_successful": total_failed == 0,
                "message": f"Found {total_failed} failed chunks" if total_failed > 0 else "All extractions successful"
            }
            
        except Exception as e:
            logger.error(f"Error getting failed chunks summary: {e}")
            return {
                "total_failed_chunks": -1,
                "failed_by_document": {},
                "failed_by_error_type": {},
                "all_extraction_successful": False,
                "message": f"Error checking failed chunks: {e}"
            }
    
    def clean_old_failed_records(self, days_old: int = 7) -> int:
        """清理指定天数之前的失败记录
        
        Args:
            days_old: 清理多少天之前的记录，默认7天
            
        Returns:
            清理的文件数量
        """
        try:
            import os
            import json
            import glob
            from datetime import datetime, timedelta
            
            failed_chunks_dir = "logs/failed_chunks"
            if not os.path.exists(failed_chunks_dir):
                return 0
            
            # 计算截止时间
            cutoff_time = datetime.now() - timedelta(days=days_old)
            
            # 查找所有失败记录文件
            pattern = os.path.join(failed_chunks_dir, "*.json")
            failed_files = glob.glob(pattern)
            
            cleaned_count = 0
            for filepath in failed_files:
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        record = json.load(f)
                        timestamp_str = record.get('timestamp')
                        if timestamp_str:
                            record_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                            if record_time < cutoff_time:
                                os.remove(filepath)
                                cleaned_count += 1
                                logger.info(f"Cleaned old failed record: {filepath}")
                except Exception as file_error:
                    logger.error(f"Error processing failed record {filepath}: {file_error}")
            
            logger.info(f"Cleaned {cleaned_count} old failed records (older than {days_old} days)")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Error cleaning old failed records: {e}")
            return 0
    
    class CustomChunkOutputParser(ChainableOutputParser):
        """自定义chunk输出解析器，处理LLM输出的JSON格式问题"""
        
        def __init__(self, output_cls, verbose: bool = False):
            self.output_cls = output_cls
            self.verbose = verbose
            self.pydantic_parser = PydanticOutputParser(output_cls=output_cls)
        
        def parse(self, output: str):
            """解析LLM输出，处理properties包装等格式问题"""
            if self.verbose:
                logger.debug(f"Raw LLM output: {output}")
            
            try:
                # 尝试标准的Pydantic解析
                return self.pydantic_parser.parse(output)
            except ValidationError as e:
                logger.warning(f"Standard Pydantic validation failed: {e}")
                
                # 尝试处理LLM输出被properties包装的情况
                try:
                    cleaned_output = self._handle_properties_wrapper(output)
                    if cleaned_output != output:
                        logger.info("Detected and handled properties wrapper in LLM output")
                        return self.pydantic_parser.parse(cleaned_output)
                except Exception as cleanup_error:
                    logger.debug(f"Properties wrapper handling failed: {cleanup_error}")
                
                # 如果所有处理都失败，抛出原始ValidationError
                logger.error(f"All parsing attempts failed for output: {output[:200]}...")
                raise e
        
        def _handle_properties_wrapper(self, output: str) -> str:
            """处理LLM输出被properties结构包装的情况"""
            import json
            import re
            
            try:
                # 首先尝试解析为JSON
                parsed_json = json.loads(output)
                
                # 检查是否数据被包装在properties字段中
                if isinstance(parsed_json, dict) and "properties" in parsed_json:
                    properties_content = parsed_json["properties"]
                    if isinstance(properties_content, dict):
                        # 检查properties内容是否像实际数据而不是schema
                        if self._looks_like_data_not_schema(properties_content):
                            logger.info("Extracting data from properties wrapper")
                            return json.dumps(properties_content)
                        else:
                            logger.warning("Detected JSON schema format instead of data - LLM returned schema instead of actual values")
                            # 这种情况下，LLM返回了schema而不是数据，应该抛出错误
                            raise ValueError("LLM returned JSON schema instead of actual data")
                
                # 如果JSON解析成功但不是properties包装，返回原始输出
                return output
                
            except json.JSONDecodeError:
                # 如果不是有效JSON，尝试提取JSON部分
                json_match = re.search(r'\{.*\}', output, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                    try:
                        parsed_json = json.loads(json_str)
                        if isinstance(parsed_json, dict) and "properties" in parsed_json:
                            properties_content = parsed_json["properties"]
                            if isinstance(properties_content, dict) and self._looks_like_data_not_schema(properties_content):
                                logger.info("Extracting data from properties wrapper in embedded JSON")
                                return json.dumps(properties_content)
                    except json.JSONDecodeError as e:
                        logger.warning(f"JSON parsing failed for extracted content: {e}")
                        # 尝试通过括号匹配来提取结构化字段
                        try:
                            cleaned_json = self._extract_json_by_brackets(json_str)
                            if cleaned_json:
                                parsed_json = json.loads(cleaned_json)
                                if isinstance(parsed_json, dict):
                                    # 检查是否有properties包装
                                    if "properties" in parsed_json:
                                        properties_content = parsed_json["properties"]
                                        if isinstance(properties_content, dict) and self._looks_like_data_not_schema(properties_content):
                                            logger.info("Successfully extracted data from properties wrapper using bracket matching")
                                            return json.dumps(properties_content)
                                    else:
                                        # 直接返回解析的JSON
                                        logger.info("Successfully extracted JSON using bracket matching")
                                        return json.dumps(parsed_json)
                        except (json.JSONDecodeError, Exception) as bracket_error:
                            logger.warning(f"Bracket-based JSON extraction also failed: {bracket_error}")
                
                # 如果无法处理，返回原始输出
                return output
        
        def _looks_like_data_not_schema(self, content: dict) -> bool:
            """判断内容是否像实际数据而不是JSON schema"""
            # JSON schema通常包含type, description, title等字段
            schema_indicators = {"type", "description", "title", "items", "required", "enum"}
            
            for key, value in content.items():
                if isinstance(value, dict):
                    # 如果值是字典且包含schema指示符，可能是schema
                    if any(indicator in value for indicator in schema_indicators):
                        return False
                elif isinstance(value, str):
                    # 如果值是字符串且不是schema描述，可能是实际数据
                    continue
                elif isinstance(value, (list, int, float, bool)):
                    # 如果值是基本数据类型，可能是实际数据
                    continue
            
            return True

        def _extract_json_by_brackets(self, json_str: str) -> Optional[str]:
            """通过括号匹配来提取和清理JSON字符串
            
            Args:
                json_str: 可能包含格式问题的JSON字符串
                
            Returns:
                清理后的有效JSON字符串，如果无法修复则返回None
            """
            try:
                # 移除可能的前后空白和换行符
                json_str = json_str.strip()
                
                # 找到第一个 { 和最后一个 } 来确定JSON边界
                start_idx = json_str.find('{')
                end_idx = json_str.rfind('}')
                
                if start_idx == -1 or end_idx == -1 or start_idx >= end_idx:
                    return None
                
                # 提取JSON部分
                json_content = json_str[start_idx:end_idx + 1]
                
                # 尝试通过括号匹配来验证和修复JSON结构
                bracket_count = 0
                quote_count = 0
                in_string = False
                escape_next = False
                valid_end = -1
                
                for i, char in enumerate(json_content):
                    if escape_next:
                        escape_next = False
                        continue
                        
                    if char == '\\' and in_string:
                        escape_next = True
                        continue
                        
                    if char == '"' and not escape_next:
                        in_string = not in_string
                        continue
                        
                    if not in_string:
                        if char == '{':
                            bracket_count += 1
                        elif char == '}':
                            bracket_count -= 1
                            if bracket_count == 0:
                                valid_end = i
                                break
                
                if valid_end > 0:
                    # 截取到有效的结束位置
                    cleaned_json = json_content[:valid_end + 1]
                    
                    # 尝试一些常见的修复
                    cleaned_json = self._fix_common_json_issues(cleaned_json)
                    
                    return cleaned_json
                    
                return None
                
            except Exception as e:
                logger.warning(f"Error in bracket-based JSON extraction: {e}")
                return None
        
        def _fix_common_json_issues(self, json_str: str) -> str:
                """修复常见的JSON格式问题
                
                Args:
                    json_str: 需要修复的JSON字符串
                    
                Returns:
                    修复后的JSON字符串
                """
                try:
                    # 移除可能的尾随逗号
                    json_str = re.sub(r',\s*}', '}', json_str)
                    json_str = re.sub(r',\s*]', ']', json_str)
                    
                    # 逐行处理字符串值中的未转义引号
                    lines = json_str.split('\n')
                    fixed_lines = []
                    
                    for line in lines:
                        if ':' in line and '"' in line:
                            # 直接处理包含引号的字符串值
                            parts = line.split(':', 1)
                            if len(parts) == 2:
                                key_part = parts[0]
                                value_part = parts[1].strip()
                                
                                if value_part.startswith('"'):
                                    # 处理字符串值
                                    if value_part.count('"') > 2:  # 有未转义的引号
                                        # 提取内容
                                        if value_part.endswith(','):
                                            content = value_part[1:-2]
                                            trailing = ','
                                        elif value_part.endswith('"'):
                                            content = value_part[1:-1]
                                            trailing = ''
                                        else:
                                            fixed_lines.append(line)
                                            continue
                                        
                                        # 转义内部引号
                                        escaped_content = content.replace('"', '\\"')
                                        fixed_line = f'{key_part}: "{escaped_content}"{trailing}'
                                        fixed_lines.append(fixed_line)
                                    else:
                                        fixed_lines.append(line)
                                else:
                                    fixed_lines.append(line)
                            else:
                                fixed_lines.append(line)
                        else:
                            fixed_lines.append(line)
                    
                    return '\n'.join(fixed_lines)
                    
                except Exception as e:
                    logger.warning(f"Error fixing JSON issues: {e}")
                    return json_str

    
    def _create_chunk_program(self, extract_summary: bool, extract_qa: bool, chunk_template: str) -> LLMTextCompletionProgram:
        """根据需要提取的字段创建chunk级提取程序"""
        
        # 动态创建Pydantic模型的字段
        annotations = {
            "title": str,
            "keywords": List[str],
        }
        
        # 从ChunkLevelMetadata模型中获取正确的字段描述
        title_field = ChunkLevelMetadata.model_fields.get("title")
        keywords_field = ChunkLevelMetadata.model_fields.get("keywords")
        
        field_definitions = {
            "title": Field(..., description=title_field.description if title_field else "chunk标题，应该简洁准确地概括chunk内容"),
            "keywords": Field(..., description=keywords_field.description if keywords_field else "关键词列表，提取3-6个最重要的关键词"),
        }
        
        optional_fields_prompt = []
        optional_fields_desc = ""

        if extract_summary:
            summary_field = ChunkLevelMetadata.model_fields.get("summary")
            if summary_field:
                annotations["summary"] = Optional[str]
                field_definitions["summary"] = Field(None, description=summary_field.description)
                optional_fields_prompt.append(f"- **summary**: {summary_field.description}")

        if extract_qa:
            qa_field = ChunkLevelMetadata.model_fields.get("questions_answered")
            if qa_field:
                annotations["questions_answered"] = Optional[List[str]]
                field_definitions["questions_answered"] = Field(None, description=qa_field.description)
                optional_fields_prompt.append(f"- **questions_answered**: {qa_field.description}")

        if optional_fields_prompt:
            optional_fields_desc = "\n请提取以下可选信息：\n" + "\n".join(optional_fields_prompt)
        
        # 动态创建模型类
        class_dict = {
            "__annotations__": annotations,
            **field_definitions
        }
        
        DynamicChunkMetadata = type("DynamicChunkMetadata", (BaseModel,), class_dict)
        
        # 创建程序
        template = chunk_template.replace("{optional_fields}", optional_fields_desc)
        
        # 使用标准的PydanticOutputParser创建程序
        standard_parser = PydanticOutputParser(output_cls=DynamicChunkMetadata)
        
        return LLMTextCompletionProgram.from_defaults(
            output_parser=standard_parser,
            prompt_template_str=template,
            verbose=True,
            llm=self.llm
        )
    
    async def aextract(self, nodes: Sequence[BaseNode]) -> List[Dict]:
        """异步提取智能元数据
        
        Args:
            nodes: 要处理的节点列表
        """
        logger.info(f"🚀 [METADATA EXTRACTOR] Starting smart metadata extraction for {len(nodes)} nodes")
        
        metadata_list = [None] * len(nodes)  # 预分配列表，保持顺序
        
        # 控制并发数量
        import asyncio
        semaphore = asyncio.Semaphore(8)  # 限制并发数量
        processed_count = 0
        progress_lock = asyncio.Lock()
        
        async def process_node(idx, node):
            nonlocal processed_count
            
            async with semaphore:
                try:
                    doc_id = node.metadata.get("original_file_path", "unknown")
                    node_type = node.metadata.get("node_type", "chunk")
                    text_content = node.get_content()
                    text_length = len(text_content)
                    
                    # 处理原始文档节点 - 提取文档级元数据
                    if node_type == "original_document":
                        logger.info(f"📋 [METADATA EXTRACTOR] Processing original document: {doc_id} (length: {text_length})")
                        
                        # 优化处理策略：对于超大文档，使用摘要或关键部分进行元数据提取
                        optimized_content = self._optimize_document_content_for_extraction(text_content, doc_id)
                        optimized_length = len(optimized_content)
                        
                        if optimized_length != text_length:
                            logger.info(f"🔧 [METADATA EXTRACTOR] Document content optimized: {text_length} -> {optimized_length} characters")
                        
                        # 提取文档级元数据并缓存
                        cached_data = await self._classify_and_extract(optimized_content, doc_id)
                        doc_metadata = cached_data["metadata"]
                        
                        # 为原始文档节点设置特殊的元数据
                        original_doc_metadata = {
                            **doc_metadata,
                            "content_optimization_applied": optimized_length != text_length,
                            "original_content_length": text_length,
                            "optimized_content_length": optimized_length
                        }
                        metadata_list[idx] = original_doc_metadata
                        
                    # 处理切分节点 - 提取chunk级元数据
                    else:
                        logger.debug(f"📄 [METADATA EXTRACTOR] Processing chunk {idx+1}: {doc_id} (length: {text_length})")
                        
                        # 检查文本是否为空或过短
                        if not text_content or not text_content.strip():
                            logger.debug(f"Skipping chunk {idx+1}: empty content")
                            metadata_list[idx] = self._create_fallback_metadata("", doc_id)
                            return
                        
                        if text_length < self.min_chunk_size_for_extraction:
                            logger.debug(f"Skipping chunk {idx+1}: too short ({text_length} < {self.min_chunk_size_for_extraction})")
                            metadata_list[idx] = self._create_fallback_metadata("", doc_id)
                            return
                        
                        # 获取文档级元数据
                        doc_metadata = {}
                        chunk_template = self._create_default_chunk_template()
                        
                        # 尝试从持久化缓存获取文档级元数据
                        if self.persistent_cache_manager:
                            cached_data = self.persistent_cache_manager.get_cached_metadata(doc_id=doc_id)
                            if cached_data:
                                doc_metadata = cached_data.get("metadata", {})
                                chunk_template = cached_data.get("chunk_template", self._create_default_chunk_template())
                            else:
                                # 如果没有原始文档节点，需要提取文档级元数据
                                logger.warning(f"⚠️ [METADATA EXTRACTOR] No original document found for {doc_id}, extracting from chunk")
                                cached_data = await self._classify_and_extract(text_content, doc_id)
                                doc_metadata = cached_data["metadata"]
                                chunk_template = cached_data["chunk_template"]
                        
                        # 检查持久化缓存
                        extract_summary = self._should_extract_summary(text_length)
                        extract_qa = self._should_extract_qa(text_length)
                        
                        extract_config = {
                            'extract_summary': extract_summary,
                            'extract_qa': extract_qa,
                            'max_keywords': self.max_keywords,
                            'min_chunk_size_for_summary': self.min_chunk_size_for_summary,
                            'min_chunk_size_for_qa': self.min_chunk_size_for_qa
                        }
                        
                        chunk_metadata = None
                        if self.persistent_cache_manager:
                            chunk_metadata = self.persistent_cache_manager.get_chunk_metadata_from_cache(
                                doc_id=doc_id, chunk_text=text_content, chunk_index=idx+1
                            )
                        
                        if chunk_metadata:
                            # 从缓存加载成功
                            self._remove_failed_chunk_record(doc_id, idx+1, text_content)
                            logger.debug(f"💾 [METADATA EXTRACTOR] Chunk {idx+1} loaded from cache")
                        else:
                            # 缓存未命中，执行LLM提取
                            logger.debug(f"🤖 [METADATA EXTRACTOR] Chunk {idx+1} calling LLM for extraction")
                            chunk_program = self._create_chunk_program(extract_summary, extract_qa, chunk_template)
                            
                            try:
                                # 添加API请求间隔控制
                                await asyncio.sleep(settings.llm_model_settings.API_REQUEST_INTERVAL)
                                
                                prompt = chunk_program.prompt.format(
                                    context_str=text_content,
                                    text_length=text_length,
                                    max_keywords=self.max_keywords
                                )
                                
                                raw_response = await self.llm.acomplete(prompt)
                                raw_output = raw_response.text
                                
                                custom_parser = self.CustomChunkOutputParser(
                                    output_cls=chunk_program.output_cls,
                                    verbose=True
                                )
                                result = custom_parser.parse(raw_output)
                                
                                if not result or not hasattr(result, 'dict'):
                                    raise ValueError("LLM returned empty or invalid response")
                                
                                chunk_metadata = result.dict()
                                
                                # 保存到chunk缓存
                                if self.persistent_cache_manager:
                                    self.persistent_cache_manager.save_chunk_metadata_to_cache(
                                        doc_id=doc_id, chunk_text=text_content, metadata=chunk_metadata, chunk_index=idx+1
                                    )
                                
                                self._remove_failed_chunk_record(doc_id, idx+1, text_content)
                                logger.debug(f"✅ [METADATA EXTRACTOR] Chunk {idx+1} LLM extraction successful")
                                
                            except Exception as e:
                                import traceback
                                logger.error(f"❌ [METADATA EXTRACTOR] Chunk {idx+1} LLM extraction FAILED: {e}")
                                
                                self._log_failed_chunk(
                                    doc_id=doc_id,
                                    chunk_index=idx+1,
                                    text_content=text_content,
                                    text_length=text_length,
                                    extract_config=extract_config,
                                    error=e,
                                    traceback_info=traceback.format_exc()
                                )
                                
                                chunk_metadata = {
                                    "title": "",
                                    "summary": "",
                                    "keywords": [],
                                    "qa_pairs": [] if extract_qa else [],
                                    "extraction_failed": True,
                                    "error_message": str(e)
                                }
                        
                        # 使用智能合并逻辑合并文档级和chunk级元数据
                        final_metadata = self._merge_document_and_chunk_metadata(doc_metadata, chunk_metadata)
                        metadata_list[idx] = final_metadata
                    
                    # 更新进度
                    async with progress_lock:
                        nonlocal processed_count
                        processed_count += 1
                        if processed_count % 10 == 0 or processed_count == len(nodes):
                            logger.info(f"📊 [METADATA EXTRACTOR] Progress: {processed_count}/{len(nodes)} nodes processed")
                
                except Exception as e:
                    logger.error(f"Unexpected error processing node {idx+1}: {e}")
                    metadata_list[idx] = {
                        "title": f"Node {idx+1} (处理异常)",
                        "keywords": [],
                        "summary": "",
                        "qa_pairs": [],
                        "processing_failed": True,
                        "error_message": str(e)
                    }
                    
                    async with progress_lock:
                        processed_count += 1
        
        # 并发处理所有节点
        tasks = [process_node(idx, node) for idx, node in enumerate(nodes)]
        await asyncio.gather(*tasks)
        
        logger.info(f"🏁 [METADATA EXTRACTOR] Smart metadata extraction completed for {len(nodes)} nodes")
        return metadata_list
    

    

    
    def extract(self, nodes: Sequence[BaseNode]) -> List[Dict]:
        """同步提取智能元数据"""
        import asyncio
        return asyncio.run(self.aextract(nodes))