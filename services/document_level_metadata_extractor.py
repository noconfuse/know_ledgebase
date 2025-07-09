# -*- coding: utf-8 -*-
"""
文档级元数据提取器
基于LlamaIndex的BaseExtractor，返回元数据字典而不是修改节点
"""

import logging
import os
import time
from typing import List, Dict, Any, Optional
from pydantic import Field
from llama_index.core.extractors import BaseExtractor
from llama_index.core.schema import BaseNode
from llama_index.core.program import LLMTextCompletionProgram
from llama_index.core.llms.llm import LLM
from llama_index.core.output_parsers import PydanticOutputParser
from models.metadata_models import DocumentLevelMetadata, LegalDocumentMetadata, PolicyDocumentMetadata
from services.metadata_cache_manager import MetadataCacheManager
from services.metadata_failure_manager import MetadataFailureManager
from config import settings

logger = logging.getLogger(__name__)

# 文档分类模板
CLASSIFICATION_PROMPT_TEMPLATE = """
请根据以下文档内容，将其分类为以下类别之一：
- 'legal_document'（法律文档）：包括法律条文、司法解释、行政法规、部门规章等
- 'policy_document'（政策文档）：包括政策解读、实施细则、通知公告、工作指南等
- 'general_document'（通用文档）：其他类型文档

只返回类别名称，不要包含其他内容。

文档内容：
---
{context_str}
---

类别："""


class DocumentLevelMetadataExtractor(BaseExtractor):
    """文档级元数据提取器
    
    支持文档分类和基于分类的元数据结构选择
    包含本地缓存功能以提高调试效率
    支持重试机制以提高提取成功率
    """
    
    llm: LLM = Field(description="语言模型实例")
    min_doc_size_for_extraction: int = Field(default=100, description="进行元数据提取的最小文档大小")
    max_doc_size_for_extraction: int = Field(default=100000, description="进行元数据提取的最大文档大小")
    enable_cache: bool = Field(default=True, description="是否启用缓存")
    cache_manager: Optional[MetadataCacheManager] = Field(default=None, description="缓存管理器")
    failure_manager: Optional[MetadataFailureManager] = Field(default=None, description="失败记录管理器")
    
    # 重试配置
    max_retries: int = Field(default=3, description="最大重试次数")
    retry_delay: float = Field(default=1.0, description="重试延迟时间（秒）")
    
    def __init__(self, 
                 llm: LLM, 
                 enable_cache: bool = True,
                 cache_dir: str = "cache/metadata",
                 min_doc_size_for_extraction: int = 500,
                 max_doc_size_for_extraction: int = 100000,
                 max_retries: int = 3,
                 retry_delay: float = 1.0,
                 **kwargs):
        # 从kwargs中移除可能冲突的参数
        cache_manager_from_kwargs = kwargs.pop('cache_manager', None)
        failure_manager_from_kwargs = kwargs.pop('failure_manager', None)
        
        # 创建cache_manager和failure_manager，优先使用传入的参数
        cache_manager = cache_manager_from_kwargs or (MetadataCacheManager(cache_dir) if enable_cache else None)
        failure_manager = failure_manager_from_kwargs or MetadataFailureManager()
        
        super().__init__(
            llm=llm,
            min_doc_size_for_extraction=min_doc_size_for_extraction,
            max_doc_size_for_extraction=max_doc_size_for_extraction,
            enable_cache=enable_cache,
            cache_manager=cache_manager,
            failure_manager=failure_manager,
            max_retries=max_retries,
            retry_delay=retry_delay,
            **kwargs
        )
        
    def extract(self, nodes: List[BaseNode]) -> List[Dict[str, Any]]:
        """提取文档级元数据
        
        Args:
            nodes: 输入节点列表
            
        Returns:
            List[Dict[str, Any]]: 每个节点对应的元数据字典列表
        """
        logger.info(f"🚀 [DOC METADATA EXTRACTOR] Starting document-level metadata extraction for {len(nodes)} documents")
        
        metadata_list = []
        
        for i, node in enumerate(nodes):
            metadata_dict = {}
            
            if hasattr(node, 'text'):  # 检查是否为文档节点
                doc_id = node.metadata.get("original_file_path", f"doc_{i}")
                text_length = len(node.text)
                
                logger.info(f"📋 [DOC METADATA EXTRACTOR] Processing document: {doc_id} (length: {text_length})")
                
                # 检查文档大小是否适合提取
                if text_length < self.min_doc_size_for_extraction:
                    logger.info(f"⏭️ Document too small ({text_length} < {self.min_doc_size_for_extraction}), skipping metadata extraction")
                    # 元数据为空字典
                    metadata_list.append({})
                    continue
                
                # 检查缓存
                if self.cache_manager:
                    cached_metadata = self.cache_manager.get_cached_metadata(doc_id, node.text)
                    if cached_metadata:
                        logger.info(f"📋 Document metadata cache hit for: {doc_id}")
                        # 直接展开存储元数据，移除失败记录
                        if self.failure_manager:
                            self.failure_manager.remove_document_failure(doc_id)
                        metadata_list.append(cached_metadata)
                        continue
                    
                try:
                    if text_length > self.max_doc_size_for_extraction:
                        logger.info(f"📏 Document too large ({text_length} > {self.max_doc_size_for_extraction}), using optimized content")
                        # 使用优化后的内容进行元数据提取
                        optimized_content = self._optimize_document_content(node.text)
                        doc_metadata = self._extract_document_metadata_with_retry(optimized_content, doc_id)
                    else:
                        # 直接使用原始内容
                        doc_metadata = self._extract_document_metadata_with_retry(node.text, doc_id)
                    
                    if doc_metadata:
                        # 直接展开存储元数据
                        metadata_dict.update(doc_metadata)
                        
                        # 保存到缓存
                        if self.cache_manager:
                            self.cache_manager.save_metadata_to_cache(doc_id, doc_metadata, node.text)
                            logger.info(f"📋 Document metadata cached for: {doc_id}")
                        
                        # 移除失败记录
                        if self.failure_manager:
                            self.failure_manager.remove_document_failure(doc_id)
                        
                        logger.info(f"✅ Document metadata extracted successfully for: {doc_id}")
                    else:
                        # 这种情况理论上不应该发生，因为失败会以异常形式抛出
                        logger.warning(f"⚠️ Metadata extraction returned None without error for: {doc_id}")
                        metadata_dict = {}

                except Exception as e:
                    error_message = f"元数据提取失败: {str(e)}"
                    logger.error(f"❌ {error_message} for document: {doc_id}")
                    # 记录失败信息
                    if self.failure_manager:
                        self.failure_manager.record_document_failure(doc_id, node.text, error_message)
                    # 元数据为空字典
                    metadata_dict = {}
            
            metadata_list.append(metadata_dict)
        
        logger.info(f"🎉 [DOC METADATA EXTRACTOR] Document-level metadata extraction completed")
        return metadata_list
    
    async def aextract(self, nodes: List[BaseNode]) -> List[Dict[str, Any]]:
        """异步提取文档级元数据
        
        Args:
            nodes: 输入节点列表
            
        Returns:
            List[Dict[str, Any]]: 每个节点对应的元数据字典列表
        """
        # 对于文档级提取，直接调用同步方法
        return self.extract(nodes)
    
    def _optimize_document_content(self, text: str) -> str:
        """优化超大文档内容用于元数据提取"""
        content_length = len(text)
        
        # 小文档直接返回
        if content_length <= 20000:
            return text
            
        logger.info(f"📊 Optimizing large document content: {content_length} chars")
        
        try:
            # 策略1: 提取文档结构和关键部分
            if content_length <= 50000:
                return self._extract_key_sections(text)
            
            # 策略2: 超大文档使用智能摘要
            else:
                return self._extract_document_summary_sections(text)
                
        except Exception as e:
            logger.warning(f"⚠️ Content optimization failed: {e}, using truncated content")
            # 降级策略：截取前配置长度的字符
            fallback_length = 30000
            return text[:fallback_length] + "\n\n[文档内容已截断以优化处理效率]" if content_length > fallback_length else text
    
    def _extract_key_sections(self, content: str) -> str:
        """提取文档的关键章节"""
        lines = content.split('\n')
        key_sections = []
        current_section = []
        
        # 定义关键章节标识符
        section_patterns = [
            r'^(第[一二三四五六七八九十\d]+章|Chapter\s+\d+)',
            r'^(第[一二三四五六七八九十\d]+条|Article\s+\d+)',
            r'^(第[一二三四五六七八九十\d]+节|Section\s+\d+)',
            r'^(\d+\.|\d+、|\([一二三四五六七八九十\d]+\))',
            r'^(#{1,6}\s+)',
            r'^([一二三四五六七八九十]、|[A-Z]\.|\d+\.)\s*[^\s]',
        ]
        
        import re
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # 检查是否是关键章节开始
            is_section_start = any(re.match(pattern, line) for pattern in section_patterns)
            
            if is_section_start:
                # 保存之前的章节
                if current_section and len('\n'.join(current_section)) <= 2000:
                    key_sections.extend(current_section)
                    key_sections.append('')
                
                current_section = [line]
            else:
                current_section.append(line)
                
                # 如果当前章节过长，截断
                if len('\n'.join(current_section)) > 2000:
                    key_sections.extend(current_section[:20])
                    key_sections.append('[章节内容已截断]')
                    key_sections.append('')
                    current_section = []
        
        # 处理最后一个章节
        if current_section and len('\n'.join(current_section)) <= 2000:
            key_sections.extend(current_section)
        
        result = '\n'.join(key_sections)
        
        # 如果提取的内容仍然过长，进一步截断
        if len(result) > 30000:
            result = result[:30000] + "\n\n[内容已优化截断]"
            
        return result
    
    def _extract_document_summary_sections(self, content: str) -> str:
        """提取超大文档的摘要部分"""
        lines = content.split('\n')
        summary_parts = []
        
        # 1. 文档开头
        beginning = '\n'.join(lines[:50])
        if len(beginning) > 2000:
            beginning = beginning[:2000]
        summary_parts.append("=== 文档开头 ===")
        summary_parts.append(beginning)
        summary_parts.append("")
        
        # 2. 提取主要标题和章节
        import re
        title_patterns = [
            r'^(第[一二三四五六七八九十\d]+章.*)',
            r'^(第[一二三四五六七八九十\d]+条.*)',
            r'^(#{1,3}\s+.*)',
            r'^([一二三四五六七八九十]、.*)',
        ]
        
        summary_parts.append("=== 主要章节标题 ===")
        title_count = 0
        for line in lines[50:-50]:
            line = line.strip()
            if any(re.match(pattern, line) for pattern in title_patterns):
                summary_parts.append(line)
                title_count += 1
                if title_count >= 20:
                    break
        summary_parts.append("")
        
        # 3. 文档结尾
        ending = '\n'.join(lines[-20:])
        if len(ending) > 2000:
            ending = ending[-2000:]
        summary_parts.append("=== 文档结尾 ===")
        summary_parts.append(ending)
        
        result = '\n'.join(summary_parts)
        
        if len(result) > 25000:
            result = result[:25000] + "\n\n[超大文档已智能摘要]"
            
        return result
    
    def _extract_document_metadata_with_retry(self, text: str, doc_id: str) -> Optional[Dict[str, Any]]:
        """带重试逻辑的文档级元数据提取
        
        Args:
            text: 文档文本内容
            doc_id: 文档标识符
            
        Returns:
            Optional[Dict[str, Any]]: 提取的元数据字典，失败时返回默认值
        """
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                if attempt > 0:
                    logger.info(f"🔄 Retrying document metadata extraction (attempt {attempt + 1}/{self.max_retries + 1}) for document: {doc_id}")
                    time.sleep(self.retry_delay)
                
                # 尝试提取元数据
                result = self._extract_document_metadata(text, doc_id)
                
                if result:
                    if attempt > 0:
                        logger.info(f"✅ Document metadata extraction succeeded on retry {attempt + 1} for document: {doc_id}")
                    return result
                else:
                    last_error = "LLM returned empty result"
                    
            except Exception as e:
                last_error = str(e)
                logger.warning(f"⚠️ Attempt {attempt + 1} failed for document {doc_id}: {last_error}")
                
                # 如果是最后一次尝试，不再重试
                if attempt == self.max_retries:
                    break
        
        # 所有重试都失败了，向上抛出最后一个遇到的异常
        logger.error(f"❌ All {self.max_retries + 1} attempts failed for document metadata extraction. Document: {doc_id}. Last error: {last_error}")
        raise Exception(f"重试{self.max_retries + 1}次后仍失败: {last_error}")
    
    def _extract_document_metadata(self, text: str, doc_id: str) -> Optional[Dict[str, Any]]:
        """提取文档级元数据（单次尝试）"""
        try:
            # 1. 文档分类
            document_category = self._classify_document(text)
            logger.info(f"📋 Document classified as: {document_category}")
            
            # 2. 根据分类选择元数据模型和模板
            metadata_model, template = self._get_model_and_template(document_category)
            
            # 3. 提取元数据 - 使用CustomOutputParser处理LLM输出
            from .custom_output_parser import CustomOutputParser
            custom_parser = CustomOutputParser(output_cls=metadata_model, verbose=False)
            extraction_program = LLMTextCompletionProgram.from_defaults(
                output_parser=custom_parser,
                output_cls=metadata_model,
                llm=self.llm,
                prompt_template_str=template,
                verbose=False
            )
            
            # 执行提取 - 使用正确的调用方式
            result = extraction_program(
                context_str=text,
                text_length=len(text)
            )
            
            # 转换为字典格式并添加分类信息
            if hasattr(result, 'dict'):
                metadata = result.dict()
            elif hasattr(result, 'model_dump'):
                metadata = result.model_dump()
            else:
                metadata = dict(result)
            
            metadata['document_category'] = document_category
            
            return metadata
                
        except Exception as e:
            logger.error(f"❌ Error extracting document metadata for {doc_id}: {str(e)}")
            # 不在这里记录失败，由上层重试逻辑处理
            raise e
    
    def _classify_document(self, content: str) -> str:
        """
        使用LLM分类文档类型
        
        Args:
            content: 文档内容
            
        Returns:
            str: 文档类型
        """
        try:
            # 截取前2000字符用于分类
            classification_content = content[:2000] if len(content) > 2000 else content
            
            # 使用LLM进行分类
            response = self.llm.complete(
                CLASSIFICATION_PROMPT_TEMPLATE.format(context_str=classification_content)
            )
            
            classification = response.text.strip().lower()
            
            # 验证分类结果
            if 'legal_document' in classification:
                return 'legal_document'
            elif 'policy_document' in classification:
                return 'policy_document'
            elif 'general_document' in classification:
                return 'general_document'
            else:
                logger.warning(f"未识别的分类结果: {classification}，默认为general_document")
                return 'general_document'
                
        except Exception as e:
            logger.error(f"文档分类失败: {str(e)}，默认为general_document")
            return 'general_document'
    
    def _get_model_and_template(self, document_category: str):
        """
        根据文档类别获取对应的模型和模板
        
        Args:
            document_category: 文档类别
            
        Returns:
            tuple: (模型类, 提示模板)
        """
        if document_category == 'legal_document':
            return LegalDocumentMetadata, self._create_legal_document_template()
        elif document_category == 'policy_document':
            return PolicyDocumentMetadata, self._create_policy_document_template()
        else:
            # 默认使用通用文档元数据模型
            return DocumentLevelMetadata, self._create_general_document_template()
    
    def _create_legal_document_template(self) -> str:
        """创建法律文档元数据提取模板"""
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
    
    def _create_policy_document_template(self) -> str:
        """创建政策文档元数据提取模板"""
        fields_desc = []
        for field_name, model_field in PolicyDocumentMetadata.model_fields.items():
            fields_desc.append(f"- **{field_name}**: {model_field.description}")
        
        return f"""
请根据以下政策文档内容，提取文档级的元数据信息。

文档内容:
----------------
{{context_str}}
----------------

请提取以下政策文档级信息：

{chr(10).join(fields_desc)}

请根据给定的结构化格式返回结果。
"""
    
    def _create_fallback_metadata(self) -> Dict[str, Any]:
        """创建回退元数据（保持向后兼容）"""
        return {
            'document_type': '未知文档',
            'target_audience': ['未知'],
            'importance_level': '一般',
            'applicable_scenarios': ['未知'],
            'related_articles': [],
            'document_summary': '元数据提取失败',
            'document_category': 'legal_document',
            'is_fallback': True
        }
    
    def _create_general_document_template(self) -> str:
        """创建通用文档级元数据提取模板"""
        fields_desc = []
        for field_name, model_field in DocumentLevelMetadata.model_fields.items():
            fields_desc.append(f"- **{field_name}**: {model_field.description}")
        
        return f"""
请根据以下文档内容，提取文档级的元数据信息。

文档内容:
----------------
{{context_str}}
----------------

文档长度: {{text_length}} 字符

请仔细分析整个文档，提取以下信息：

{chr(10).join(fields_desc)}

注意事项：
- 请基于整个文档内容进行分析，不要只关注开头部分
- 提取的信息应该准确、简洁、有用
- 关键实体应该是文档中真实出现的重要概念
- 主要话题应该覆盖文档的核心内容领域
- 对于可选字段，如果文档中没有相关信息可以不填写

请根据给定的结构化格式返回结果。
"""