# -*- coding: utf-8 -*-
"""
chunk级元数据提取器
从文档级元数据中继承信息，并提取chunk特定的元数据
支持基于文档分类的动态元数据字段调整
"""

import logging
import hashlib
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Tuple
from pydantic import Field
from llama_index.core.schema import BaseNode, TextNode
from llama_index.core.extractors import BaseExtractor
from llama_index.core.llms import LLM
from llama_index.core.program import LLMTextCompletionProgram
from llama_index.core.output_parsers import PydanticOutputParser
from models.metadata_models import ChunkLevelMetadata
from services.metadata_cache_manager import MetadataCacheManager
from services.metadata_failure_manager import MetadataFailureManager
from config import settings

logger = logging.getLogger(__name__)

class ChunkLevelMetadataExtractor(BaseExtractor):
    """chunk级元数据提取器
    
    提取chunk级元数据并自动继承文档级元数据
    支持并发处理和失败重试机制
    """
    
    llm: LLM = Field(description="语言模型实例")
    min_chunk_size_for_extraction: int = Field(default=100, description="进行元数据提取的最小chunk大小")
    max_keywords: int = Field(default=6, description="最大关键词数量")
    cache_manager: Optional[MetadataCacheManager] = Field(default=None, description="缓存管理器")
    max_chunk_length: int = Field(default=2000, description="最大chunk长度")
    min_chunk_length: int = Field(default=50, description="最小chunk长度")
    failure_manager: Optional[MetadataFailureManager] = Field(default=None, description="失败记录管理器")
    
    # 并发和重试配置
    max_workers: int = Field(default=4, description="最大并发工作线程数")
    max_retries: int = Field(default=3, description="最大重试次数")
    retry_delay: float = Field(default=1.0, description="重试延迟时间（秒）")
    enable_concurrent: bool = Field(default=True, description="是否启用并发处理")
    batch_size: int = Field(default=10, description="批处理大小")
    
    def __init__(
        self,
        llm: LLM,
        min_chunk_size_for_extraction: int = 20,
        max_keywords: int = 6,
        max_workers: int = 4,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        enable_concurrent: bool = True,
        batch_size: int = 10,
        **kwargs
    ):
        # 从kwargs中移除可能冲突的参数
        cache_manager_from_kwargs = kwargs.pop('cache_manager', None)
        failure_manager_from_kwargs = kwargs.pop('failure_manager', None)
        
        # 创建cache_manager和failure_manager，优先使用传入的参数
        cache_manager = cache_manager_from_kwargs or MetadataCacheManager()
        failure_manager = failure_manager_from_kwargs or MetadataFailureManager()
        max_chunk_length = getattr(settings, 'MAX_CHUNK_LENGTH_FOR_METADATA', 2000)
        min_chunk_length = getattr(settings, 'MIN_CHUNK_LENGTH_FOR_METADATA', 50)
        
        super().__init__(
            llm=llm,
            min_chunk_size_for_extraction=min_chunk_size_for_extraction,
            max_keywords=max_keywords,
            cache_manager=cache_manager,
            max_chunk_length=max_chunk_length,
            min_chunk_length=min_chunk_length,
            failure_manager=failure_manager,
            max_workers=max_workers,
            max_retries=max_retries,
            retry_delay=retry_delay,
            enable_concurrent=enable_concurrent,
            batch_size=batch_size,
            **kwargs
        )
        
    def extract(self, nodes: List[BaseNode]) -> List[Dict[str, Any]]:
        """提取chunk级元数据
        
        Args:
            nodes: 输入节点列表
            
        Returns:
            List[Dict[str, Any]]: 每个节点对应的元数据字典列表
        """
        logger.info(f"🚀 [CHUNK METADATA EXTRACTOR] Starting chunk-level metadata extraction for {len(nodes)} chunks")
        
        if self.enable_concurrent and len(nodes) > 1:
            return self._extract_concurrent(nodes)
        else:
            return self._extract_sequential(nodes)
    
    def _extract_sequential(self, nodes: List[BaseNode]) -> List[Dict[str, Any]]:
        """顺序提取元数据（原有逻辑）"""
        metadata_list = []
        processed_count = 0
        
        for i, node in enumerate(nodes):
            metadata_dict = {}
            
            if isinstance(node, TextNode):
                text_content = node.get_content()
                text_length = len(text_content)
                
                # 检查chunk大小
                if text_length < self.min_chunk_size_for_extraction:
                    logger.debug(f"⏭️ Chunk {i+1} too small ({text_length} < {self.min_chunk_size_for_extraction}), skipping")
                    metadata_list.append(metadata_dict)
                    continue
                
                # 获取文档ID
                doc_id = node.metadata.get('original_file_path', 'unknown')
                
                # 尝试从缓存获取
                chunk_id = f"{doc_id}_chunk_{i}"
                cached_metadata = self.cache_manager.get_chunk_metadata_from_cache(doc_id, text_content, i)
                if cached_metadata:
                    logger.debug(f"📋 Chunk metadata cache hit for: {chunk_id}")
                    metadata_dict.update(cached_metadata)
                    
                    # 删除之前的失败记录（如果存在）
                    if self.failure_manager:
                        self.failure_manager.remove_chunk_failure(chunk_id)
                    
                    metadata_list.append(metadata_dict)
                    processed_count += 1
                    continue
                
                # 获取文档级元数据
                doc_metadata = node.metadata.get('document_metadata', {})
                
                # 提取chunk级元数据（带重试）
                chunk_id = f"{doc_id}_chunk_{i}"
                chunk_metadata = self._extract_chunk_metadata_with_retry(text_content, doc_metadata, chunk_id)
                
                if chunk_metadata:
                    # 直接展开元数据
                    metadata_dict.update(chunk_metadata)
                    
                    # 缓存元数据
                    self.cache_manager.save_chunk_metadata_to_cache(doc_id, text_content, chunk_metadata, i)
                    
                    # 删除之前的失败记录（如果存在）
                    if self.failure_manager:
                        self.failure_manager.remove_chunk_failure(chunk_id)
                    
                    processed_count += 1
                    
                    if processed_count % 10 == 0:
                        logger.info(f"📊 Processed {processed_count}/{len(nodes)} chunks")
                else:
                    logger.warning(f"⚠️ Failed to extract chunk metadata for chunk {i+1}")
                    # 记录失败
                    if self.failure_manager:
                        self.failure_manager.record_chunk_failure(chunk_id, text_content, "元数据提取失败")
            
            metadata_list.append(metadata_dict)
        
        logger.info(f"🎉 [CHUNK METADATA EXTRACTOR] Sequential extraction completed. Processed: {processed_count}/{len(nodes)}")
        return metadata_list
    
    def _extract_concurrent(self, nodes: List[BaseNode]) -> List[Dict[str, Any]]:
        """并发提取元数据"""
        logger.info(f"🔄 [CHUNK METADATA EXTRACTOR] Using concurrent extraction with {self.max_workers} workers")
        
        # 初始化结果列表，保持原有顺序
        metadata_list = [{} for _ in nodes]
        processed_count = 0
        
        # 准备需要处理的任务
        tasks = []
        for i, node in enumerate(nodes):
            if isinstance(node, TextNode):
                text_content = node.get_content()
                text_length = len(text_content)
                
                # 检查chunk大小
                if text_length < self.min_chunk_size_for_extraction:
                    logger.debug(f"⏭️ Chunk {i+1} too small ({text_length} < {self.min_chunk_size_for_extraction}), skipping")
                    continue
                
                # 获取文档ID
                doc_id = node.metadata.get('original_file_path', 'unknown')
                chunk_id = f"{doc_id}_chunk_{i}"
                
                # 检查缓存
                cached_metadata = self.cache_manager.get_chunk_metadata_from_cache(doc_id, text_content, i)
                if cached_metadata:
                    logger.debug(f"📋 Chunk metadata cache hit for: {chunk_id}")
                    metadata_list[i].update(cached_metadata)
                    
                    # 删除之前的失败记录（如果存在）
                    if self.failure_manager:
                        self.failure_manager.remove_chunk_failure(chunk_id)
                    
                    processed_count += 1
                    continue
                
                # 获取文档级元数据
                doc_metadata = node.metadata.get('document_metadata', {})
                
                # 添加到任务列表
                tasks.append((i, text_content, doc_metadata, chunk_id, doc_id))
        
        # 分批处理任务
        total_tasks = len(tasks)
        if total_tasks == 0:
            logger.info(f"🎉 [CHUNK METADATA EXTRACTOR] All chunks processed from cache. Total: {processed_count}")
            return metadata_list
        
        logger.info(f"📋 Processing {total_tasks} chunks concurrently in batches of {self.batch_size}")
        
        # 使用线程池执行器进行并发处理
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 分批提交任务
            for batch_start in range(0, total_tasks, self.batch_size):
                batch_end = min(batch_start + self.batch_size, total_tasks)
                batch_tasks = tasks[batch_start:batch_end]
                
                # 提交当前批次的任务
                future_to_task = {
                    executor.submit(self._process_single_chunk, task): task
                    for task in batch_tasks
                }
                
                # 处理完成的任务
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    i, text_content, doc_metadata, chunk_id, doc_id = task
                    
                    try:
                        chunk_metadata = future.result()
                        if chunk_metadata:
                            metadata_list[i].update(chunk_metadata)
                            
                            # 缓存元数据
                            self.cache_manager.save_chunk_metadata_to_cache(doc_id, text_content, chunk_metadata, i)
                            
                            # 删除之前的失败记录（如果存在）
                            if self.failure_manager:
                                self.failure_manager.remove_chunk_failure(chunk_id)
                            
                            processed_count += 1
                        else:
                            logger.warning(f"⚠️ Failed to extract chunk metadata for chunk {i+1}")
                            # 记录失败
                            if self.failure_manager:
                                self.failure_manager.record_chunk_failure(chunk_id, text_content, "元数据提取失败")
                    
                    except Exception as e:
                        logger.error(f"❌ Error processing chunk {i+1}: {str(e)}")
                        # 记录失败
                        if self.failure_manager:
                            self.failure_manager.record_chunk_failure(chunk_id, text_content, f"处理异常: {str(e)}")
                
                # 批次完成后的进度报告
                logger.info(f"📊 Completed batch {batch_start//self.batch_size + 1}/{(total_tasks-1)//self.batch_size + 1}. Processed: {processed_count}/{len(nodes)} chunks")
        
        logger.info(f"🎉 [CHUNK METADATA EXTRACTOR] Concurrent extraction completed. Processed: {processed_count}/{len(nodes)}")
        return metadata_list
    
    def _process_single_chunk(self, task: Tuple[int, str, Dict[str, Any], str, str]) -> Optional[Dict[str, Any]]:
        """处理单个chunk的元数据提取任务"""
        i, text_content, doc_metadata, chunk_id, doc_id = task
        return self._extract_chunk_metadata_with_retry(text_content, doc_metadata, chunk_id)
    
    async def aextract(self, nodes: List[BaseNode]) -> List[Dict[str, Any]]:
        """异步提取chunk级元数据
        
        Args:
            nodes: 输入节点列表
            
        Returns:
            List[Dict[str, Any]]: 每个节点对应的元数据字典列表
        """
        logger.info(f"🚀 [CHUNK METADATA EXTRACTOR] Starting async chunk-level metadata extraction for {len(nodes)} chunks")
        
        if self.enable_concurrent and len(nodes) > 1:
            return await self._extract_async_concurrent(nodes)
        else:
            # 对于单个chunk或禁用并发时，使用同步方法
            return self.extract(nodes)
    
    async def _extract_async_concurrent(self, nodes: List[BaseNode]) -> List[Dict[str, Any]]:
        """异步并发提取元数据"""
        logger.info(f"🔄 [CHUNK METADATA EXTRACTOR] Using async concurrent extraction")
        
        # 初始化结果列表，保持原有顺序
        metadata_list = [{} for _ in nodes]
        processed_count = 0
        
        # 准备需要处理的任务
        tasks = []
        for i, node in enumerate(nodes):
            if isinstance(node, TextNode):
                text_content = node.get_content()
                text_length = len(text_content)
                
                # 检查chunk大小
                if text_length < self.min_chunk_size_for_extraction:
                    logger.debug(f"⏭️ Chunk {i+1} too small ({text_length} < {self.min_chunk_size_for_extraction}), skipping")
                    continue
                
                # 获取文档ID
                doc_id = node.metadata.get('original_file_path', 'unknown')
                chunk_id = f"{doc_id}_chunk_{i}"
                
                # 检查缓存
                cached_metadata = self.cache_manager.get_chunk_metadata_from_cache(doc_id, text_content, i)
                if cached_metadata:
                    logger.debug(f"📋 Chunk metadata cache hit for: {chunk_id}")
                    metadata_list[i].update(cached_metadata)
                    
                    # 删除之前的失败记录（如果存在）
                    if self.failure_manager:
                        self.failure_manager.remove_chunk_failure(chunk_id)
                    
                    processed_count += 1
                    continue
                
                # 获取文档级元数据
                doc_metadata = node.metadata.get('document_metadata', {})
                
                # 添加到任务列表
                tasks.append((i, text_content, doc_metadata, chunk_id, doc_id))
        
        # 分批处理任务
        total_tasks = len(tasks)
        if total_tasks == 0:
            logger.info(f"🎉 [CHUNK METADATA EXTRACTOR] All chunks processed from cache. Total: {processed_count}")
            return metadata_list
        
        logger.info(f"📋 Processing {total_tasks} chunks asynchronously in batches of {self.batch_size}")
        
        # 分批异步处理任务
        for batch_start in range(0, total_tasks, self.batch_size):
            batch_end = min(batch_start + self.batch_size, total_tasks)
            batch_tasks = tasks[batch_start:batch_end]
            
            # 创建异步任务
            async_tasks = [
                self._process_single_chunk_async(task)
                for task in batch_tasks
            ]
            
            # 等待当前批次完成
            results = await asyncio.gather(*async_tasks, return_exceptions=True)
            
            # 处理结果
            for task, result in zip(batch_tasks, results):
                i, text_content, doc_metadata, chunk_id, doc_id = task
                
                if isinstance(result, Exception):
                    logger.error(f"❌ Error processing chunk {i+1}: {str(result)}")
                    # 记录失败
                    if self.failure_manager:
                        self.failure_manager.record_chunk_failure(chunk_id, text_content, f"异步处理异常: {str(result)}")
                elif result:
                    metadata_list[i].update(result)
                    
                    # 缓存元数据
                    self.cache_manager.save_chunk_metadata_to_cache(doc_id, text_content, result, i)
                    
                    # 删除之前的失败记录（如果存在）
                    if self.failure_manager:
                        self.failure_manager.remove_chunk_failure(chunk_id)
                    
                    processed_count += 1
                else:
                    logger.warning(f"⚠️ Failed to extract chunk metadata for chunk {i+1}")
                    # 记录失败
                    if self.failure_manager:
                        self.failure_manager.record_chunk_failure(chunk_id, text_content, "异步元数据提取失败")
            
            # 批次完成后的进度报告
            logger.info(f"📊 Completed async batch {batch_start//self.batch_size + 1}/{(total_tasks-1)//self.batch_size + 1}. Processed: {processed_count}/{len(nodes)} chunks")
        
        logger.info(f"🎉 [CHUNK METADATA EXTRACTOR] Async concurrent extraction completed. Processed: {processed_count}/{len(nodes)}")
        return metadata_list
    
    async def _process_single_chunk_async(self, task: Tuple[int, str, Dict[str, Any], str, str]) -> Optional[Dict[str, Any]]:
        """异步处理单个chunk的元数据提取任务"""
        i, text_content, doc_metadata, chunk_id, doc_id = task
        # 在事件循环中运行同步的重试逻辑
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._extract_chunk_metadata_with_retry, text_content, doc_metadata, chunk_id)
    
    def _generate_chunk_hash(self, text: str) -> str:
        """生成chunk的哈希值用于缓存"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _extract_chunk_metadata(self, text: str, doc_metadata: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """提取chunk级元数据"""
        try:
            # 优化chunk内容长度
            optimized_text = self._optimize_chunk_content(text)
            
            # 创建提取程序 - 使用CustomOutputParser处理LLM输出
            from .custom_output_parser import CustomOutputParser
            custom_parser = CustomOutputParser(output_cls=ChunkLevelMetadata, verbose=False)
            extraction_program = LLMTextCompletionProgram.from_defaults(
                output_parser=custom_parser,
                output_cls=ChunkLevelMetadata,
                llm=self.llm,
                prompt_template_str=self._create_chunk_template(),
                verbose=False
            )
            
            # 准备上下文信息
            doc_context = ""
            if doc_metadata:
                doc_context = f"""
文档背景信息：
- 文档标题：{doc_metadata.get('title', '未知')}
- 文档类型：{doc_metadata.get('document_type', '未知')}
- 主要话题：{', '.join(doc_metadata.get('main_topics', []))}
- 关键实体：{', '.join(doc_metadata.get('key_entities', []))}
"""
            
            # 执行提取
            result = extraction_program(
                context_str=optimized_text,
                text_length=len(optimized_text),
                doc_context=doc_context,
                max_keywords=self.max_keywords
            )
            
            # 转换为字典
            if hasattr(result, 'dict'):
                return result.dict()
            elif hasattr(result, 'model_dump'):
                return result.model_dump()
            else:
                return dict(result)
                
        except Exception as e:
            logger.error(f"❌ Error extracting chunk metadata: {str(e)}")
            # 记录详细的失败原因已在上层处理
            # 返回默认元数据，动态生成字段
            return self._create_default_chunk_metadata()
    
    def _extract_chunk_metadata_with_retry(self, text: str, doc_metadata: Dict[str, Any], chunk_id: str = None) -> Optional[Dict[str, Any]]:
        """带重试逻辑的chunk元数据提取
        
        Args:
            text: chunk文本内容
            doc_metadata: 文档级元数据
            chunk_id: chunk标识符（用于失败记录）
            
        Returns:
            Optional[Dict[str, Any]]: 提取的元数据字典，失败时返回默认值
        """
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                if attempt > 0:
                    logger.info(f"🔄 Retrying chunk metadata extraction (attempt {attempt + 1}/{self.max_retries + 1}) for chunk: {chunk_id or 'unknown'}")
                    time.sleep(self.retry_delay)
                
                # 尝试提取元数据
                result = self._extract_chunk_metadata(text, doc_metadata)
                
                if result:
                    if attempt > 0:
                        logger.info(f"✅ Chunk metadata extraction succeeded on retry {attempt + 1} for chunk: {chunk_id or 'unknown'}")
                    return result
                else:
                    last_error = "LLM returned empty result"
                    
            except Exception as e:
                last_error = str(e)
                logger.warning(f"⚠️ Attempt {attempt + 1} failed for chunk {chunk_id or 'unknown'}: {last_error}")
                
                # 如果是最后一次尝试，不再重试
                if attempt == self.max_retries:
                    break
        
        # 所有重试都失败了
        logger.error(f"❌ All {self.max_retries + 1} attempts failed for chunk metadata extraction. Chunk: {chunk_id or 'unknown'}. Last error: {last_error}")
        
        # 记录失败（如果有失败管理器）
        if self.failure_manager and chunk_id:
            self.failure_manager.record_chunk_failure(chunk_id, text, f"重试{self.max_retries + 1}次后仍失败: {last_error}")
        
        # 返回默认元数据
        return self._create_default_chunk_metadata()
    
    def _create_default_chunk_metadata(self) -> Dict[str, Any]:
        """动态创建默认chunk级元数据"""
        default_metadata = {}
        
        for field_name, model_field in ChunkLevelMetadata.model_fields.items():
            # 根据字段类型设置默认值
            if model_field.annotation == str:
                if 'score' in field_name or 'level' in field_name:
                    default_metadata[field_name] = "中等"
                elif 'tone' in field_name:
                    default_metadata[field_name] = "中性"
                elif 'type' in field_name:
                    default_metadata[field_name] = "未知类型"
                elif 'topic' in field_name:
                    default_metadata[field_name] = "未知主题"
                else:
                    default_metadata[field_name] = "未知"
            elif hasattr(model_field.annotation, '__origin__') and model_field.annotation.__origin__ == list:
                default_metadata[field_name] = []
            elif model_field.annotation == float:
                default_metadata[field_name] = 0.5
            elif model_field.annotation == int:
                default_metadata[field_name] = 0
            else:
                default_metadata[field_name] = None
        
        return default_metadata
    
    def _optimize_chunk_content(self, text: str) -> str:
        """优化chunk内容长度"""
        if len(text) <= self.max_chunk_length:
            return text
        
        # 如果文本过长，截取前部分内容
        truncated = text[:self.max_chunk_length]
        # 尝试在句号处截断
        last_period = truncated.rfind('。')
        if last_period > self.min_chunk_length:
            return truncated[:last_period + 1]
        
        return truncated + "..."
    
    def _create_chunk_template(self) -> str:
        """动态创建chunk级元数据提取模板"""
        # 从ChunkLevelMetadata模型中动态生成字段描述
        fields_desc = []
        for field_name, model_field in ChunkLevelMetadata.model_fields.items():
            fields_desc.append(f"- **{field_name}**: {model_field.description}")
        
        return f"""
请根据以下文本片段内容，结合文档背景信息，提取chunk级的元数据信息。

{{doc_context}}

文本片段内容:
----------------
{{context_str}}
----------------

文本长度: {{text_length}} 字符
最大关键词数量: {{max_keywords}}

请仔细分析这个文本片段，提取以下信息：

{chr(10).join(fields_desc)}

请以JSON格式返回提取的元数据，确保所有字段都包含在内：

{{
  "{list(ChunkLevelMetadata.model_fields.keys())[0]}": "...",
  "{list(ChunkLevelMetadata.model_fields.keys())[1]}": [...],
  "{list(ChunkLevelMetadata.model_fields.keys())[2]}": [...],
  "{list(ChunkLevelMetadata.model_fields.keys())[3]}": [...],
  "{list(ChunkLevelMetadata.model_fields.keys())[4]}": "...",
  "{list(ChunkLevelMetadata.model_fields.keys())[5]}": 0.0,
  "{list(ChunkLevelMetadata.model_fields.keys())[6]}": 0.0,
  "{list(ChunkLevelMetadata.model_fields.keys())[7]}": 0.0,
  "{list(ChunkLevelMetadata.model_fields.keys())[8]}": "...",
  "{list(ChunkLevelMetadata.model_fields.keys())[9]}": "..."
}}

注意事项：
- 请结合文档背景信息来理解这个chunk在整个文档中的作用
- 关键词应该是这个chunk中最重要的概念，不要重复文档级的通用关键词
- 必须返回有效的JSON格式
- 数值类型字段请返回0.0-1.0之间的浮点数
- 列表类型字段请返回字符串数组
"""