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
from typing import List, Dict, Sequence, Any, Optional
from pydantic import BaseModel, Field
from llama_index.core.extractors import BaseExtractor
from llama_index.core.program import LLMTextCompletionProgram
from llama_index.core.extractors import PydanticProgramExtractor
from llama_index.core.schema import BaseNode, TextNode, Document

logger = logging.getLogger(__name__)


class DocumentLevelMetadata(BaseModel):
    """文档级元数据模型 - 每个文档只提取一次"""
    
    # 文档整体属性
    document_type: str = Field(
        ...,
        description="文档类型分类，如：法律条文、司法解释、行政法规、合同条款、案例分析、理论文章、操作指南等"
    )
    
    difficulty_level: str = Field(
        ...,
        description="内容难度等级：初级（基础概念）、中级（实务应用）、高级（复杂理论）"
    )
    
    legal_domain: List[str] = Field(
        ...,
        description="法律领域分类，如：民法、刑法、行政法、商法、劳动法、知识产权法等，最多3个"
    )
    
    target_audience: List[str] = Field(
        ...,
        description="目标受众，如：律师、法官、企业法务、普通公民、学生等"
    )
    
    importance_level: str = Field(
        ...,
        description="重要性等级：核心（基础重要条文）、重要（常用条文）、一般（特定情况适用）"
    )
    
    applicable_scenarios: List[str] = Field(
        ...,
        description="适用场景列表，描述该内容在哪些具体情况下会被使用，如：合同纠纷、刑事案件、行政处罚、企业合规等，最多3个"
    )
    
    related_articles: List[str] = Field(
        default_factory=list,
        description="相关法条或条款，提取文中提到的其他法律条文、司法解释等"
    )


class ChunkLevelMetadata(BaseModel):
    """chunk级元数据模型 - 每个chunk都会提取"""
    
    title: str = Field(
        ..., 
        description="chunk标题，应该简洁准确地概括chunk内容。对于法律条文，格式如：第X条 关于XXX的规定"
    )
    
    keywords: List[str] = Field(
        ..., 
        description="关键词列表，提取3-6个最重要的关键词。重点提取：法律概念、适用范围、责任主体、处罚措施、法律术语等"
    )
    
    # 可选的chunk级元数据（根据chunk大小决定是否提取）
    summary: Optional[str] = Field(
        None,
        description="chunk摘要，仅对较长的chunk提取。对于法律条文，应概括条文的适用情形、法律后果、关键要素"
    )
    
    questions_answered: Optional[List[str]] = Field(
        None,
        description="问答对列表，仅对较长的chunk提取。生成2-3个这段文本可以回答的问题"
    )
    
    practical_tips: Optional[List[str]] = Field(
        None,
        description="实务要点或注意事项，仅对包含实务内容的chunk提取"
    )


class SmartMetadataExtractor(BaseExtractor):
    """智能元数据提取器"""
    
    # Pydantic 字段声明
    llm: Any = Field(description="语言模型实例")
    min_chunk_size_for_summary: int = Field(default=500, description="提取摘要的最小chunk大小")
    min_chunk_size_for_qa: int = Field(default=300, description="提取问答对的最小chunk大小")
    max_keywords: int = Field(default=5, description="最大关键词数量")
    num_questions: int = Field(default=3, description="问答对数量")
    show_progress: bool = Field(default=True, description="是否显示进度")
    extract_mode: str = Field(default="enhanced", description="提取模式")
    document_metadata_cache: Dict[str, Any] = Field(default_factory=dict, description="文档级元数据缓存")
    document_template: str = Field(default="", description="文档级元数据提取模板")
    chunk_template: str = Field(default="", description="chunk级元数据提取模板")
    document_program: Optional[Any] = Field(default=None, description="文档级元数据提取程序")
    
    def __init__(
        self,
        llm: Any,
        min_chunk_size_for_summary: int = 500,  # 超过500字符才提取摘要
        min_chunk_size_for_qa: int = 300,       # 超过300字符才提取问答对
        max_keywords: int = 5,
        num_questions: int = 3,
        show_progress: bool = True,
        extract_mode: str = "enhanced",  # "basic" 或 "enhanced"
        **kwargs
    ):
        """
        初始化智能元数据提取器
        
        Args:
            llm: 语言模型实例
            min_chunk_size_for_summary: 提取摘要的最小chunk大小
            min_chunk_size_for_qa: 提取问答对的最小chunk大小
            max_keywords: 最大关键词数量
            num_questions: 问答对数量
            show_progress: 是否显示进度
            extract_mode: 提取模式，"basic"只提取基础元数据，"enhanced"提取增强元数据
        """
        super().__init__(
            llm=llm,
            min_chunk_size_for_summary=min_chunk_size_for_summary,
            min_chunk_size_for_qa=min_chunk_size_for_qa,
            max_keywords=max_keywords,
            num_questions=num_questions,
            show_progress=show_progress,
            extract_mode=extract_mode,
            **kwargs
        )
        
        # 初始化模板
        self._setup_templates()
        
        # 创建文档级元数据提取程序
        if self.extract_mode == "enhanced":
            self.document_program = LLMTextCompletionProgram.from_defaults(
                output_cls=DocumentLevelMetadata,
                prompt_template_str=self.document_template,
                verbose=True,
                llm=self.llm
            )
        
        logger.info(f"SmartMetadataExtractor initialized with mode={extract_mode}, min_summary_size={min_chunk_size_for_summary}, min_qa_size={min_chunk_size_for_qa}")
    
    def _setup_templates(self):
        """设置提取模板"""
        self.document_template = """
请根据以下完整文档内容，提取文档级的元数据信息。这些信息将应用于整个文档的所有部分。请使用与原文相同的语言回答。

文档内容：
----------------
{context_str}
----------------

请提取以下文档级信息：

1. **文档类型**：判断文档类型，选择最合适的分类：法律条文、司法解释、行政法规、合同条款、案例分析、理论文章、操作指南、其他。

2. **难度等级**：评估整体内容难度，选择：初级（基础概念，易于理解）、中级（需要一定法律基础）、高级（复杂理论或专业性强）。

3. **法律领域**：确定所属的法律领域，最多3个，如：民法、刑法、行政法、商法、劳动法、知识产权法、环境法、税法等。

4. **目标受众**：确定主要的目标受众群体，如：律师、法官、企业法务、普通公民、学生、政府工作人员等。

5. **重要性等级**：评估重要性，选择：核心（基础重要，经常使用）、重要（较常使用）、一般（特定情况适用）。

6. **适用场景**：列出最多3个具体的适用场景，如：合同纠纷、刑事案件、行政处罚、企业合规、日常咨询等。

7. **相关法条**：提取文中明确提到的其他法律条文、司法解释等（如果没有则留空）。

请根据给定的结构化格式返回结果。
"""
        
        # chunk级元数据提取模板
        self.chunk_template = """
请根据以下文本片段内容，提取chunk级的元数据信息。请使用与原文相同的语言回答。

文本内容：
----------------
{context_str}
----------------

文本长度：{text_length} 字符

请提取以下信息：

1. **标题**：提取一个简洁准确的标题。对于法律条文，请提取条文编号和条文主题，格式如：第X条 关于XXX的规定。

2. **关键词**：提取{max_keywords}个最重要的关键词。重点提取：法律概念、适用范围、责任主体、处罚措施、法律术语等。避免使用停用词。

{optional_fields}

请根据给定的结构化格式返回结果。
"""
    
    def _extract_document_metadata(self, document_text: str, doc_id: str) -> Dict[str, Any]:
        """提取文档级元数据"""
        if doc_id in self.document_metadata_cache:
            return self.document_metadata_cache[doc_id]
        
        if self.extract_mode != "enhanced":
            # 基础模式不提取文档级元数据
            return {}
        
        try:
            logger.info(f"Extracting document-level metadata for document {doc_id}")
            
            # 如果文档太长，只取前2000字符进行文档级分析
            analysis_text = document_text[:2000] if len(document_text) > 2000 else document_text
            
            result = self.document_program(
                context_str=analysis_text
            )
            
            doc_metadata = {
                "document_type": result.document_type,
                "difficulty_level": result.difficulty_level,
                "legal_domain": result.legal_domain,
                "target_audience": result.target_audience,
                "importance_level": result.importance_level,
                "applicable_scenarios": result.applicable_scenarios,
                "related_articles": result.related_articles,
            }
            
            # 缓存结果
            self.document_metadata_cache[doc_id] = doc_metadata
            logger.info(f"Document-level metadata extracted successfully for {doc_id}")
            
            return doc_metadata
            
        except Exception as e:
            logger.error(f"Error extracting document metadata for {doc_id}: {e}")
            return {}
    
    def _should_extract_summary(self, text_length: int) -> bool:
        """判断是否应该提取摘要"""
        return text_length >= self.min_chunk_size_for_summary
    
    def _should_extract_qa(self, text_length: int) -> bool:
        """判断是否应该提取问答对"""
        return text_length >= self.min_chunk_size_for_qa
    
    def _create_chunk_program(self, extract_summary: bool, extract_qa: bool) -> LLMTextCompletionProgram:
        """根据需要提取的字段创建chunk级提取程序"""
        
        # 动态创建Pydantic模型的字段
        annotations = {
            "title": str,
            "keywords": List[str],
        }
        
        field_definitions = {
            "title": Field(..., description="chunk标题，应该简洁准确地概括chunk内容"),
            "keywords": Field(..., description="关键词列表，提取3-6个最重要的关键词"),
        }
        
        optional_fields_desc = []
        
        if extract_summary:
            annotations["summary"] = str
            field_definitions["summary"] = Field(..., description="chunk摘要，简洁概括chunk的主要内容")
            optional_fields_desc.append("3. **摘要**：写一个简洁的摘要。对于法律条文，请概括条文的适用情形、法律后果、关键要素。")
        
        if extract_qa:
            annotations["questions_answered"] = List[str]
            field_definitions["questions_answered"] = Field(..., description=f"问答对列表，生成{self.num_questions}个这段文本可以回答的问题")
            optional_fields_desc.append(f"4. **问答对**：生成{self.num_questions}个这段文本可以回答的问题。重点生成：什么情况下适用、违反后果是什么、适用主体是谁、如何执行等类型的问题。")
        
        # 动态创建模型类
        class_dict = {
            "__annotations__": annotations,
            **field_definitions
        }
        
        DynamicChunkMetadata = type("DynamicChunkMetadata", (BaseModel,), class_dict)
        
        # 构建可选字段描述
        optional_fields_str = "\n".join(optional_fields_desc) if optional_fields_desc else "（无额外字段需要提取）"
        
        # 创建程序
        template = self.chunk_template.replace("{optional_fields}", optional_fields_str)
        
        return LLMTextCompletionProgram.from_defaults(
            output_cls=DynamicChunkMetadata,
            prompt_template_str=template,
            verbose=True,
            llm=self.llm
        )
    
    async def aextract(self, nodes: Sequence[BaseNode]) -> List[Dict]:
        """异步提取智能元数据"""
        logger.info(f"Starting smart metadata extraction for {len(nodes)} nodes")
        
        metadata_list = []
        
        # 按文档分组处理
        nodes_by_doc = {}
        for node in nodes:
            doc_id = node.metadata.get("file_path", "unknown")
            if doc_id not in nodes_by_doc:
                nodes_by_doc[doc_id] = []
            nodes_by_doc[doc_id].append(node)
        
        for doc_id, doc_nodes in nodes_by_doc.items():
            logger.info(f"Processing document {doc_id} with {len(doc_nodes)} chunks")
            
            # 提取文档级元数据（仅一次）
            if self.extract_mode == "enhanced":
                # 合并所有chunk的文本来分析文档级元数据
                full_doc_text = "\n\n".join([node.get_content() for node in doc_nodes])
                doc_metadata = self._extract_document_metadata(full_doc_text, doc_id)
            else:
                doc_metadata = {}
            
            # 处理每个chunk
            for i, node in enumerate(doc_nodes):
                try:
                    text_content = node.get_content()
                    text_length = len(text_content)
                    
                    logger.info(f"Processing chunk {i+1}/{len(doc_nodes)} (length: {text_length} chars)")
                    
                    # 判断是否需要提取可选字段
                    extract_summary = self._should_extract_summary(text_length)
                    extract_qa = self._should_extract_qa(text_length)
                    
                    logger.info(f"Chunk {i+1}: extract_summary={extract_summary}, extract_qa={extract_qa}")
                    
                    # 创建对应的提取程序
                    chunk_program = self._create_chunk_program(extract_summary, extract_qa)
                    
                    # 提取chunk级元数据
                    result = chunk_program(
                        context_str=text_content,
                        text_length=text_length,
                        max_keywords=self.max_keywords
                    )
                    
                    # 构建最终元数据
                    chunk_metadata = {
                        "title": result.title,
                        "keywords": result.keywords,
                    }
                    
                    # 添加可选字段
                    if extract_summary and hasattr(result, "summary"):
                        chunk_metadata["summary"] = result.summary
                    
                    if extract_qa and hasattr(result, "questions_answered"):
                        chunk_metadata["questions_answered"] = result.questions_answered
                    
                    # 合并文档级元数据
                    final_metadata = {**doc_metadata, **chunk_metadata}
                    
                    metadata_list.append(final_metadata)
                    
                    logger.info(f"Chunk {i+1} metadata extracted successfully")
                    
                except Exception as e:
                    logger.error(f"Error extracting metadata for chunk {i+1} in document {doc_id}: {e}")
                    # 添加基础元数据作为fallback
                    fallback_metadata = {
                        "title": f"Chunk {i+1}",
                        "keywords": [],
                        **doc_metadata
                    }
                    metadata_list.append(fallback_metadata)
        
        logger.info(f"Smart metadata extraction completed for {len(nodes)} nodes")
        return metadata_list
    
    def extract(self, nodes: Sequence[BaseNode]) -> List[Dict]:
        """同步提取智能元数据"""
        import asyncio
        return asyncio.run(self.aextract(nodes))