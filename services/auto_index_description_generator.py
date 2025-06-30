# -*- coding: utf-8 -*-
"""
自动索引描述生成器
基于向量数据库的内容自动生成索引描述，而不需要通过接口传递
"""

import logging
from typing import Dict, Any, List, Optional
from llama_index.core.program import LLMTextCompletionProgram
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.core import VectorStoreIndex
from pydantic import BaseModel, Field
import json
from collections import Counter

logger = logging.getLogger(__name__)


class IndexDescription(BaseModel):
    """索引描述模型"""
    
    description: str = Field(
        ...,
        description="详细的索引描述，说明索引包含的内容类型、适用场景和主要特点"
    )


class AutoIndexDescriptionGenerator:
    """自动索引描述生成器"""
    
    def __init__(self, llm: Any):
        """
        初始化自动索引描述生成器
        
        Args:
            llm: 语言模型实例
        """
        self.llm = llm
        
        # 描述生成模板
        self.description_template = """
基于以下索引的内容分析，请生成一个简洁而全面的索引描述。

索引统计信息：
- 文档数量：{doc_count}
- 节点数量：{node_count}
- 主要关键词：{top_keywords}
- 文档类型分布：{doc_types}

样本内容摘要：
{sample_summaries}

样本标题：
{sample_titles}

请根据以上信息生成一个详细的索引描述（150-300字），说明：
- 索引包含的内容类型和范围
- 涵盖的主要话题和领域
- 适用的场景和用途
- 主要特点和优势
- 目标用户群体

请确保描述准确、专业且易于理解。
"""
        
        # 创建LLMTextCompletionProgram
        self.program = LLMTextCompletionProgram.from_defaults(
            output_cls=IndexDescription,
            prompt_template_str=self.description_template,
            verbose=True,
            llm=self.llm
        )
        
        logger.info("AutoIndexDescriptionGenerator initialized")
    
    def analyze_index_content(self, index: VectorStoreIndex, sample_size: int = 50) -> Dict[str, Any]:
        """
        分析索引内容，提取统计信息
        
        Args:
            index: 向量存储索引
            sample_size: 采样大小
            
        Returns:
            索引内容分析结果
        """
        logger.info(f"Analyzing index content with sample size: {sample_size}")
        
        try:
            # 获取所有节点
            retriever = index.as_retriever(similarity_top_k=sample_size)
            
            # 使用一个通用查询来获取代表性节点
            sample_query = "内容 摘要 信息 文档 主题"
            nodes = retriever.retrieve(sample_query)
            
            if not nodes:
                logger.warning("No nodes retrieved from index")
                return self._get_empty_analysis()
            
            # 提取元数据信息
            all_keywords = []
            all_summaries = []
            all_titles = []
            doc_types = []

            for node in nodes:
                metadata = node.metadata if hasattr(node, 'metadata') else {}

                # 提取关键词
                keywords = metadata.get('excerpt_keywords', '')
                if keywords:
                    if isinstance(keywords, str):
                        all_keywords.extend([k.strip() for k in keywords.split(',') if k.strip()])
                    elif isinstance(keywords, list):
                        all_keywords.extend([str(k).strip() for k in keywords if str(k).strip()])

                # 提取摘要
                summary = metadata.get('section_summary', '')
                if summary:
                    all_summaries.append(summary)

                # 提取标题
                title = metadata.get('document_title', '')
                if title:
                    all_titles.append(title)

                # 提取文档类型
                doc_type = metadata.get('document_type', '')
                if doc_type:
                    doc_types.append(doc_type)

            # 统计分析
            top_keywords = [item[0] for item in Counter(all_keywords).most_common(10)]
            doc_type_dist = dict(Counter(doc_types).most_common(5))
            
            # 选择代表性摘要和标题
            sample_summaries = all_summaries[:5] if all_summaries else []
            sample_titles = all_titles[:8] if all_titles else []
            
            analysis_result = {
                'doc_count': len(set(node.metadata.get('file_path', f'doc_{i}') for i, node in enumerate(nodes))),
                'node_count': len(nodes),
                'top_keywords': top_keywords,
                'doc_types': doc_type_dist,
                'sample_summaries': sample_summaries,
                'sample_titles': sample_titles
            }
            
            logger.info(f"Index analysis completed: {len(nodes)} nodes analyzed")
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error analyzing index content: {str(e)}")
            return self._get_empty_analysis()
    
    def _get_empty_analysis(self) -> Dict[str, Any]:
        """返回空的分析结果"""
        return {
            'doc_count': 0,
            'node_count': 0,
            'top_keywords': [],
            'doc_types': {},
            'sample_summaries': [],
            'sample_titles': []
        }
    
    def generate_description(self, index: VectorStoreIndex, sample_size: int = 50) -> str:
        """
        生成索引描述
        
        Args:
            index: 向量存储索引
            sample_size: 分析采样大小
            
        Returns:
            生成的索引描述JSON字符串
        """
        logger.info("Starting index description generation")
        
        try:
            # 分析索引内容
            analysis = self.analyze_index_content(index, sample_size)
            
            # 格式化分析结果用于提示词
            formatted_analysis = {
                'doc_count': analysis['doc_count'],
                'node_count': analysis['node_count'],
                'top_keywords': ', '.join(analysis['top_keywords'][:10]),
                'doc_types': ', '.join([f"{k}({v})" for k, v in analysis['doc_types'].items()]),
                'sample_summaries': '\n'.join([f"- {s[:100]}..." for s in analysis['sample_summaries'][:3]]),
                'sample_titles': '\n'.join([f"- {t}" for t in analysis['sample_titles'][:5]])
            }

            # 使用LLM生成描述
            description_result = self.program(
                doc_count=formatted_analysis['doc_count'],
                node_count=formatted_analysis['node_count'],
                top_keywords=formatted_analysis['top_keywords'],
                doc_types=formatted_analysis['doc_types'],
                sample_summaries=formatted_analysis['sample_summaries'],
                sample_titles=formatted_analysis['sample_titles']
            )
            
            # 直接返回description字符串
            logger.info("Index description generated successfully")
            return description_result.description
            
        except Exception as e:
            logger.error(f"Error generating index description: {str(e)}")
            # 返回基础描述
            return self._generate_fallback_description(analysis if 'analysis' in locals() else self._get_empty_analysis())
    
    def _generate_fallback_description(self, analysis: Dict[str, Any]) -> str:
        """生成备用描述"""
        doc_count = analysis.get('doc_count', 0)
        node_count = analysis.get('node_count', 0)
        top_keywords = analysis.get('top_keywords', [])
        doc_types = analysis.get('doc_types', {})

        # 构建简单的描述文本
        description_parts = [
            f"该知识库索引包含{doc_count}个文档和{node_count}个文本节点",
        ]

        if top_keywords:
            keywords_str = '、'.join(top_keywords[:5])
            description_parts.append(f"主要涵盖{keywords_str}等主题")

        if doc_types:
            types_str = '、'.join(list(doc_types.keys())[:3])
            description_parts.append(f"包含{types_str}等类型的文档")

        description_parts.append("适用于一般性的知识问答、内容检索和信息查询等多种应用场景，为用户提供全面的信息检索服务")

        return '，'.join(description_parts) + '。'