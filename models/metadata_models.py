# -*- coding: utf-8 -*-
"""
数据模型定义
"""

from typing import List, Optional
from pydantic import BaseModel, Field


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
  