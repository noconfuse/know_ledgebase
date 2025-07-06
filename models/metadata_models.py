# -*- coding: utf-8 -*-
"""
数据模型定义
"""

from typing import List, Optional
from pydantic import BaseModel, Field

class DocumentLevelMetadata(BaseModel):
    """文档级元数据模型 - 所有字段都是可选的，支持空字典"""
    title: Optional[str] = Field(None, description="文档标题")
    summary: Optional[str] = Field(None, description="文档摘要")
    keywords: Optional[List[str]] = Field(default_factory=list, description="关键词列表")
    topics: Optional[List[str]] = Field(default_factory=list, description="主题列表")
    entities: Optional[List[str]] = Field(default_factory=list, description="实体列表")
    language: Optional[str] = Field(None, description="文档语言")
    document_type: Optional[str] = Field(None, description="文档类型")
    creation_date: Optional[str] = Field(None, description="创建日期")
    author: Optional[str] = Field(None, description="作者")
    department: Optional[str] = Field(None, description="部门")
    version: Optional[str] = Field(None, description="版本")
    classification: Optional[str] = Field(None, description="分类级别")
    tags: Optional[List[str]] = Field(default_factory=list, description="标签列表")
    
    class Config:
        extra = "allow"  # 允许额外字段
        # 允许空字典初始化
        validate_assignment = True


class LegalDocumentMetadata(BaseModel):
    """法律文档专用元数据模型 - 包含法律特有字段"""
    
    # 继承通用字段
    document_type: str = Field(
        ...,
        description="文档类型分类，如：法律条文、司法解释、行政法规、部门规章、地方性法规等"
    )
    
    legal_domain: List[str] = Field(
        ...,
        description="法律领域分类，如：民法、刑法、行政法、商法、劳动法、知识产权法等，最多3个"
    )
    
    target_audience: List[str] = Field(
        ...,
        description="文档目标受众，如：律师、法官、企业法务、普通公民、学生等"
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
    
    document_summary: str = Field(
        ...,
        description="文档摘要，简要概括文档的主要内容、目的和关键要点"
    )
    
    # 法律文档特有字段
    legal_provisions: Optional[List[str]] = Field(
        None,
        description="法条编号列表，如：第一条、第二十三条、第一百零八条等，提取文档中的具体条文编号"
    )
    
    legal_effect: Optional[str] = Field(
        None,
        description="法律效力等级，如：法律、行政法规、部门规章、地方性法规、司法解释等"
    )
    
    jurisdiction: Optional[List[str]] = Field(
        None,
        description="管辖范围，如：全国、省级、市级、特定行业、特定地区等"
    )
    
    enforcement_agency: Optional[List[str]] = Field(
        None,
        description="执法机关，如：人民法院、公安机关、市场监管部门、税务部门等"
    )
    
    legal_consequences: Optional[List[str]] = Field(
        None,
        description="法律后果类型，如：民事责任、刑事责任、行政责任、经济处罚等"
    )


class PolicyDocumentMetadata(BaseModel):
    """政策文档专用元数据模型 - 包含政策特有字段"""
    
    # 继承通用字段
    document_type: str = Field(
        ...,
        description="文档类型分类，如：政策解读、实施细则、通知公告、工作指南、行业规范等"
    )
    
    legal_domain: Optional[List[str]] = Field(
        None,
        description="相关法律领域，如：民法、刑法、行政法、商法、劳动法等，最多3个。政策文档可选填写"
    )
    
    target_audience: List[str] = Field(
        ...,
        description="文档目标受众，如：企业、个人、政府部门、行业协会、中小企业等"
    )
    
    importance_level: str = Field(
        ...,
        description="重要性等级：核心（重大政策）、重要（常规政策）、一般（专项政策）"
    )
    
    applicable_scenarios: List[str] = Field(
        ...,
        description="适用场景列表，如：企业合规、政策申报、资质认定、补贴申请、监管检查等，最多3个"
    )
    
    related_articles: List[str] = Field(
        default_factory=list,
        description="相关政策或法规，提取文中提到的其他政策文件、法律条文等"
    )
    
    document_summary: str = Field(
        ...,
        description="文档摘要，简要概括文档的主要内容、目的和关键要点"
    )
    
    # 政策文档特有字段
    policy_scope: Optional[List[str]] = Field(
        None,
        description="政策适用范围，如：全国、特定省市、特定行业、特定企业类型等，List类型"
    )
    
    implementation_timeline: Optional[str] = Field(
        None,
        description="实施时间安排，如：立即生效、2024年1月1日起实施、过渡期至2024年底等"
    )
    
    policy_benefits: Optional[List[str]] = Field(
        None,
        description="政策优惠或支持措施，如：税收减免、资金补贴、简化流程、优先审批等"
    )
    
    compliance_requirements: Optional[List[str]] = Field(
        None,
        description="合规要求，如：资质要求、申报义务、备案要求、定期报告等"
    )
    
    responsible_department: Optional[List[str]] = Field(
        None,
        description="负责部门，如：工信部、发改委、财政部、地方政府等"
    )


class ChunkLevelMetadata(BaseModel):
    """Chunk级元数据模型 - 所有字段都是可选的，支持空字典"""
    main_topic: Optional[str] = Field(None, description="主要主题")
    key_concepts: Optional[List[str]] = Field(default_factory=list, description="关键概念")
    entities: Optional[List[str]] = Field(default_factory=list, description="实体")
    keywords: Optional[List[str]] = Field(default_factory=list, description="关键词")
    content_type: Optional[str] = Field(None, description="内容类型")
    importance_score: Optional[float] = Field(None, description="重要性评分")
    semantic_density: Optional[float] = Field(None, description="语义密度")
    readability_score: Optional[float] = Field(None, description="可读性评分")
    emotional_tone: Optional[str] = Field(None, description="情感色调")
    complexity_level: Optional[str] = Field(None, description="复杂度级别")
    
    class Config:
        extra = "allow"  # 允许额外字段
        # 允许空字典初始化
        validate_assignment = True
  