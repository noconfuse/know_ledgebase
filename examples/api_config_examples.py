#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
向量化API配置示例
展示如何使用新的智能元数据提取器配置参数
"""

import requests
import json
from typing import Dict, Any

# API基础URL
API_BASE_URL = "http://localhost:8000"


def create_vector_store_config(
    # 基础配置
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    
    # 智能元数据提取器配置
    extract_mode: str = "enhanced",
    min_chunk_size_for_summary: int = 500,
    min_chunk_size_for_qa: int = 300,
    max_keywords: int = 5,
    num_questions: int = 3,
    
    # 索引描述
    index_description: str = None
) -> Dict[str, Any]:
    """
    创建向量存储配置
    
    Args:
        chunk_size: 文本块大小
        chunk_overlap: 文本块重叠
        extract_mode: 提取模式（basic或enhanced）
        min_chunk_size_for_summary: 提取摘要的最小chunk大小（仅enhanced模式）
        min_chunk_size_for_qa: 提取问答对的最小chunk大小
        max_keywords: 最大关键词数量
        num_questions: 问答对数量
        index_description: 索引描述信息
    
    Returns:
        配置字典
    """
    config = {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "extract_mode": extract_mode,
        "min_chunk_size_for_summary": min_chunk_size_for_summary,
        "min_chunk_size_for_qa": min_chunk_size_for_qa,
        "max_keywords": max_keywords,
        "num_questions": num_questions
    }
    
    if index_description:
        config["index_description"] = index_description
    
    return config


def build_vector_store_api(directory_path: str, config: Dict[str, Any]) -> str:
    """
    调用向量存储构建API
    
    Args:
        directory_path: 文档目录路径
        config: 配置参数
    
    Returns:
        任务ID
    """
    url = f"{API_BASE_URL}/vector-store/build"
    
    payload = {
        "directory_path": directory_path,
        "config": config
    }
    
    response = requests.post(url, json=payload)
    response.raise_for_status()
    
    result = response.json()
    return result["task_id"]


def get_task_status(task_id: str) -> Dict[str, Any]:
    """
    获取任务状态
    
    Args:
        task_id: 任务ID
    
    Returns:
        任务状态信息
    """
    url = f"{API_BASE_URL}/vector-store/status/{task_id}"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


# 配置示例

# 1. 高性能配置（快速处理，基础元数据）
HIGH_PERFORMANCE_CONFIG = create_vector_store_config(
    extract_mode="basic",
    use_enhanced_extractor=False,
    min_chunk_size_for_summary=1000,  # 更高的阈值，减少LLM调用
    min_chunk_size_for_qa=800,
    max_keywords=3,
    num_questions=2,
    chunk_size=256,  # 较小的chunk，处理更快
    chunk_overlap=25
)

# 2. 平衡配置（默认推荐）
BALANCED_CONFIG = create_vector_store_config(
    extract_mode="enhanced",
    use_enhanced_extractor=True,
    min_chunk_size_for_summary=500,
    min_chunk_size_for_qa=300,
    max_keywords=5,
    num_questions=3,
    chunk_size=512,
    chunk_overlap=50
)

# 3. 详细配置（高质量元数据，较慢）
DETAILED_CONFIG = create_vector_store_config(
    extract_mode="enhanced",
    use_enhanced_extractor=True,
    min_chunk_size_for_summary=200,  # 更低的阈值，提取更多元数据
    min_chunk_size_for_qa=150,
    max_keywords=8,
    num_questions=5,
    chunk_size=1024,  # 更大的chunk，包含更多上下文
    chunk_overlap=100
)

# 4. 成本优化配置（最少LLM调用）
COST_OPTIMIZED_CONFIG = create_vector_store_config(
    extract_mode="basic",
    use_enhanced_extractor=False,
    min_chunk_size_for_summary=2000,  # 很高的阈值
    min_chunk_size_for_qa=1500,
    max_keywords=3,
    num_questions=1,
    chunk_size=512,
    chunk_overlap=50
)

# 5. 法律文档专用配置
LEGAL_DOCUMENT_CONFIG = create_vector_store_config(
    extract_mode="enhanced",
    use_enhanced_extractor=True,
    min_chunk_size_for_summary=400,
    min_chunk_size_for_qa=250,
    max_keywords=6,
    num_questions=4,
    chunk_size=768,  # 适合法律条文的长度
    chunk_overlap=75,
    index_description="法律文档知识库，包含法律条文、司法解释和案例分析"
)


def example_usage():
    """
    使用示例
    """
    # 示例1：使用平衡配置构建向量存储
    directory_path = "/path/to/documents"
    
    try:
        # 构建向量存储
        task_id = build_vector_store_api(directory_path, BALANCED_CONFIG)
        print(f"任务已创建，ID: {task_id}")
        
        # 检查状态
        status = get_task_status(task_id)
        print(f"任务状态: {status['status']}")
        print(f"进度: {status['progress']}%")
        
    except requests.exceptions.RequestException as e:
        print(f"API调用失败: {e}")
    
    # 示例2：自定义配置
    custom_config = create_vector_store_config(
        extract_mode="enhanced",
        min_chunk_size_for_summary=600,
        max_keywords=4,
        chunk_size=400,
        index_description="企业合规文档库"
    )
    
    print("自定义配置:")
    print(json.dumps(custom_config, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    print("向量化API配置示例")
    print("=" * 50)
    
    print("\n1. 高性能配置:")
    print(json.dumps(HIGH_PERFORMANCE_CONFIG, indent=2, ensure_ascii=False))
    
    print("\n2. 平衡配置:")
    print(json.dumps(BALANCED_CONFIG, indent=2, ensure_ascii=False))
    
    print("\n3. 详细配置:")
    print(json.dumps(DETAILED_CONFIG, indent=2, ensure_ascii=False))
    
    print("\n4. 成本优化配置:")
    print(json.dumps(COST_OPTIMIZED_CONFIG, indent=2, ensure_ascii=False))
    
    print("\n5. 法律文档专用配置:")
    print(json.dumps(LEGAL_DOCUMENT_CONFIG, indent=2, ensure_ascii=False))
    
    # example_usage()  # 取消注释以运行示例