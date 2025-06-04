#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速RAG召回测试脚本
用于快速验证RAG系统的基本召回功能
"""

import requests
import json
import time
from typing import List, Dict

def test_basic_recall(rag_service_url: str = "http://localhost:8001", 
                     index_id: str = "1b70e012-79b7-4b20-8f70-9e94646e3aad"):
    """基础召回测试"""
    
    # 简单测试查询
    test_queries = [
        "消防法的立法目的是什么？",
        "单位应当履行哪些消防安全职责？",
        "违反消防设计审查规定会受到什么处罚？",
        "易燃易爆危险品场所有什么特殊要求？",
        "国家综合性消防救援队承担什么工作？"
    ]
    
    print("开始RAG召回快速测试...")
    print(f"测试索引ID: {index_id}")
    print(f"RAG服务地址: {rag_service_url}")
    print("-" * 60)
    
    session = requests.Session()
    total_success = 0
    total_time = 0
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n测试 {i}: {query}")
        
        start_time = time.time()
        try:
            payload = {
                "index_id": index_id,
                "query": query,
                "top_k": 5
            }
            
            response = session.post(
                f"{rag_service_url}/retrieve",
                json=payload,
                timeout=30
            )
            
            response_time = time.time() - start_time
            total_time += response_time
            
            if response.status_code == 200:
                result = response.json()
                docs = result.get("documents", [])
                
                print(f"✅ 成功 - 召回{len(docs)}个文档，耗时{response_time:.2f}s")
                
                # 显示前2个结果的摘要
                for j, doc in enumerate(docs[:2]):
                    content = doc.get("content", "") or doc.get("text", "")
                    preview = content[:100] + "..." if len(content) > 100 else content
                    score = doc.get("score", "N/A")
                    print(f"  文档{j+1} (相似度:{score}): {preview}")
                
                total_success += 1
            else:
                print(f"❌ 失败 - 状态码:{response.status_code}, 错误:{response.text}")
                
        except Exception as e:
            response_time = time.time() - start_time
            total_time += response_time
            print(f"❌ 异常 - {str(e)}")
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结:")
    print(f"成功率: {total_success}/{len(test_queries)} ({total_success/len(test_queries)*100:.1f}%)")
    print(f"平均响应时间: {total_time/len(test_queries):.2f}s")
    print("=" * 60)
    
    return total_success == len(test_queries)

def check_service_status(rag_service_url: str = "http://localhost:8001"):
    """检查RAG服务状态"""
    try:
        response = requests.get(f"{rag_service_url}/health", timeout=5)
        if response.status_code == 200:
            print(f"✅ RAG服务运行正常: {rag_service_url}")
            return True
        else:
            print(f"❌ RAG服务状态异常: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 无法连接RAG服务: {str(e)}")
        return False

def check_index_status(rag_service_url: str = "http://localhost:8001", 
                      index_id: str = "1b70e012-79b7-4b20-8f70-9e94646e3aad"):
    """检查索引状态"""
    try:
        # 尝试加载索引
        response = requests.post(
            f"{rag_service_url}/index/load",
            params={"index_id": index_id},
            timeout=10
        )
        
        if response.status_code == 200:
            print(f"✅ 索引加载成功: {index_id}")
            return True
        else:
            print(f"❌ 索引加载失败: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ 索引检查异常: {str(e)}")
        return False

def main():
    """主函数"""
    RAG_SERVICE_URL = "http://localhost:8001"
    INDEX_ID = "1b70e012-79b7-4b20-8f70-9e94646e3aad"
    
    print("RAG召回功能快速测试")
    print("=" * 60)
    
    # 1. 检查服务状态
    print("1. 检查RAG服务状态...")
    if not check_service_status(RAG_SERVICE_URL):
        print("请先启动RAG服务")
        return False
    
    # 2. 检查索引状态
    print("\n2. 检查索引状态...")
    if not check_index_status(RAG_SERVICE_URL, INDEX_ID):
        print("请确保索引已正确构建")
        return False
    
    # 3. 执行召回测试
    print("\n3. 执行召回测试...")
    success = test_basic_recall(RAG_SERVICE_URL, INDEX_ID)
    
    if success:
        print("\n🎉 所有测试通过！RAG召回功能正常")
    else:
        print("\n⚠️  部分测试失败，请检查系统配置")
    
    return success

if __name__ == "__main__":
    main()