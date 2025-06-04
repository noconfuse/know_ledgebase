#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速RAG召回测试脚本
用于快速验证RAG服务的基本功能
"""

import json
import requests
import time
from typing import List, Dict

def test_rag_service():
    """测试RAG服务"""
    base_url = "http://localhost:8001"
    index_id = "aa1030ad-c615-4d49-b359-340f0fb3afaf"
    
    # 简单测试查询
    test_queries = [
        "什么是消防设施？",
        "政府在消防工作中的职责是什么？",
        "未经消防设计审查擅自施工会受到什么处罚？",
        "消防产品包括哪些？",
        "公民在消防工作中有什么义务？"
    ]
    
    print("开始快速RAG召回测试...")
    print(f"服务地址: {base_url}")
    print(f"索引ID: {index_id}")
    print("-" * 50)
    
    success_count = 0
    total_time = 0
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n测试 {i}/{len(test_queries)}: {query}")
        
        try:
            start_time = time.time()
            
            payload = {
                "query": query,
                "index_id": index_id,
                "top_k": 5
            }
            
            response = requests.post(
                f"{base_url}/retrieve",
                json=payload,
                timeout=10
            )
            
            response_time = time.time() - start_time
            total_time += response_time
            
            if response.status_code == 200:
                data = response.json()
                documents = data.get("documents", [])
                
                print(f"✓ 成功 - 检索到 {len(documents)} 个文档 (耗时: {response_time:.2f}秒)")
                
                # 显示前2个文档的预览
                for j, doc in enumerate(documents[:2], 1):
                    title = doc.get("metadata", {}).get("title", "未知标题")
                    content = doc.get("content", "")[:100] + "..."
                    score = doc.get("score", 0.0)
                    print(f"  文档{j}: {title} (相似度: {score:.3f})")
                    print(f"    内容: {content}")
                
                success_count += 1
            else:
                print(f"✗ 失败 - HTTP {response.status_code}: {response.text}")
                
        except Exception as e:
            print(f"✗ 错误 - {str(e)}")
    
    print("\n" + "="*50)
    print("测试总结:")
    print(f"成功率: {success_count}/{len(test_queries)} ({success_count/len(test_queries)*100:.1f}%)")
    if success_count > 0:
        print(f"平均响应时间: {total_time/success_count:.2f}秒")
    print("="*50)
    
    return success_count == len(test_queries)

if __name__ == "__main__":
    success = test_rag_service()
    if success:
        print("\n🎉 所有测试通过！RAG服务运行正常。")
        print("现在可以运行完整的召回准确率测试：")
        print("python3 enhanced_recall_test.py")
    else:
        print("\n⚠️  部分测试失败，请检查RAG服务状态。")