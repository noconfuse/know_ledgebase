#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复后的聊天测试脚本
用于测试修复上下文窗口问题后的RAG服务聊天功能
"""

import asyncio
import logging
from services.rag_service import RAGService

# 设置日志级别
logging.basicConfig(level=logging.INFO)

async def main():
    # 初始化RAG服务
    rag_service = RAGService()
    
    # 使用已有的索引ID
    index_id = "7ebebfb2-78f7-4426-992e-2ca149924ba5"  # 请替换为实际的索引ID
    
    # 创建会话
    session_id = await rag_service.create_chat_session([index_id])
    print(f"创建会话成功: {session_id}")
    
    # 测试消息列表
    test_messages = [
        "你好，请介绍一下自己",
        "这个知识库包含什么内容？",
        "建筑消防设施检测的要求有哪些？",  # 这是之前出错的问题
        "消防法规定了哪些消防安全责任？"
    ]
    
    # 逐个发送消息并获取回复
    for i, message in enumerate(test_messages, 1):
        print(f"\n--- 测试 {i}/{len(test_messages)} ---")
        print(f"用户: {message}")
        
        try:
            # 发送消息并获取回复
            response = await rag_service.chat(session_id, message)
            
            # 打印回复
            print(f"助手: {response['response']}")
            
            # 打印参考来源数量
            source_count = len(response.get('source_nodes', []))
            print(f"参考来源数量: {source_count}")
            
            # 打印前两个参考来源
            if source_count > 0:
                print("参考来源:")
                for j, node in enumerate(response['source_nodes'][:2], 1):
                    score = node.get('score', 0)
                    metadata = node.get('metadata', {})
                    file_path = metadata.get('file_path', '未知文件')
                    print(f"  {j}. [{score:.3f}] {file_path}")
            
        except Exception as e:
            print(f"错误: {e}")
    
    print("\n测试完成!")

if __name__ == "__main__":
    asyncio.run(main())