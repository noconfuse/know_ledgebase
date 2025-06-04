#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速功能测试脚本
用于快速验证系统的基本功能
"""

import os
import sys
import time
import json
import requests
from pathlib import Path
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QuickTester:
    """快速测试器"""
    
    def __init__(self):
        self.doc_service_url = "http://localhost:8000"
        self.rag_service_url = "http://localhost:8001"
        self.workspace_dir = Path("/home/ubuntu/workspace/know_ledgebase")
        self.test_dir = self.workspace_dir / "quick_test_data"
        self.session = requests.Session()
        
        # 创建测试目录
        self.test_dir.mkdir(exist_ok=True)
    
    def create_simple_test_file(self) -> str:
        """创建简单的测试文件"""
        test_content = """
人工智能基础知识

人工智能（AI）是计算机科学的一个分支，旨在创建能够执行通常需要人类智能的任务的系统。

主要技术包括：
1. 机器学习 - 让计算机从数据中学习
2. 深度学习 - 使用神经网络进行复杂模式识别
3. 自然语言处理 - 理解和生成人类语言
4. 计算机视觉 - 分析和理解图像

应用领域：
- 智能助手和聊天机器人
- 自动驾驶汽车
- 医疗诊断系统
- 推荐系统
- 语音识别

机器学习算法类型：
- 监督学习：使用标记数据进行训练
- 无监督学习：从未标记数据中发现模式
- 强化学习：通过试错学习最优策略
"""
        
        test_file = self.test_dir / "ai_basics.txt"
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(test_content)
        
        logger.info(f"创建测试文件: {test_file}")
        return str(test_file)
    
    def check_services(self) -> bool:
        """检查服务状态"""
        logger.info("检查服务状态...")
        
        try:
            # 检查文档服务
            doc_response = self.session.get(f"{self.doc_service_url}/health", timeout=5)
            doc_ok = doc_response.status_code == 200
            
            # 检查RAG服务
            rag_response = self.session.get(f"{self.rag_service_url}/health", timeout=5)
            rag_ok = rag_response.status_code == 200
            
            logger.info(f"文档服务: {'✅' if doc_ok else '❌'}")
            logger.info(f"RAG服务: {'✅' if rag_ok else '❌'}")
            
            return doc_ok and rag_ok
            
        except Exception as e:
            logger.error(f"服务检查失败: {e}")
            return False
    
    def test_document_parsing(self, file_path: str) -> bool:
        """测试文档解析"""
        logger.info("测试文档解析...")
        
        try:
            payload = {
                "file_path": file_path,
                "config": {
                    "save_to_file": True
                }
            }
            
            response = self.session.post(
                f"{self.doc_service_url}/parse/file",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                task_id = response.json().get("task_id")
                logger.info(f"解析任务创建成功: {task_id}")
                
                # 等待任务完成
                return self._wait_for_task(task_id, "parse")
            else:
                logger.error(f"解析任务创建失败: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"文档解析测试失败: {e}")
            return False
    
    def test_vector_store_building(self) -> bool:
        """测试向量数据库构建"""
        logger.info("测试向量数据库构建...")
        
        try:
            payload = {
                "directory_path": str(self.test_dir),
                "config": {
                    "chunk_size": 256,
                    "chunk_overlap": 20
                }
            }
            
            response = self.session.post(
                f"{self.doc_service_url}/vector-store/build",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                task_id = response.json().get("task_id")
                logger.info(f"向量数据库构建任务创建成功: {task_id}")
                
                # 等待任务完成
                return self._wait_for_task(task_id, "vector-store")
            else:
                logger.error(f"向量数据库构建失败: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"向量数据库构建测试失败: {e}")
            return False
    
    def test_retrieval(self) -> bool:
        """测试检索功能"""
        logger.info("测试检索功能...")
        
        index_id = self.test_dir.name
        
        try:
            # 加载索引
            load_response = self.session.post(
                f"{self.rag_service_url}/index/load",
                params={"index_id": index_id},
                timeout=30
            )
            
            if load_response.status_code != 200:
                logger.error(f"索引加载失败: {load_response.status_code} - {load_response.text}")
                return False
            
            logger.info("索引加载成功")
            
            # 测试检索
            payload = {
                "index_id": index_id,
                "query": "什么是机器学习？",
                "top_k": 3
            }
            
            response = self.session.post(
                f"{self.rag_service_url}/retrieve",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                results_count = result.get("count", 0)
                logger.info(f"检索成功，返回 {results_count} 个结果")
                
                # 显示检索结果
                for i, res in enumerate(result.get("results", [])[:2]):
                    content = res.get("content", "")[:100] + "..."
                    score = res.get("score", 0)
                    logger.info(f"结果 {i+1}: 分数={score:.3f}, 内容={content}")
                
                return results_count > 0
            else:
                logger.error(f"检索失败: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"检索测试失败: {e}")
            return False
    
    def test_chat(self) -> bool:
        """测试对话功能"""
        logger.info("测试对话功能...")
        
        index_id = self.test_dir.name
        session_id = f"quick_test_{int(time.time())}"
        
        try:
            # 创建会话
            session_payload = {
                "index_id": index_id,
                "session_id": session_id
            }
            
            session_response = self.session.post(
                f"{self.rag_service_url}/chat/session",
                json=session_payload,
                timeout=30
            )
            
            if session_response.status_code != 200:
                logger.error(f"会话创建失败: {session_response.status_code} - {session_response.text}")
                return False
            
            logger.info(f"会话创建成功: {session_id}")
            
            # 发送消息
            chat_payload = {
                "session_id": session_id,
                "message": "请简单介绍一下人工智能"
            }
            
            response = self.session.post(
                f"{self.rag_service_url}/chat",
                json=chat_payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get("response", "")
                logger.info(f"对话成功，响应长度: {len(response_text)}")
                logger.info(f"响应预览: {response_text[:200]}...")
                return len(response_text) > 0
            else:
                logger.error(f"对话失败: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"对话测试失败: {e}")
            return False
    
    def _wait_for_task(self, task_id: str, task_type: str, max_wait: int = 120) -> bool:
        """等待任务完成"""
        start_time = time.time()
        
        while (time.time() - start_time) < max_wait:
            try:
                if task_type == "parse":
                    url = f"{self.doc_service_url}/parse/status/{task_id}"
                else:
                    url = f"{self.doc_service_url}/vector-store/status/{task_id}"
                
                response = self.session.get(url, timeout=10)
                
                if response.status_code == 200:
                    task_status = response.json()
                    status = task_status.get("status")
                    progress = task_status.get("progress", 0)
                    
                    if status == "completed":
                        logger.info(f"任务完成: {task_id}")
                        return True
                    elif status == "failed":
                        error = task_status.get("error", "未知错误")
                        logger.error(f"任务失败: {task_id} - {error}")
                        return False
                    else:
                        logger.info(f"任务进行中: {progress}%")
                
                time.sleep(5)
                
            except Exception as e:
                logger.error(f"检查任务状态失败: {e}")
                time.sleep(5)
        
        logger.error(f"任务超时: {task_id}")
        return False
    
    def run_quick_test(self) -> bool:
        """运行快速测试"""
        print("\n" + "="*50)
        print("LlamaIndex RAG知识库系统 - 快速功能测试")
        print("="*50)
        
        test_results = []
        
        # 1. 检查服务状态
        print("\n1. 检查服务状态")
        service_ok = self.check_services()
        test_results.append(("服务状态检查", service_ok))
        
        if not service_ok:
            print("❌ 服务未启动，请先启动服务")
            return False
        
        # 2. 创建测试文件
        print("\n2. 创建测试文件")
        test_file = self.create_simple_test_file()
        test_results.append(("测试文件创建", True))
        
        # 3. 测试文档解析
        print("\n3. 测试文档解析")
        parse_ok = self.test_document_parsing(test_file)
        test_results.append(("文档解析", parse_ok))
        
        # 4. 测试向量数据库构建
        print("\n4. 测试向量数据库构建")
        vector_ok = self.test_vector_store_building()
        test_results.append(("向量数据库构建", vector_ok))
        
        # 5. 测试检索功能
        print("\n5. 测试检索功能")
        retrieval_ok = self.test_retrieval()
        test_results.append(("检索功能", retrieval_ok))
        
        # 6. 测试对话功能
        print("\n6. 测试对话功能")
        chat_ok = self.test_chat()
        test_results.append(("对话功能", chat_ok))
        
        # 显示测试结果
        print("\n" + "="*50)
        print("测试结果汇总")
        print("="*50)
        
        success_count = 0
        for test_name, success in test_results:
            status = "✅ 成功" if success else "❌ 失败"
            print(f"{test_name}: {status}")
            if success:
                success_count += 1
        
        overall_success = success_count == len(test_results)
        print(f"\n总体结果: {'✅ 全部成功' if overall_success else f'❌ {success_count}/{len(test_results)} 成功'}")
        
        if overall_success:
            print("\n🎉 恭喜！所有功能测试通过，系统运行正常！")
        else:
            print("\n⚠️  部分功能测试失败，请检查日志和配置")
        
        return overall_success


def main():
    """主函数"""
    tester = QuickTester()
    success = tester.run_quick_test()
    
    print("\n测试完成！")
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)