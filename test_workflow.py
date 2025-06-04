#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整功能验证脚本
涵盖解析本地文件、建立向量数据库、检索召回测试的完整工作流程
"""

import os
import sys
import time
import json
import shutil
import requests
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('test_workflow.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

class WorkflowTester:
    """完整工作流程测试器"""
    
    def __init__(self, 
                 doc_service_url: str = "http://localhost:8001",
                 rag_service_url: str = "http://localhost:8002",
                 workspace_dir: str = "/home/ubuntu/workspace/know_ledgebase"):
        self.doc_service_url = doc_service_url
        self.rag_service_url = rag_service_url
        self.workspace_dir = Path(workspace_dir)
        self.test_data_dir = self.workspace_dir / "test_data"
        self.output_dir = self.workspace_dir / "test_output"
        self.session = requests.Session()
        
        # 创建测试目录
        self.test_data_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        
        # 测试结果存储
        self.results = {
            "start_time": datetime.now().isoformat(),
            "steps": {},
            "summary": {}
        }
    
    def create_test_documents(self) -> List[str]:
        """创建测试文档"""
        logger.info("创建测试文档...")
        
        test_files = []
        
        # 创建测试文本文件
        test_content = {
            "ai_introduction.txt": """
人工智能简介

人工智能（Artificial Intelligence，AI）是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。

主要领域包括：
1. 机器学习（Machine Learning）
   - 监督学习
   - 无监督学习
   - 强化学习

2. 深度学习（Deep Learning）
   - 神经网络
   - 卷积神经网络（CNN）
   - 循环神经网络（RNN）
   - Transformer架构

3. 自然语言处理（NLP）
   - 文本分析
   - 语言模型
   - 机器翻译
   - 问答系统

4. 计算机视觉
   - 图像识别
   - 目标检测
   - 图像分割

应用场景：
- 智能助手
- 自动驾驶
- 医疗诊断
- 金融风控
- 推荐系统
""",
            "machine_learning.txt": """
机器学习详解

机器学习是人工智能的核心技术之一，通过算法让计算机从数据中学习模式和规律。

主要算法类型：

1. 监督学习算法
   - 线性回归
   - 逻辑回归
   - 决策树
   - 随机森林
   - 支持向量机（SVM）
   - 神经网络

2. 无监督学习算法
   - K-means聚类
   - 层次聚类
   - DBSCAN
   - 主成分分析（PCA）
   - t-SNE

3. 强化学习算法
   - Q-learning
   - Deep Q-Network（DQN）
   - Policy Gradient
   - Actor-Critic

评估指标：
- 准确率（Accuracy）
- 精确率（Precision）
- 召回率（Recall）
- F1分数
- AUC-ROC

模型优化：
- 交叉验证
- 网格搜索
- 正则化
- 特征工程
""",
            "deep_learning.txt": """
深度学习技术

深度学习是机器学习的一个子领域，使用多层神经网络来学习数据的复杂表示。

核心概念：

1. 神经网络基础
   - 感知器
   - 多层感知器（MLP）
   - 激活函数（ReLU、Sigmoid、Tanh）
   - 反向传播算法

2. 卷积神经网络（CNN）
   - 卷积层
   - 池化层
   - 全连接层
   - 经典架构：LeNet、AlexNet、VGG、ResNet

3. 循环神经网络（RNN）
   - 标准RNN
   - 长短期记忆网络（LSTM）
   - 门控循环单元（GRU）
   - 双向RNN

4. Transformer架构
   - 自注意力机制
   - 多头注意力
   - 位置编码
   - BERT、GPT系列模型

训练技巧：
- 批量归一化
- Dropout
- 学习率调度
- 数据增强
- 迁移学习

应用领域：
- 图像分类
- 目标检测
- 语音识别
- 自然语言处理
- 生成对抗网络（GAN）
""",
            "nlp_applications.txt": """
自然语言处理应用

自然语言处理（NLP）是人工智能的重要分支，专注于让计算机理解和生成人类语言。

主要任务：

1. 文本预处理
   - 分词
   - 词性标注
   - 命名实体识别
   - 句法分析

2. 文本分类
   - 情感分析
   - 主题分类
   - 垃圾邮件检测
   - 新闻分类

3. 信息抽取
   - 关键词提取
   - 实体关系抽取
   - 事件抽取
   - 知识图谱构建

4. 文本生成
   - 机器翻译
   - 文本摘要
   - 对话系统
   - 创意写作

5. 问答系统
   - 检索式问答
   - 生成式问答
   - 知识库问答
   - 阅读理解

技术方法：
- 词袋模型（Bag of Words）
- TF-IDF
- Word2Vec
- GloVe
- BERT
- GPT
- T5

评估方法：
- BLEU分数（机器翻译）
- ROUGE分数（文本摘要）
- 困惑度（语言模型）
- 人工评估

实际应用：
- 搜索引擎
- 智能客服
- 内容推荐
- 舆情分析
- 法律文档分析
"""
        }
        
        # 写入测试文件
        for filename, content in test_content.items():
            file_path = self.test_data_dir / filename
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            test_files.append(str(file_path))
            logger.info(f"创建测试文件: {file_path}")
        
        return test_files
    
    def wait_for_services(self, max_wait: int = 60) -> bool:
        """等待服务启动"""
        logger.info("等待服务启动...")
        
        start_time = time.time()
        while time.time() - start_time < max_wait:
            try:
                # 检查文档服务
                doc_response = self.session.get(f"{self.doc_service_url}/health", timeout=5)
                # 检查RAG服务
                rag_response = self.session.get(f"{self.rag_service_url}/health", timeout=5)
                if doc_response.status_code == 200 and rag_response.status_code == 200:
                    logger.info("所有服务已启动")
                    return True
                    
            except requests.exceptions.RequestException:
                pass
            
            time.sleep(2)
        
        logger.error("服务启动超时")
        return False
    
    def step1_parse_documents(self, test_files: List[str]) -> Dict[str, Any]:
        """步骤1：解析文档"""
        logger.info("步骤1：开始解析文档...")
        
        step_results = {
            "start_time": time.time(),
            "files_parsed": [],
            "task_ids": [],
            "errors": []
        }
        
        parse_config = {
            "ocr_enabled": True,
            "extract_tables": True,
            "extract_images": True,
            "save_to_file": True
        }
        
        for file_path in test_files:
            try:
                logger.info(f"解析文件: {file_path}")
                
                payload = {
                    "file_path": file_path,
                    "config": parse_config
                }
                
                response = self.session.post(
                    f"{self.doc_service_url}/parse/file",
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    task_id = result.get("task_id")
                    step_results["task_ids"].append(task_id)
                    step_results["files_parsed"].append(file_path)
                    logger.info(f"文件解析任务创建成功: {task_id}")
                else:
                    error_msg = f"解析文件失败 {file_path}: {response.status_code} - {response.text}"
                    step_results["errors"].append(error_msg)
                    logger.error(error_msg)
                    
            except Exception as e:
                error_msg = f"解析文件异常 {file_path}: {str(e)}"
                step_results["errors"].append(error_msg)
                logger.error(error_msg)
        
        # 等待解析任务完成
        logger.info("等待文档解析完成...")
        completed_tasks = self._wait_for_parse_tasks(step_results["task_ids"])
        step_results["completed_tasks"] = completed_tasks
        
        step_results["end_time"] = time.time()
        step_results["duration"] = step_results["end_time"] - step_results["start_time"]
        step_results["success"] = len(step_results["errors"]) == 0 and len(completed_tasks) > 0
        
        logger.info(f"步骤1完成，解析了 {len(step_results['files_parsed'])} 个文件")
        return step_results
    
    def step2_build_vector_store(self) -> Dict[str, Any]:
        """步骤2：构建向量数据库"""
        logger.info("步骤2：开始构建向量数据库...")
        
        step_results = {
            "start_time": time.time(),
            "task_id": None,
            "errors": []
        }
        
        vector_config = {
            "extract_keywords": True,
            "extract_summary": True,
            "generate_qa": True,
            "chunk_size": 512,
            "chunk_overlap": 50
        }
        
        try:
            payload = {
                "directory_path": str(self.test_data_dir),
                "config": vector_config
            }
            
            response = self.session.post(
                f"{self.doc_service_url}/vector-store/build",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                task_id = result.get("task_id")
                step_results["task_id"] = task_id
                logger.info(f"向量数据库构建任务创建成功: {task_id}")
                
                # 等待构建完成
                logger.info("等待向量数据库构建完成...")
                task_status = self._wait_for_vector_task(task_id)
                step_results["task_status"] = task_status
                
            else:
                error_msg = f"构建向量数据库失败: {response.status_code} - {response.text}"
                step_results["errors"].append(error_msg)
                logger.error(error_msg)
                
        except Exception as e:
            error_msg = f"构建向量数据库异常: {str(e)}"
            step_results["errors"].append(error_msg)
            logger.error(error_msg)
        
        step_results["end_time"] = time.time()
        step_results["duration"] = step_results["end_time"] - step_results["start_time"]
        step_results["success"] = len(step_results["errors"]) == 0 and step_results.get("task_status", {}).get("status") == "completed"
        
        logger.info(f"步骤2完成，向量数据库构建{'成功' if step_results['success'] else '失败'}")
        return step_results
    
    def step3_test_retrieval(self) -> Dict[str, Any]:
        """步骤3：测试检索功能"""
        logger.info("步骤3：开始测试检索功能...")
        
        step_results = {
            "start_time": time.time(),
            "index_id": os.path.basename(self.test_data_dir),
            "retrieval_tests": [],
            "errors": []
        }
        
        # 测试查询列表
        test_queries = [
            "什么是人工智能？",
            "机器学习的主要算法有哪些？",
            "深度学习和传统机器学习的区别",
            "自然语言处理的应用场景",
            "神经网络的基本结构",
            "如何评估机器学习模型？"
        ]
        
        try:
            # 首先加载索引
            logger.info(f"加载索引: {step_results['index_id']}")
            load_response = self.session.post(
                f"{self.rag_service_url}/index/load",
                params={"index_id": step_results["index_id"]},
                timeout=30
            )
            
            if load_response.status_code != 200:
                error_msg = f"加载索引失败: {load_response.status_code} - {load_response.text}"
                step_results["errors"].append(error_msg)
                logger.error(error_msg)
                return step_results
            
            logger.info("索引加载成功")
            
            # 测试每个查询
            for query in test_queries:
                logger.info(f"测试查询: {query}")
                
                retrieval_result = {
                    "query": query,
                    "start_time": time.time()
                }
                
                try:
                    payload = {
                        "index_id": step_results["index_id"],
                        "query": query,
                        "top_k": 5
                    }
                    
                    response = self.session.post(
                        f"{self.rag_service_url}/retrieve",
                        json=payload,
                        timeout=30
                    )
                    
                    retrieval_result["end_time"] = time.time()
                    retrieval_result["duration"] = retrieval_result["end_time"] - retrieval_result["start_time"]
                    
                    if response.status_code == 200:
                        result = response.json()
                        retrieval_result["success"] = True
                        retrieval_result["results_count"] = result.get("count", 0)
                        retrieval_result["results"] = result.get("results", [])
                        
                        # 计算平均相似度分数
                        scores = [r.get("score", 0) for r in retrieval_result["results"]]
                        retrieval_result["avg_score"] = sum(scores) / len(scores) if scores else 0
                        
                        logger.info(f"检索成功，返回 {retrieval_result['results_count']} 个结果")
                    else:
                        retrieval_result["success"] = False
                        retrieval_result["error"] = f"{response.status_code} - {response.text}"
                        logger.error(f"检索失败: {retrieval_result['error']}")
                        
                except Exception as e:
                    retrieval_result["end_time"] = time.time()
                    retrieval_result["duration"] = retrieval_result["end_time"] - retrieval_result["start_time"]
                    retrieval_result["success"] = False
                    retrieval_result["error"] = str(e)
                    logger.error(f"检索异常: {str(e)}")
                
                step_results["retrieval_tests"].append(retrieval_result)
                
        except Exception as e:
            error_msg = f"检索测试异常: {str(e)}"
            step_results["errors"].append(error_msg)
            logger.error(error_msg)
        
        # 计算统计信息
        successful_tests = [t for t in step_results["retrieval_tests"] if t.get("success")]
        step_results["success_rate"] = len(successful_tests) / len(step_results["retrieval_tests"]) if step_results["retrieval_tests"] else 0
        step_results["avg_response_time"] = sum(t.get("duration", 0) for t in successful_tests) / len(successful_tests) if successful_tests else 0
        step_results["avg_results_count"] = sum(t.get("results_count", 0) for t in successful_tests) / len(successful_tests) if successful_tests else 0
        
        step_results["end_time"] = time.time()
        step_results["duration"] = step_results["end_time"] - step_results["start_time"]
        step_results["success"] = len(step_results["errors"]) == 0 and step_results["success_rate"] > 0.8
        
        logger.info(f"步骤3完成，成功率: {step_results['success_rate']:.2%}")
        return step_results
    
    def step4_test_chat(self) -> Dict[str, Any]:
        """步骤4：测试对话功能"""
        logger.info("步骤4：开始测试对话功能...")
        
        step_results = {
            "start_time": time.time(),
            "session_id": f"test_session_{int(time.time())}",
            "chat_tests": [],
            "errors": []
        }
        
        try:
            # 创建聊天会话
            logger.info("创建聊天会话...")
            session_payload = {
                "index_id": os.path.basename(self.test_data_dir),
                "session_id": step_results["session_id"]
            }
            
            session_response = self.session.post(
                f"{self.rag_service_url}/chat/session",
                json=session_payload,
                timeout=30
            )
            
            if session_response.status_code != 200:
                error_msg = f"创建会话失败: {session_response.status_code} - {session_response.text}"
                step_results["errors"].append(error_msg)
                logger.error(error_msg)
                return step_results
            
            logger.info(f"会话创建成功: {step_results['session_id']}")
            
            # 测试对话
            test_messages = [
                "你好，请介绍一下人工智能",
                "机器学习有哪些主要的算法类型？",
                "深度学习相比传统机器学习有什么优势？",
                "自然语言处理可以应用在哪些场景？"
            ]
            
            for message in test_messages:
                logger.info(f"发送消息: {message}")
                
                chat_result = {
                    "message": message,
                    "start_time": time.time()
                }
                
                try:
                    chat_payload = {
                        "session_id": step_results["session_id"],
                        "message": message
                    }
                    
                    response = self.session.post(
                        f"{self.rag_service_url}/chat",
                        json=chat_payload,
                        timeout=60
                    )
                    
                    chat_result["end_time"] = time.time()
                    chat_result["duration"] = chat_result["end_time"] - chat_result["start_time"]
                    
                    if response.status_code == 200:
                        result = response.json()
                        chat_result["success"] = True
                        chat_result["response"] = result.get("response", "")
                        chat_result["response_length"] = len(chat_result["response"])
                        
                        logger.info(f"对话成功，响应长度: {chat_result['response_length']}")
                    else:
                        chat_result["success"] = False
                        chat_result["error"] = f"{response.status_code} - {response.text}"
                        logger.error(f"对话失败: {chat_result['error']}")
                        
                except Exception as e:
                    chat_result["end_time"] = time.time()
                    chat_result["duration"] = chat_result["end_time"] - chat_result["start_time"]
                    chat_result["success"] = False
                    chat_result["error"] = str(e)
                    logger.error(f"对话异常: {str(e)}")
                
                step_results["chat_tests"].append(chat_result)
                time.sleep(1)  # 避免请求过快
                
        except Exception as e:
            error_msg = f"对话测试异常: {str(e)}"
            step_results["errors"].append(error_msg)
            logger.error(error_msg)
        
        # 计算统计信息
        successful_chats = [c for c in step_results["chat_tests"] if c.get("success")]
        step_results["success_rate"] = len(successful_chats) / len(step_results["chat_tests"]) if step_results["chat_tests"] else 0
        step_results["avg_response_time"] = sum(c.get("duration", 0) for c in successful_chats) / len(successful_chats) if successful_chats else 0
        step_results["avg_response_length"] = sum(c.get("response_length", 0) for c in successful_chats) / len(successful_chats) if successful_chats else 0
        
        step_results["end_time"] = time.time()
        step_results["duration"] = step_results["end_time"] - step_results["start_time"]
        step_results["success"] = len(step_results["errors"]) == 0 and step_results["success_rate"] > 0.5
        
        logger.info(f"步骤4完成，成功率: {step_results['success_rate']:.2%}")
        return step_results
    
    def _wait_for_parse_tasks(self, task_ids: List[str], max_wait: int = 300) -> List[Dict]:
        """等待解析任务完成"""
        completed_tasks = []
        start_time = time.time()
        
        while task_ids and (time.time() - start_time) < max_wait:
            remaining_tasks = []
            
            for task_id in task_ids:
                try:
                    response = self.session.get(
                        f"{self.doc_service_url}/parse/status/{task_id}",
                        timeout=10
                    )
                    
                    if response.status_code == 200:
                        task_status = response.json()
                        status = task_status.get("status")
                        
                        if status == "completed":
                            completed_tasks.append(task_status)
                            logger.info(f"解析任务完成: {task_id}")
                        elif status == "failed":
                            logger.error(f"解析任务失败: {task_id} - {task_status.get('error')}")
                        else:
                            remaining_tasks.append(task_id)
                    else:
                        remaining_tasks.append(task_id)
                        
                except Exception as e:
                    logger.error(f"检查任务状态异常 {task_id}: {str(e)}")
                    remaining_tasks.append(task_id)
            
            task_ids = remaining_tasks
            if task_ids:
                time.sleep(5)
        
        return completed_tasks
    
    def _wait_for_vector_task(self, task_id: str, max_wait: int = 600) -> Dict:
        """等待向量数据库构建任务完成"""
        start_time = time.time()
        
        while (time.time() - start_time) < max_wait:
            try:
                response = self.session.get(
                    f"{self.doc_service_url}/vector-store/status/{task_id}",
                    timeout=10
                )
                
                if response.status_code == 200:
                    task_status = response.json()
                    status = task_status.get("status")
                    progress = task_status.get("progress", 0)
                    
                    logger.info(f"向量数据库构建进度: {progress}%")
                    
                    if status in ["completed", "failed"]:
                        return task_status
                
                time.sleep(10)
                
            except Exception as e:
                logger.error(f"检查向量任务状态异常: {str(e)}")
                time.sleep(10)
        
        return {"status": "timeout", "error": "Task timeout"}
    
    def run_complete_workflow(self) -> Dict[str, Any]:
        """运行完整工作流程"""
        logger.info("开始运行完整工作流程测试...")
        
        # 等待服务启动
        if not self.wait_for_services():
            self.results["error"] = "服务启动失败"
            return self.results
        
        try:
            # 创建测试文档
            test_files = self.create_test_documents()
            self.results["test_files"] = test_files
            
            # 步骤1：解析文档
            self.results["steps"]["step1_parse_documents"] = self.step1_parse_documents(test_files)
            
            # 步骤2：构建向量数据库
            self.results["steps"]["step2_build_vector_store"] = self.step2_build_vector_store()
            
            # 步骤3：测试检索功能
            self.results["steps"]["step3_test_retrieval"] = self.step3_test_retrieval()
            
            # 步骤4：测试对话功能
            self.results["steps"]["step4_test_chat"] = self.step4_test_chat()
            
            # 生成总结
            self._generate_summary()
            
        except Exception as e:
            logger.error(f"工作流程执行异常: {str(e)}")
            self.results["error"] = str(e)
        
        self.results["end_time"] = datetime.now().isoformat()
        logger.info("完整工作流程测试完成")
        
        return self.results
    
    def _generate_summary(self):
        """生成测试总结"""
        steps = self.results["steps"]
        
        self.results["summary"] = {
            "total_steps": len(steps),
            "successful_steps": sum(1 for step in steps.values() if step.get("success")),
            "failed_steps": sum(1 for step in steps.values() if not step.get("success")),
            "overall_success": all(step.get("success") for step in steps.values()),
            "total_duration": sum(step.get("duration", 0) for step in steps.values()),
            "step_details": {
                name: {
                    "success": step.get("success"),
                    "duration": step.get("duration"),
                    "error_count": len(step.get("errors", []))
                }
                for name, step in steps.items()
            }
        }
    
    def print_results(self):
        """打印测试结果"""
        print("\n" + "="*80)
        print("完整工作流程测试结果")
        print("="*80)
        
        summary = self.results.get("summary", {})
        print(f"\n总体结果: {'✅ 成功' if summary.get('overall_success') else '❌ 失败'}")
        print(f"总耗时: {summary.get('total_duration', 0):.2f} 秒")
        print(f"成功步骤: {summary.get('successful_steps', 0)}/{summary.get('total_steps', 0)}")
        
        print("\n步骤详情:")
        print("-" * 60)
        
        for step_name, step_data in self.results.get("steps", {}).items():
            status = "✅ 成功" if step_data.get("success") else "❌ 失败"
            duration = step_data.get("duration", 0)
            error_count = len(step_data.get("errors", []))
            
            print(f"{step_name}: {status} (耗时: {duration:.2f}s, 错误: {error_count})")
            
            # 显示特定步骤的详细信息
            if "retrieval" in step_name:
                success_rate = step_data.get("success_rate", 0)
                avg_time = step_data.get("avg_response_time", 0)
                print(f"  检索成功率: {success_rate:.2%}, 平均响应时间: {avg_time:.2f}s")
            
            elif "chat" in step_name:
                success_rate = step_data.get("success_rate", 0)
                avg_time = step_data.get("avg_response_time", 0)
                print(f"  对话成功率: {success_rate:.2%}, 平均响应时间: {avg_time:.2f}s")
            
            # 显示错误信息
            if step_data.get("errors"):
                for error in step_data["errors"][:3]:  # 只显示前3个错误
                    print(f"  错误: {error}")
        
        print("\n" + "="*80)
    
    def save_results(self, output_file: str = None):
        """保存测试结果"""
        if output_file is None:
            output_file = self.output_dir / f"workflow_test_results_{int(time.time())}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"测试结果已保存到: {output_file}")
        return output_file


def main():
    """主函数"""
    print("LlamaIndex RAG知识库系统 - 完整功能验证")
    print("="*50)
    
    # 创建测试器
    tester = WorkflowTester()
    
    # 运行完整工作流程
    results = tester.run_complete_workflow()
    
    # 打印结果
    tester.print_results()
    
    # 保存结果
    output_file = tester.save_results()
    
    print(f"\n详细测试结果已保存到: {output_file}")
    print("\n测试完成！")
    
    # 返回总体成功状态
    return results.get("summary", {}).get("overall_success", False)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)