#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
接口测试用例
测试文档解析服务和RAG检索服务的所有API接口
"""

import requests
import json
import time
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class APITestCase:
    """API测试用例基类"""
    
    def __init__(self, doc_service_url: str = "http://localhost:8000", 
                 rag_service_url: str = "http://localhost:8001"):
        self.doc_service_url = doc_service_url
        self.rag_service_url = rag_service_url
        self.session = requests.Session()
        
    def test_service_health(self) -> Dict[str, Any]:
        """测试服务健康状态"""
        results = {}
        
        # 测试文档解析服务
        try:
            response = self.session.get(f"{self.doc_service_url}/health")
            results["document_service"] = {
                "status_code": response.status_code,
                "response": response.json() if response.status_code == 200 else response.text,
                "success": response.status_code == 200
            }
        except Exception as e:
            results["document_service"] = {
                "status_code": None,
                "response": str(e),
                "success": False
            }
        
        # 测试RAG服务
        try:
            response = self.session.get(f"{self.rag_service_url}/health")
            results["rag_service"] = {
                "status_code": response.status_code,
                "response": response.json() if response.status_code == 200 else response.text,
                "success": response.status_code == 200
            }
        except Exception as e:
            results["rag_service"] = {
                "status_code": None,
                "response": str(e),
                "success": False
            }
        
        return results
    
    def test_document_parsing(self, file_path: str, config: Optional[Dict] = None) -> Dict[str, Any]:
        """测试文档解析接口"""
        results = {}
        
        # 测试文件路径解析
        try:
            payload = {"file_path": file_path}
            if config:
                payload["config"] = config
            
            response = self.session.post(
                f"{self.doc_service_url}/parse/file",
                json=payload
            )
            
            results["parse_file"] = {
                "status_code": response.status_code,
                "response": response.json() if response.status_code == 200 else response.text,
                "success": response.status_code == 200
            }
            
            # 如果解析成功，获取任务ID并检查状态
            if response.status_code == 200:
                task_id = response.json().get("task_id")
                if task_id:
                    results["task_status"] = self._check_task_status(task_id, "parse")
                    
        except Exception as e:
            results["parse_file"] = {
                "status_code": None,
                "response": str(e),
                "success": False
            }
        
        # 测试文件上传解析
        if os.path.exists(file_path):
            try:
                with open(file_path, 'rb') as f:
                    files = {'file': (os.path.basename(file_path), f, 'application/octet-stream')}
                    data = {'save_to_file': 'true'}
                    if config:
                        data['config'] = json.dumps(config)
                    
                    response = self.session.post(
                        f"{self.doc_service_url}/parse/upload",
                        files=files,
                        data=data
                    )
                    
                    results["parse_upload"] = {
                        "status_code": response.status_code,
                        "response": response.json() if response.status_code == 200 else response.text,
                        "success": response.status_code == 200
                    }
                    
            except Exception as e:
                results["parse_upload"] = {
                    "status_code": None,
                    "response": str(e),
                    "success": False
                }
        
        return results
    
    def test_vector_store_building(self, directory_path: str, config: Optional[Dict] = None) -> Dict[str, Any]:
        """测试向量数据库构建接口"""
        results = {}
        
        try:
            payload = {"directory_path": directory_path}
            if config:
                payload["config"] = config
            
            response = self.session.post(
                f"{self.doc_service_url}/vector-store/build",
                json=payload
            )
            
            results["build_vector_store"] = {
                "status_code": response.status_code,
                "response": response.json() if response.status_code == 200 else response.text,
                "success": response.status_code == 200
            }
            
            # 如果构建成功，获取任务ID并检查状态
            if response.status_code == 200:
                task_id = response.json().get("task_id")
                if task_id:
                    results["task_status"] = self._check_task_status(task_id, "vector-store")
                    
        except Exception as e:
            results["build_vector_store"] = {
                "status_code": None,
                "response": str(e),
                "success": False
            }
        
        return results
    
    def test_rag_operations(self, index_id: str, query: str = "测试查询") -> Dict[str, Any]:
        """测试RAG相关操作"""
        results = {}
        
        # 测试加载索引
        try:
            response = self.session.post(
                f"{self.rag_service_url}/index/load",
                params={"index_id": index_id}
            )
            
            results["load_index"] = {
                "status_code": response.status_code,
                "response": response.json() if response.status_code == 200 else response.text,
                "success": response.status_code == 200
            }
            
        except Exception as e:
            results["load_index"] = {
                "status_code": None,
                "response": str(e),
                "success": False
            }
        
        # 测试列出已加载索引
        try:
            response = self.session.get(f"{self.rag_service_url}/index/list")
            
            results["list_indexes"] = {
                "status_code": response.status_code,
                "response": response.json() if response.status_code == 200 else response.text,
                "success": response.status_code == 200
            }
            
        except Exception as e:
            results["list_indexes"] = {
                "status_code": None,
                "response": str(e),
                "success": False
            }
        
        # 测试混合检索
        try:
            payload = {
                "index_id": index_id,
                "query": query,
                "top_k": 5
            }
            
            response = self.session.post(
                f"{self.rag_service_url}/retrieve",
                json=payload
            )
            
            results["hybrid_retrieve"] = {
                "status_code": response.status_code,
                "response": response.json() if response.status_code == 200 else response.text,
                "success": response.status_code == 200
            }
            
        except Exception as e:
            results["hybrid_retrieve"] = {
                "status_code": None,
                "response": str(e),
                "success": False
            }
        
        # 测试创建聊天会话
        try:
            payload = {
                "index_id": index_id,
                "session_id": f"test_session_{int(time.time())}"
            }
            
            response = self.session.post(
                f"{self.rag_service_url}/chat/session",
                json=payload
            )
            
            results["create_session"] = {
                "status_code": response.status_code,
                "response": response.json() if response.status_code == 200 else response.text,
                "success": response.status_code == 200
            }
            
            # 如果会话创建成功，测试聊天
            if response.status_code == 200:
                session_id = response.json().get("session_id")
                if session_id:
                    results["chat"] = self._test_chat(session_id, query)
                    
        except Exception as e:
            results["create_session"] = {
                "status_code": None,
                "response": str(e),
                "success": False
            }
        
        return results
    
    def test_recall_performance(self, index_id: str, test_queries: list) -> Dict[str, Any]:
        """测试召回性能"""
        results = {}
        
        try:
            payload = {
                "index_id": index_id,
                "test_queries": test_queries
            }
            
            response = self.session.post(
                f"{self.rag_service_url}/test/recall",
                json=payload
            )
            
            results["recall_test"] = {
                "status_code": response.status_code,
                "response": response.json() if response.status_code == 200 else response.text,
                "success": response.status_code == 200
            }
            
        except Exception as e:
            results["recall_test"] = {
                "status_code": None,
                "response": str(e),
                "success": False
            }
        
        return results
    
    def test_configuration_apis(self) -> Dict[str, Any]:
        """测试配置相关API"""
        results = {}
        
        # 测试文档服务配置
        try:
            response = self.session.get(f"{self.doc_service_url}/config")
            
            results["doc_service_config"] = {
                "status_code": response.status_code,
                "response": response.json() if response.status_code == 200 else response.text,
                "success": response.status_code == 200
            }
            
        except Exception as e:
            results["doc_service_config"] = {
                "status_code": None,
                "response": str(e),
                "success": False
            }
        
        # 测试RAG服务配置
        try:
            response = self.session.get(f"{self.rag_service_url}/config")
            
            results["rag_service_config"] = {
                "status_code": response.status_code,
                "response": response.json() if response.status_code == 200 else response.text,
                "success": response.status_code == 200
            }
            
        except Exception as e:
            results["rag_service_config"] = {
                "status_code": None,
                "response": str(e),
                "success": False
            }
        
        return results
    
    def _check_task_status(self, task_id: str, task_type: str) -> Dict[str, Any]:
        """检查任务状态"""
        try:
            if task_type == "parse":
                url = f"{self.doc_service_url}/parse/status/{task_id}"
            else:
                url = f"{self.doc_service_url}/vector-store/status/{task_id}"
            
            # 轮询任务状态
            max_attempts = 10
            for attempt in range(max_attempts):
                response = self.session.get(url)
                
                if response.status_code == 200:
                    status_data = response.json()
                    if status_data.get("status") in ["completed", "failed"]:
                        return {
                            "status_code": response.status_code,
                            "response": status_data,
                            "success": status_data.get("status") == "completed"
                        }
                
                time.sleep(2)  # 等待2秒后重试
            
            return {
                "status_code": response.status_code,
                "response": "Task timeout",
                "success": False
            }
            
        except Exception as e:
            return {
                "status_code": None,
                "response": str(e),
                "success": False
            }
    
    def _test_chat(self, session_id: str, message: str) -> Dict[str, Any]:
        """测试聊天功能"""
        try:
            payload = {
                "session_id": session_id,
                "message": message
            }
            
            response = self.session.post(
                f"{self.rag_service_url}/chat",
                json=payload
            )
            
            return {
                "status_code": response.status_code,
                "response": response.json() if response.status_code == 200 else response.text,
                "success": response.status_code == 200
            }
            
        except Exception as e:
            return {
                "status_code": None,
                "response": str(e),
                "success": False
            }
    
    def run_comprehensive_test(self, test_file_path: str, test_dir_path: str) -> Dict[str, Any]:
        """运行综合测试"""
        logger.info("开始运行综合API测试...")
        
        all_results = {}
        
        # 1. 测试服务健康状态
        logger.info("测试服务健康状态...")
        all_results["health_check"] = self.test_service_health()
        
        # 2. 测试配置API
        logger.info("测试配置API...")
        all_results["configuration"] = self.test_configuration_apis()
        
        # 3. 测试文档解析
        if os.path.exists(test_file_path):
            logger.info(f"测试文档解析: {test_file_path}")
            parse_config = {
                "ocr_enabled": True,
                "extract_tables": True,
                "extract_images": True,
                "save_to_file": True
            }
            all_results["document_parsing"] = self.test_document_parsing(test_file_path, parse_config)
        
        # 4. 测试向量数据库构建
        if os.path.exists(test_dir_path):
            logger.info(f"测试向量数据库构建: {test_dir_path}")
            vector_config = {
                "extract_keywords": True,
                "extract_summary": True,
                "generate_qa": True,
                "chunk_size": 512,
                "chunk_overlap": 50
            }
            all_results["vector_store_building"] = self.test_vector_store_building(test_dir_path, vector_config)
        
        # 5. 测试RAG操作（需要等待向量数据库构建完成）
        logger.info("等待向量数据库构建完成...")
        time.sleep(10)  # 等待一段时间
        
        index_id = os.path.basename(test_dir_path)
        logger.info(f"测试RAG操作: {index_id}")
        all_results["rag_operations"] = self.test_rag_operations(index_id, "什么是人工智能？")
        
        # 6. 测试召回性能
        test_queries = [
            {"query": "人工智能的定义", "expected_docs": []},
            {"query": "机器学习算法", "expected_docs": []},
            {"query": "深度学习应用", "expected_docs": []}
        ]
        logger.info("测试召回性能...")
        all_results["recall_performance"] = self.test_recall_performance(index_id, test_queries)
        
        logger.info("综合API测试完成")
        return all_results
    
    def print_test_results(self, results: Dict[str, Any]):
        """打印测试结果"""
        print("\n" + "="*80)
        print("API测试结果报告")
        print("="*80)
        
        for category, tests in results.items():
            print(f"\n【{category.upper()}】")
            print("-" * 40)
            
            if isinstance(tests, dict):
                for test_name, result in tests.items():
                    status = "✅ 成功" if result.get("success") else "❌ 失败"
                    status_code = result.get("status_code", "N/A")
                    print(f"{test_name}: {status} (状态码: {status_code})")
                    
                    if not result.get("success") and result.get("response"):
                        print(f"  错误信息: {result['response']}")
        
        print("\n" + "="*80)


def main():
    """主函数 - 运行测试用例"""
    # 配置测试参数
    test_file_path = "/home/ubuntu/workspace/know_ledgebase/test_data/sample.pdf"  # 测试文件路径
    test_dir_path = "/home/ubuntu/workspace/know_ledgebase/test_data"  # 测试目录路径
    
    # 创建测试实例
    tester = APITestCase()
    
    # 运行综合测试
    results = tester.run_comprehensive_test(test_file_path, test_dir_path)
    
    # 打印结果
    tester.print_test_results(results)
    
    # 保存结果到文件
    output_file = "/home/ubuntu/workspace/know_ledgebase/test_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n详细测试结果已保存到: {output_file}")


if __name__ == "__main__":
    main()