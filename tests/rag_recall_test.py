#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG召回准确率测试脚本
基于消防法文档测试RAG系统的召回准确率
"""

import requests
import json
import time
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple
import logging
from dataclasses import dataclass
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class TestQuery:
    """测试查询数据结构"""
    query: str
    expected_keywords: List[str]  # 期望在召回结果中出现的关键词
    expected_articles: List[str]  # 期望召回的具体条文编号
    category: str  # 问题类别
    difficulty: str  # 难度等级: easy, medium, hard

@dataclass
class RecallResult:
    """召回结果数据结构"""
    query: str
    retrieved_docs: List[Dict]
    recall_score: float
    precision_score: float
    response_time: float
    category: str
    difficulty: str

class RAGRecallTester:
    """RAG召回准确率测试器"""
    
    def __init__(self, rag_service_url: str = "http://localhost:8001"):
        self.rag_service_url = rag_service_url
        self.session = requests.Session()
        self.test_queries = self._generate_test_queries()
        
    def _generate_test_queries(self) -> List[TestQuery]:
        """生成测试查询集合"""
        queries = [
            # 基础法条查询 - 简单
            TestQuery(
                query="消防法的立法目的是什么？",
                expected_keywords=["预防火灾", "减少火灾危害", "应急救援", "保护人身", "财产安全"],
                expected_articles=["第一条"],
                category="基础法条",
                difficulty="easy"
            ),
            TestQuery(
                query="消防工作的方针是什么？",
                expected_keywords=["预防为主", "防消结合", "政府统一领导", "部门依法监管"],
                expected_articles=["第二条"],
                category="基础法条",
                difficulty="easy"
            ),
            
            # 具体职责查询 - 中等
            TestQuery(
                query="单位应当履行哪些消防安全职责？",
                expected_keywords=["消防安全责任制", "消防设施", "防火检查", "消防演练", "应急疏散预案"],
                expected_articles=["第十六条"],
                category="职责规定",
                difficulty="medium"
            ),
            TestQuery(
                query="消防安全重点单位有哪些特殊职责？",
                expected_keywords=["消防安全管理人", "消防档案", "防火巡查", "岗前培训"],
                expected_articles=["第十七条"],
                category="职责规定",
                difficulty="medium"
            ),
            
            # 建设工程相关 - 中等
            TestQuery(
                query="建设工程消防设计审查制度是怎样的？",
                expected_keywords=["消防设计审查", "特殊建设工程", "住房和城乡建设", "施工许可证"],
                expected_articles=["第十条", "第十一条"],
                category="建设工程",
                difficulty="medium"
            ),
            TestQuery(
                query="公众聚集场所投入使用前需要什么手续？",
                expected_keywords=["消防安全检查", "告知承诺", "消防救援机构", "许可"],
                expected_articles=["第十五条"],
                category="建设工程",
                difficulty="medium"
            ),
            
            # 法律责任查询 - 困难
            TestQuery(
                query="违反消防设计审查规定会受到什么处罚？",
                expected_keywords=["停止施工", "三万元以上三十万元以下", "住房和城乡建设主管部门"],
                expected_articles=["第五十八条"],
                category="法律责任",
                difficulty="hard"
            ),
            TestQuery(
                query="损坏消防设施会面临什么法律后果？",
                expected_keywords=["五千元以上五万元以下", "责令改正", "强制执行"],
                expected_articles=["第六十条"],
                category="法律责任",
                difficulty="hard"
            ),
            
            # 特殊场所规定 - 中等到困难
            TestQuery(
                query="易燃易爆危险品场所有什么特殊要求？",
                expected_keywords=["不得与居住场所", "同一建筑物", "安全距离", "消防技术标准"],
                expected_articles=["第十九条", "第二十二条"],
                category="特殊场所",
                difficulty="medium"
            ),
            TestQuery(
                query="电焊气焊作业人员有什么要求？",
                expected_keywords=["持证上岗", "消防安全操作规程", "火灾危险作业"],
                expected_articles=["第二十一条"],
                category="特殊场所",
                difficulty="medium"
            ),
            
            # 消防组织相关 - 中等
            TestQuery(
                query="国家综合性消防救援队承担什么工作？",
                expected_keywords=["火灾扑救", "重大灾害事故", "应急救援", "抢救人员生命"],
                expected_articles=["第三十六条", "第三十七条"],
                category="消防组织",
                difficulty="medium"
            ),
            
            # 复合查询 - 困难
            TestQuery(
                query="消防产品的质量监督和认证是如何规定的？",
                expected_keywords=["国家标准", "强制性产品认证", "技术鉴定", "质量监督部门"],
                expected_articles=["第二十四条", "第二十五条"],
                category="产品质量",
                difficulty="hard"
            ),
            TestQuery(
                query="农村消防工作有哪些特殊规定？",
                expected_keywords=["公共消防设施建设", "消防安全责任", "防火期间", "消防宣传教育"],
                expected_articles=["第三十条", "第三十一条", "第三十二条"],
                category="农村消防",
                difficulty="hard"
            )
        ]
        
        return queries
    
    def retrieve_documents(self, index_id: str, query: str, top_k: int = 10) -> Tuple[List[Dict], float]:
        """执行文档检索"""
        start_time = time.time()
        
        try:
            payload = {
                "index_id": index_id,
                "query": query,
                "top_k": top_k
            }
            
            response = self.session.post(
                f"{self.rag_service_url}/retrieve",
                json=payload
            )
            
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                return result.get("documents", []), response_time
            else:
                logger.error(f"检索失败: {response.status_code} - {response.text}")
                return [], response_time
                
        except Exception as e:
            response_time = time.time() - start_time
            logger.error(f"检索异常: {str(e)}")
            return [], response_time
    
    def calculate_recall_score(self, retrieved_docs: List[Dict], expected_keywords: List[str], 
                             expected_articles: List[str]) -> Tuple[float, float]:
        """计算召回率和精确率"""
        if not retrieved_docs:
            return 0.0, 0.0
        
        # 提取检索到的文档内容
        retrieved_content = ""
        for doc in retrieved_docs:
            if isinstance(doc, dict):
                content = doc.get("content", "") or doc.get("text", "")
                retrieved_content += content + " "
        
        retrieved_content = retrieved_content.lower()
        
        # 计算关键词召回率
        keyword_hits = 0
        for keyword in expected_keywords:
            if keyword.lower() in retrieved_content:
                keyword_hits += 1
        
        keyword_recall = keyword_hits / len(expected_keywords) if expected_keywords else 0
        
        # 计算条文召回率
        article_hits = 0
        for article in expected_articles:
            if article in retrieved_content:
                article_hits += 1
        
        article_recall = article_hits / len(expected_articles) if expected_articles else 0
        
        # 综合召回率（关键词权重0.6，条文权重0.4）
        recall_score = keyword_recall * 0.6 + article_recall * 0.4
        
        # 简单的精确率计算（基于相关内容比例）
        relevant_docs = 0
        for doc in retrieved_docs:
            doc_content = ""
            if isinstance(doc, dict):
                doc_content = (doc.get("content", "") or doc.get("text", "")).lower()
            
            # 如果文档包含任何期望的关键词，认为是相关的
            is_relevant = any(keyword.lower() in doc_content for keyword in expected_keywords)
            if is_relevant:
                relevant_docs += 1
        
        precision_score = relevant_docs / len(retrieved_docs) if retrieved_docs else 0
        
        return recall_score, precision_score
    
    def run_single_test(self, index_id: str, test_query: TestQuery) -> RecallResult:
        """运行单个测试查询"""
        logger.info(f"测试查询: {test_query.query}")
        
        # 执行检索
        retrieved_docs, response_time = self.retrieve_documents(index_id, test_query.query)
        
        # 计算召回率和精确率
        recall_score, precision_score = self.calculate_recall_score(
            retrieved_docs, test_query.expected_keywords, test_query.expected_articles
        )
        
        result = RecallResult(
            query=test_query.query,
            retrieved_docs=retrieved_docs,
            recall_score=recall_score,
            precision_score=precision_score,
            response_time=response_time,
            category=test_query.category,
            difficulty=test_query.difficulty
        )
        
        logger.info(f"召回率: {recall_score:.3f}, 精确率: {precision_score:.3f}, 响应时间: {response_time:.3f}s")
        
        return result
    
    def run_all_tests(self, index_id: str) -> List[RecallResult]:
        """运行所有测试"""
        logger.info(f"开始RAG召回准确率测试，共{len(self.test_queries)}个查询")
        
        results = []
        for i, test_query in enumerate(self.test_queries, 1):
            logger.info(f"\n=== 测试 {i}/{len(self.test_queries)} ===")
            result = self.run_single_test(index_id, test_query)
            results.append(result)
            
            # 避免请求过于频繁
            time.sleep(0.5)
        
        return results
    
    def generate_report(self, results: List[RecallResult]) -> Dict[str, Any]:
        """生成测试报告"""
        if not results:
            return {"error": "没有测试结果"}
        
        # 总体统计
        total_recall = sum(r.recall_score for r in results) / len(results)
        total_precision = sum(r.precision_score for r in results) / len(results)
        avg_response_time = sum(r.response_time for r in results) / len(results)
        
        # 按类别统计
        category_stats = {}
        for result in results:
            if result.category not in category_stats:
                category_stats[result.category] = []
            category_stats[result.category].append(result)
        
        category_summary = {}
        for category, cat_results in category_stats.items():
            category_summary[category] = {
                "count": len(cat_results),
                "avg_recall": sum(r.recall_score for r in cat_results) / len(cat_results),
                "avg_precision": sum(r.precision_score for r in cat_results) / len(cat_results),
                "avg_response_time": sum(r.response_time for r in cat_results) / len(cat_results)
            }
        
        # 按难度统计
        difficulty_stats = {}
        for result in results:
            if result.difficulty not in difficulty_stats:
                difficulty_stats[result.difficulty] = []
            difficulty_stats[result.difficulty].append(result)
        
        difficulty_summary = {}
        for difficulty, diff_results in difficulty_stats.items():
            difficulty_summary[difficulty] = {
                "count": len(diff_results),
                "avg_recall": sum(r.recall_score for r in diff_results) / len(diff_results),
                "avg_precision": sum(r.precision_score for r in diff_results) / len(diff_results),
                "avg_response_time": sum(r.response_time for r in diff_results) / len(diff_results)
            }
        
        # 详细结果
        detailed_results = []
        for result in results:
            detailed_results.append({
                "query": result.query,
                "category": result.category,
                "difficulty": result.difficulty,
                "recall_score": round(result.recall_score, 3),
                "precision_score": round(result.precision_score, 3),
                "response_time": round(result.response_time, 3),
                "retrieved_count": len(result.retrieved_docs)
            })
        
        report = {
            "test_summary": {
                "total_queries": len(results),
                "avg_recall_score": round(total_recall, 3),
                "avg_precision_score": round(total_precision, 3),
                "avg_response_time": round(avg_response_time, 3),
                "test_time": datetime.now().isoformat()
            },
            "category_analysis": category_summary,
            "difficulty_analysis": difficulty_summary,
            "detailed_results": detailed_results
        }
        
        return report
    
    def save_report(self, report: Dict[str, Any], output_file: str):
        """保存测试报告"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info(f"测试报告已保存到: {output_file}")
    
    def print_summary(self, report: Dict[str, Any]):
        """打印测试摘要"""
        print("\n" + "="*80)
        print("RAG召回准确率测试报告")
        print("="*80)
        
        summary = report["test_summary"]
        print(f"\n总体结果:")
        print(f"  测试查询数量: {summary['total_queries']}")
        print(f"  平均召回率: {summary['avg_recall_score']:.3f}")
        print(f"  平均精确率: {summary['avg_precision_score']:.3f}")
        print(f"  平均响应时间: {summary['avg_response_time']:.3f}s")
        
        print(f"\n按类别分析:")
        for category, stats in report["category_analysis"].items():
            print(f"  {category}:")
            print(f"    查询数量: {stats['count']}")
            print(f"    平均召回率: {stats['avg_recall']:.3f}")
            print(f"    平均精确率: {stats['avg_precision']:.3f}")
        
        print(f"\n按难度分析:")
        for difficulty, stats in report["difficulty_analysis"].items():
            print(f"  {difficulty}:")
            print(f"    查询数量: {stats['count']}")
            print(f"    平均召回率: {stats['avg_recall']:.3f}")
            print(f"    平均精确率: {stats['avg_precision']:.3f}")
        
        print("\n" + "="*80)

def main():
    """主函数"""
    # 配置参数
    RAG_SERVICE_URL = "http://localhost:8001"
    INDEX_ID = "1b70e012-79b7-4b20-8f70-9e94646e3aad"  # 消防法文档的索引ID
    OUTPUT_FILE = "/home/ubuntu/workspace/know_ledgebase/tests/rag_recall_test_report.json"
    
    # 创建测试器
    tester = RAGRecallTester(RAG_SERVICE_URL)
    
    try:
        # 运行测试
        results = tester.run_all_tests(INDEX_ID)
        
        # 生成报告
        report = tester.generate_report(results)
        
        # 保存报告
        tester.save_report(report, OUTPUT_FILE)
        
        # 打印摘要
        tester.print_summary(report)
        
    except Exception as e:
        logger.error(f"测试执行失败: {str(e)}")
        raise

if __name__ == "__main__":
    main()