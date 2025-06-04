#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版RAG召回准确率测试脚本
支持多种评估指标和详细的分析报告
"""

import json
import time
import requests
import statistics
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import re
from test_data_generator import TestDataGenerator, TestCase

@dataclass
class RetrievalResult:
    """检索结果数据结构"""
    query: str
    documents: List[Dict[str, Any]]
    response_time: float
    total_docs: int
    error: str = None

@dataclass
class EvaluationMetrics:
    """评估指标数据结构"""
    keyword_recall: float
    keyword_precision: float
    article_recall: float
    article_precision: float
    semantic_relevance: float
    response_time: float
    success_rate: float

@dataclass
class TestResult:
    """测试结果数据结构"""
    test_case: TestCase
    retrieval_result: RetrievalResult
    metrics: EvaluationMetrics
    detailed_analysis: Dict[str, Any]

class EnhancedRAGTester:
    """增强版RAG测试器"""
    
    def __init__(self, base_url: str = "http://localhost:8001", index_id: str = "771f3739-7448-488a-89d3-d18a99b16408"):
        self.base_url = base_url
        self.index_id = index_id
        self.test_generator = TestDataGenerator()
        self.test_results: List[TestResult] = []
        
    def check_service_status(self) -> bool:
        """检查RAG服务状态"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except Exception as e:
            print(f"服务连接失败: {e}")
            return False
    
    def retrieve_documents(self, query: str, top_k: int = 10) -> RetrievalResult:
        """检索文档"""
        start_time = time.time()
        
        try:
            payload = {
                "query": query,
                "index_id": self.index_id,
                "top_k": top_k
            }
            
            response = requests.post(
                f"{self.base_url}/retrieve",
                json=payload,
                timeout=30
            )
            
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                documents = data.get("results", [])  # 修改：使用 'results' 而不是 'documents'
                
                return RetrievalResult(
                    query=query,
                    documents=documents,
                    response_time=response_time,
                    total_docs=len(documents)
                )
            else:
                return RetrievalResult(
                    query=query,
                    documents=[],
                    response_time=response_time,
                    total_docs=0,
                    error=f"HTTP {response.status_code}: {response.text}"
                )
                
        except Exception as e:
            response_time = time.time() - start_time
            return RetrievalResult(
                query=query,
                documents=[],
                response_time=response_time,
                total_docs=0,
                error=str(e)
            )
    
    def extract_article_numbers(self, text: str) -> List[str]:
        """从文本中提取条文编号"""
        # 匹配各种条文格式
        patterns = [
            r'第[一二三四五六七八九十百千万\d]+条',
            r'第\d+条',
            r'Article\s+\d+',
            r'条文\s*[\d一二三四五六七八九十百千万]+'
        ]
        
        articles = set()
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            articles.update(matches)
        
        return list(articles)
    
    def calculate_keyword_metrics(self, expected_keywords: List[str], retrieved_docs: List[Dict]) -> Tuple[float, float]:
        """计算关键词召回率和精确率"""
        if not expected_keywords:
            return 0.0, 0.0
        
        # 合并所有检索到的文档内容
        all_content = ""
        for doc in retrieved_docs:
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})
            title = metadata.get("title", "")
            all_content += f"{title} {content} "
        
        # 计算关键词匹配
        found_keywords = []
        for keyword in expected_keywords:
            if keyword.lower() in all_content.lower():
                found_keywords.append(keyword)
        
        recall = len(found_keywords) / len(expected_keywords) if expected_keywords else 0.0
        
        # 精确率基于找到的关键词与总关键词的比例
        precision = len(found_keywords) / max(len(expected_keywords), 1)
        
        return recall, precision
    
    def calculate_article_metrics(self, expected_articles: List[str], retrieved_docs: List[Dict]) -> Tuple[float, float]:
        """计算条文召回率和精确率"""
        if not expected_articles:
            return 0.0, 0.0
        
        # 从检索结果中提取所有条文编号
        found_articles = set()
        for doc in retrieved_docs:
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})
            title = metadata.get("title", "")
            
            # 从内容和标题中提取条文编号
            articles_in_content = self.extract_article_numbers(content)
            articles_in_title = self.extract_article_numbers(title)
            
            found_articles.update(articles_in_content)
            found_articles.update(articles_in_title)
        
        # 计算匹配的条文
        expected_set = set(expected_articles)
        matched_articles = expected_set.intersection(found_articles)
        
        recall = len(matched_articles) / len(expected_set) if expected_set else 0.0
        precision = len(matched_articles) / max(len(found_articles), 1) if found_articles else 0.0
        
        return recall, precision
    
    def calculate_semantic_relevance(self, query: str, retrieved_docs: List[Dict]) -> float:
        """计算语义相关性（简化版本）"""
        if not retrieved_docs:
            return 0.0
        
        # 简化的语义相关性计算
        # 基于查询词在文档中的出现频率和位置
        query_terms = set(query.lower().split())
        
        relevance_scores = []
        for doc in retrieved_docs:
            content = doc.get("content", "").lower()
            score = doc.get("score", 0.0)  # 使用检索系统返回的相似度分数
            
            # 如果没有分数，基于词汇重叠计算
            if score == 0.0:
                content_terms = set(content.split())
                overlap = len(query_terms.intersection(content_terms))
                score = overlap / max(len(query_terms), 1)
            
            relevance_scores.append(score)
        
        # 返回平均相关性分数
        return statistics.mean(relevance_scores) if relevance_scores else 0.0
    
    def evaluate_test_case(self, test_case: TestCase, top_k: int = 10) -> TestResult:
        """评估单个测试用例"""
        # 执行检索
        retrieval_result = self.retrieve_documents(test_case.query, top_k)
        
        if retrieval_result.error:
            # 如果检索失败，返回零分
            metrics = EvaluationMetrics(
                keyword_recall=0.0,
                keyword_precision=0.0,
                article_recall=0.0,
                article_precision=0.0,
                semantic_relevance=0.0,
                response_time=retrieval_result.response_time,
                success_rate=0.0
            )
            
            detailed_analysis = {
                "error": retrieval_result.error,
                "found_keywords": [],
                "found_articles": [],
                "expected_keywords": test_case.expected_keywords,
                "expected_articles": test_case.expected_articles
            }
        else:
            # 计算各项指标
            keyword_recall, keyword_precision = self.calculate_keyword_metrics(
                test_case.expected_keywords, retrieval_result.documents
            )
            
            article_recall, article_precision = self.calculate_article_metrics(
                test_case.expected_articles, retrieval_result.documents
            )
            
            semantic_relevance = self.calculate_semantic_relevance(
                test_case.query, retrieval_result.documents
            )
            
            metrics = EvaluationMetrics(
                keyword_recall=keyword_recall,
                keyword_precision=keyword_precision,
                article_recall=article_recall,
                article_precision=article_precision,
                semantic_relevance=semantic_relevance,
                response_time=retrieval_result.response_time,
                success_rate=1.0
            )
            
            # 详细分析
            all_content = " ".join([doc.get("content", "") for doc in retrieval_result.documents])
            found_keywords = [kw for kw in test_case.expected_keywords if kw.lower() in all_content.lower()]
            
            found_articles = set()
            for doc in retrieval_result.documents:
                content = doc.get("content", "")
                articles = self.extract_article_numbers(content)
                found_articles.update(articles)
            
            detailed_analysis = {
                "found_keywords": found_keywords,
                "found_articles": list(found_articles),
                "expected_keywords": test_case.expected_keywords,
                "expected_articles": test_case.expected_articles,
                "top_documents": [
                    {
                        "title": doc.get("metadata", {}).get("title", "未知"),
                        "content_preview": doc.get("content", "")[:200] + "...",
                        "score": doc.get("score", 0.0)
                    }
                    for doc in retrieval_result.documents[:3]
                ]
            }
        
        return TestResult(
            test_case=test_case,
            retrieval_result=retrieval_result,
            metrics=metrics,
            detailed_analysis=detailed_analysis
        )
    
    def run_comprehensive_test(self, test_cases: List[TestCase] = None, top_k: int = 10) -> Dict[str, Any]:
        """运行综合测试"""
        if test_cases is None:
            test_cases = self.test_generator.test_cases
        
        print(f"开始运行 {len(test_cases)} 个测试用例...")
        
        self.test_results = []
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\r进度: {i}/{len(test_cases)} - {test_case.query[:30]}...", end="", flush=True)
            
            result = self.evaluate_test_case(test_case, top_k)
            self.test_results.append(result)
            
            # 短暂延迟避免过载
            time.sleep(0.1)
        
        print("\n测试完成！")
        
        # 生成综合报告
        return self.generate_comprehensive_report()
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """生成综合测试报告"""
        if not self.test_results:
            return {"error": "没有测试结果"}
        
        # 计算总体指标
        all_metrics = [result.metrics for result in self.test_results]
        
        overall_metrics = {
            "keyword_recall": statistics.mean([m.keyword_recall for m in all_metrics]),
            "keyword_precision": statistics.mean([m.keyword_precision for m in all_metrics]),
            "article_recall": statistics.mean([m.article_recall for m in all_metrics]),
            "article_precision": statistics.mean([m.article_precision for m in all_metrics]),
            "semantic_relevance": statistics.mean([m.semantic_relevance for m in all_metrics]),
            "avg_response_time": statistics.mean([m.response_time for m in all_metrics]),
            "success_rate": statistics.mean([m.success_rate for m in all_metrics])
        }
        
        # 按类别分析
        category_analysis = {}
        for result in self.test_results:
            category = result.test_case.category
            if category not in category_analysis:
                category_analysis[category] = []
            category_analysis[category].append(result.metrics)
        
        category_metrics = {}
        for category, metrics_list in category_analysis.items():
            category_metrics[category] = {
                "keyword_recall": statistics.mean([m.keyword_recall for m in metrics_list]),
                "article_recall": statistics.mean([m.article_recall for m in metrics_list]),
                "semantic_relevance": statistics.mean([m.semantic_relevance for m in metrics_list]),
                "count": len(metrics_list)
            }
        
        # 按难度分析
        difficulty_analysis = {}
        for result in self.test_results:
            difficulty = result.test_case.difficulty
            if difficulty not in difficulty_analysis:
                difficulty_analysis[difficulty] = []
            difficulty_analysis[difficulty].append(result.metrics)
        
        difficulty_metrics = {}
        for difficulty, metrics_list in difficulty_analysis.items():
            difficulty_metrics[difficulty] = {
                "keyword_recall": statistics.mean([m.keyword_recall for m in metrics_list]),
                "article_recall": statistics.mean([m.article_recall for m in metrics_list]),
                "semantic_relevance": statistics.mean([m.semantic_relevance for m in metrics_list]),
                "count": len(metrics_list)
            }
        
        # 找出表现最好和最差的测试用例
        sorted_results = sorted(
            self.test_results,
            key=lambda r: (r.metrics.keyword_recall + r.metrics.article_recall) / 2,
            reverse=True
        )
        
        best_cases = sorted_results[:3]
        worst_cases = sorted_results[-3:]
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_test_cases": len(self.test_results),
            "overall_metrics": overall_metrics,
            "category_metrics": category_metrics,
            "difficulty_metrics": difficulty_metrics,
            "best_performing_cases": [
                {
                    "query": result.test_case.query,
                    "category": result.test_case.category,
                    "difficulty": result.test_case.difficulty,
                    "keyword_recall": result.metrics.keyword_recall,
                    "article_recall": result.metrics.article_recall
                }
                for result in best_cases
            ],
            "worst_performing_cases": [
                {
                    "query": result.test_case.query,
                    "category": result.test_case.category,
                    "difficulty": result.test_case.difficulty,
                    "keyword_recall": result.metrics.keyword_recall,
                    "article_recall": result.metrics.article_recall,
                    "error": result.detailed_analysis.get("error")
                }
                for result in worst_cases
            ]
        }
        
        return report
    
    def save_detailed_results(self, output_file: str):
        """保存详细测试结果"""
        detailed_results = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_cases": len(self.test_results),
                "base_url": self.base_url,
                "index_id": self.index_id
            },
            "results": [
                {
                    "test_case": asdict(result.test_case),
                    "metrics": asdict(result.metrics),
                    "detailed_analysis": result.detailed_analysis,
                    "retrieval_info": {
                        "total_docs": result.retrieval_result.total_docs,
                        "response_time": result.retrieval_result.response_time,
                        "error": result.retrieval_result.error
                    }
                }
                for result in self.test_results
            ]
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, ensure_ascii=False, indent=2)
        
        print(f"详细结果已保存到: {output_file}")
    
    def print_summary_report(self, report: Dict[str, Any]):
        """打印摘要报告"""
        print("\n" + "="*60)
        print("RAG召回准确率测试报告")
        print("="*60)
        
        overall = report["overall_metrics"]
        print(f"\n总体指标:")
        print(f"  关键词召回率: {overall['keyword_recall']:.3f}")
        print(f"  关键词精确率: {overall['keyword_precision']:.3f}")
        print(f"  条文召回率: {overall['article_recall']:.3f}")
        print(f"  条文精确率: {overall['article_precision']:.3f}")
        print(f"  语义相关性: {overall['semantic_relevance']:.3f}")
        print(f"  平均响应时间: {overall['avg_response_time']:.3f}秒")
        print(f"  成功率: {overall['success_rate']:.3f}")
        
        print(f"\n按类别分析:")
        for category, metrics in report["category_metrics"].items():
            print(f"  {category}:")
            print(f"    关键词召回率: {metrics['keyword_recall']:.3f}")
            print(f"    条文召回率: {metrics['article_recall']:.3f}")
            print(f"    测试用例数: {metrics['count']}")
        
        print(f"\n按难度分析:")
        for difficulty, metrics in report["difficulty_metrics"].items():
            print(f"  {difficulty}:")
            print(f"    关键词召回率: {metrics['keyword_recall']:.3f}")
            print(f"    条文召回率: {metrics['article_recall']:.3f}")
            print(f"    测试用例数: {metrics['count']}")
        
        print(f"\n表现最佳的查询:")
        for case in report["best_performing_cases"]:
            print(f"  {case['query']} (关键词:{case['keyword_recall']:.3f}, 条文:{case['article_recall']:.3f})")
        
        print(f"\n需要改进的查询:")
        for case in report["worst_performing_cases"]:
            error_info = f" - 错误: {case['error']}" if case['error'] else ""
            print(f"  {case['query']} (关键词:{case['keyword_recall']:.3f}, 条文:{case['article_recall']:.3f}){error_info}")

def main():
    """主函数"""
    # 初始化测试器
    tester = EnhancedRAGTester()
    
    # 检查服务状态
    print("检查RAG服务状态...")
    if not tester.check_service_status():
        print("RAG服务不可用，请确保服务正在运行")
        return
    
    print("RAG服务正常")
    
    # 运行测试
    print("\n开始运行综合测试...")
    report = tester.run_comprehensive_test()
    
    # 打印报告
    tester.print_summary_report(report)
    
    # 保存详细结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    detailed_file = f"/home/ubuntu/workspace/know_ledgebase/tests/test_results_{timestamp}.json"
    tester.save_detailed_results(detailed_file)
    
    # 保存摘要报告
    summary_file = f"/home/ubuntu/workspace/know_ledgebase/tests/test_summary_{timestamp}.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n摘要报告已保存到: {summary_file}")
    print("测试完成！")

if __name__ == "__main__":
    main()