# -*- coding: utf-8 -*-
"""
高级检索优化器
实现对话接口的增强检索能力，包括：
1. 查询扩展和重写
2. 多路召回融合
3. 动态权重调整
4. 上下文感知检索
5. 对话历史增强
"""

import logging
import asyncio
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.llms.llm import LLM
from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
from services.filtered_retriever import FilteredRetriever
from config import settings

logger = logging.getLogger(__name__)


class QueryExpander:
    """查询扩展器"""
    
    def __init__(self, llm: LLM):
        self.llm = llm
        self.expansion_cache = {}
    
    async def expand_query(self, query: str, context: Optional[str] = None) -> List[str]:
        """扩展查询，生成相关的查询变体"""
        cache_key = f"{query}_{hash(context) if context else 'no_context'}"
        
        if cache_key in self.expansion_cache:
            return self.expansion_cache[cache_key]
        
        context_prompt = f"\n上下文信息：{context}" if context else ""
        
        prompt = f"""
请为以下查询生成3-5个相关的查询变体，用于提高检索召回率。
变体应该：
1. 使用不同的表达方式
2. 包含同义词和相关术语
3. 考虑不同的问法角度
4. 保持原始查询的核心意图
{context_prompt}

原始查询：{query}

请以JSON格式返回查询变体列表：
{{"expanded_queries": ["变体1", "变体2", "变体3"]}}
"""
        
        try:
            response = await self.llm.acomplete(prompt)
            result = json.loads(response.text.strip())
            expanded_queries = result.get("expanded_queries", [query])
            
            # 确保原始查询在列表中
            if query not in expanded_queries:
                expanded_queries.insert(0, query)
            
            self.expansion_cache[cache_key] = expanded_queries
            logger.info(f"Query expanded: {query} -> {len(expanded_queries)} variants")
            return expanded_queries
            
        except Exception as e:
            logger.error(f"Query expansion failed: {e}")
            return [query]
    
    async def rewrite_query_for_context(self, query: str, chat_history: List[Dict[str, str]]) -> str:
        """基于对话历史重写查询"""
        if not chat_history:
            return query
        
        # 获取最近的对话上下文
        recent_context = "\n".join([
            f"{msg['role']}: {msg['content']}"
            for msg in chat_history[-6:]  # 最近3轮对话
        ])
        
        prompt = f"""
基于以下对话历史，重写用户的最新查询，使其更加完整和明确。
重写后的查询应该：
1. 包含必要的上下文信息
2. 消除代词歧义
3. 补充隐含的信息
4. 保持原始意图

对话历史：
{recent_context}

当前查询：{query}

请直接返回重写后的查询（不要包含其他解释）：
"""
        
        try:
            response = await self.llm.acomplete(prompt)
            rewritten_query = response.text.strip()
            
            if rewritten_query and len(rewritten_query) > len(query) * 0.5:
                logger.info(f"Query rewritten: '{query}' -> '{rewritten_query}'")
                return rewritten_query
            else:
                return query
                
        except Exception as e:
            logger.error(f"Query rewriting failed: {e}")
            return query


class MultiPathRetriever:
    """多路召回检索器"""
    
    def __init__(self, indices: Dict[str, VectorStoreIndex]):
        self.indices = indices
        self.retrieval_paths = {
            "dense_vector": self._dense_vector_retrieve,
            "sparse_bm25": self._sparse_bm25_retrieve,
            "hybrid_fusion": self._hybrid_fusion_retrieve,
            "semantic_similarity": self._semantic_similarity_retrieve
        }
    
    async def _dense_vector_retrieve(self, query: str, top_k: int, index_ids: List[str]) -> List[NodeWithScore]:
        """密集向量检索"""
        all_nodes = []
        
        for index_id in index_ids:
            if index_id in self.indices:
                index = self.indices[index_id]
                retriever = index.as_retriever(similarity_top_k=top_k)
                nodes = await retriever.aretrieve(QueryBundle(query_str=query))
                
                # 添加检索路径标识
                for node in nodes:
                    node.metadata = node.metadata or {}
                    node.metadata["retrieval_path"] = "dense_vector"
                    node.metadata["source_index"] = index_id
                
                all_nodes.extend(nodes)
        
        return all_nodes
    
    async def _sparse_bm25_retrieve(self, query: str, top_k: int, index_ids: List[str]) -> List[NodeWithScore]:
        """稀疏BM25检索"""
        all_nodes = []
        
        for index_id in index_ids:
            if index_id in self.indices:
                index = self.indices[index_id]
                try:
                    if index._storage_context and index._storage_context.docstore:
                        bm25_retriever = BM25Retriever.from_defaults(
                            docstore=index._storage_context.docstore,
                            similarity_top_k=top_k
                        )
                        nodes = await bm25_retriever.aretrieve(QueryBundle(query_str=query))
                        
                        # 添加检索路径标识
                        for node in nodes:
                            node.metadata = node.metadata or {}
                            node.metadata["retrieval_path"] = "sparse_bm25"
                            node.metadata["source_index"] = index_id
                        
                        all_nodes.extend(nodes)
                except Exception as e:
                    logger.warning(f"BM25 retrieval failed for index {index_id}: {e}")
        
        return all_nodes
    
    async def _hybrid_fusion_retrieve(self, query: str, top_k: int, index_ids: List[str]) -> List[NodeWithScore]:
        """混合融合检索"""
        all_retrievers = []
        
        for index_id in index_ids:
            if index_id in self.indices:
                index = self.indices[index_id]
                
                # 向量检索器
                vector_retriever = index.as_retriever(similarity_top_k=top_k)
                all_retrievers.append(vector_retriever)
                
                # BM25检索器
                try:
                    if index._storage_context and index._storage_context.docstore:
                        bm25_retriever = BM25Retriever.from_defaults(
                            docstore=index._storage_context.docstore,
                            similarity_top_k=top_k
                        )
                        all_retrievers.append(bm25_retriever)
                except Exception as e:
                    logger.warning(f"BM25 setup failed for index {index_id}: {e}")
        
        if not all_retrievers:
            return []
        
        try:
            fusion_retriever = QueryFusionRetriever(
                retrievers=all_retrievers,
                similarity_top_k=top_k,
                num_queries=3,
                mode="reciprocal_rerank",
                use_async=True
            )
            
            nodes = await fusion_retriever.aretrieve(QueryBundle(query_str=query))
            
            # 添加检索路径标识
            for node in nodes:
                node.metadata = node.metadata or {}
                node.metadata["retrieval_path"] = "hybrid_fusion"
            
            return nodes
            
        except Exception as e:
            logger.error(f"Hybrid fusion retrieval failed: {e}")
            return []
    
    async def _semantic_similarity_retrieve(self, query: str, top_k: int, index_ids: List[str]) -> List[NodeWithScore]:
        """语义相似度检索（使用更高的相似度阈值）"""
        all_nodes = []
        
        for index_id in index_ids:
            if index_id in self.indices:
                index = self.indices[index_id]
                retriever = index.as_retriever(similarity_top_k=top_k * 2)  # 获取更多候选
                
                # 使用过滤检索器提高质量
                filtered_retriever = FilteredRetriever(
                    base_retriever=retriever,
                    similarity_threshold=0.7  # 更高的阈值
                )
                
                nodes = await filtered_retriever.aretrieve(QueryBundle(query_str=query))
                
                # 添加检索路径标识
                for node in nodes:
                    node.metadata = node.metadata or {}
                    node.metadata["retrieval_path"] = "semantic_similarity"
                    node.metadata["source_index"] = index_id
                
                all_nodes.extend(nodes)
        
        return all_nodes[:top_k]
    
    async def multi_path_retrieve(self, 
                                query: str, 
                                top_k: int, 
                                index_ids: List[str],
                                enabled_paths: List[str] = None) -> Dict[str, List[NodeWithScore]]:
        """执行多路召回检索"""
        if enabled_paths is None:
            enabled_paths = list(self.retrieval_paths.keys())
        
        results = {}
        tasks = []
        
        for path_name in enabled_paths:
            if path_name in self.retrieval_paths:
                task = self.retrieval_paths[path_name](query, top_k, index_ids)
                tasks.append((path_name, task))
        
        # 并发执行所有检索路径
        for path_name, task in tasks:
            try:
                nodes = await task
                results[path_name] = nodes
                logger.info(f"Path '{path_name}' retrieved {len(nodes)} nodes")
            except Exception as e:
                logger.error(f"Path '{path_name}' failed: {e}")
                results[path_name] = []
        
        return results


class DynamicWeightAdjuster:
    """动态权重调整器"""
    
    def __init__(self):
        self.performance_history = {}
        self.base_weights = {
            "dense_vector": 0.4,
            "sparse_bm25": 0.3,
            "hybrid_fusion": 0.2,
            "semantic_similarity": 0.1
        }
    
    def calculate_adaptive_weights(self, 
                                 query_type: str,
                                 retrieval_results: Dict[str, List[NodeWithScore]]) -> Dict[str, float]:
        """基于检索结果质量动态调整权重"""
        weights = self.base_weights.copy()
        
        # 基于结果数量调整权重
        total_results = sum(len(nodes) for nodes in retrieval_results.values())
        if total_results == 0:
            return weights
        
        for path_name, nodes in retrieval_results.items():
            if path_name in weights:
                # 基于结果数量和平均分数调整权重
                if nodes:
                    avg_score = sum(node.score for node in nodes) / len(nodes)
                    result_ratio = len(nodes) / total_results
                    
                    # 权重调整因子
                    quality_factor = min(avg_score * 2, 1.0)  # 基于质量
                    quantity_factor = min(result_ratio * 2, 1.0)  # 基于数量
                    
                    weights[path_name] *= (quality_factor + quantity_factor) / 2
        
        # 归一化权重
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        logger.debug(f"Adaptive weights calculated: {weights}")
        return weights
    
    def record_performance(self, query: str, path_name: str, success_rate: float):
        """记录检索路径性能"""
        if path_name not in self.performance_history:
            self.performance_history[path_name] = []
        
        self.performance_history[path_name].append({
            "query": query,
            "success_rate": success_rate,
            "timestamp": datetime.now()
        })
        
        # 保持历史记录在合理范围内
        if len(self.performance_history[path_name]) > 100:
            self.performance_history[path_name] = self.performance_history[path_name][-100:]


class AdvancedRetrievalOptimizer:
    """高级检索优化器"""
    
    def __init__(self, llm: LLM, indices: Dict[str, VectorStoreIndex]):
        self.llm = llm
        self.indices = indices
        self.query_expander = QueryExpander(llm)
        self.multi_path_retriever = MultiPathRetriever(indices)
        self.weight_adjuster = DynamicWeightAdjuster()
        
        # 检索策略配置
        self.strategies = {
            "conservative": {  # 保守策略：注重精确性
                "enabled_paths": ["semantic_similarity", "dense_vector"],
                "expansion_enabled": False,
                "fusion_weight": 0.3
            },
            "balanced": {  # 平衡策略：精确性和召回率平衡
                "enabled_paths": ["dense_vector", "sparse_bm25", "hybrid_fusion"],
                "expansion_enabled": True,
                "fusion_weight": 0.5
            },
            "aggressive": {  # 激进策略：注重召回率
                "enabled_paths": list(self.multi_path_retriever.retrieval_paths.keys()),
                "expansion_enabled": True,
                "fusion_weight": 0.7
            }
        }
    
    async def optimize_retrieval_for_chat(self,
                                        query: str,
                                        index_ids: List[str],
                                        chat_history: List[Dict[str, str]] = None,
                                        top_k: int = 10,
                                        strategy: str = "balanced") -> List[NodeWithScore]:
        """为对话优化检索"""
        logger.info(f"🚀 Starting optimized retrieval for chat query: {query[:100]}...")
        
        # 1. 基于对话历史重写查询
        optimized_query = query
        if chat_history:
            optimized_query = await self.query_expander.rewrite_query_for_context(query, chat_history)
        
        # 2. 查询扩展
        queries_to_search = [optimized_query]
        strategy_config = self.strategies.get(strategy, self.strategies["balanced"])
        
        if strategy_config["expansion_enabled"]:
            context = self._extract_context_from_history(chat_history) if chat_history else None
            expanded_queries = await self.query_expander.expand_query(optimized_query, context)
            queries_to_search.extend(expanded_queries[1:])  # 排除原始查询
        
        logger.info(f"📝 Using {len(queries_to_search)} queries for retrieval")
        
        # 3. 多路召回检索
        all_retrieval_results = {}
        
        for i, search_query in enumerate(queries_to_search):
            logger.debug(f"🔍 Processing query {i+1}/{len(queries_to_search)}: {search_query[:50]}...")
            
            path_results = await self.multi_path_retriever.multi_path_retrieve(
                query=search_query,
                top_k=top_k,
                index_ids=index_ids,
                enabled_paths=strategy_config["enabled_paths"]
            )
            
            # 合并结果
            for path_name, nodes in path_results.items():
                if path_name not in all_retrieval_results:
                    all_retrieval_results[path_name] = []
                all_retrieval_results[path_name].extend(nodes)
        
        # 4. 动态权重调整
        adaptive_weights = self.weight_adjuster.calculate_adaptive_weights(
            query_type=self._classify_query_type(query),
            retrieval_results=all_retrieval_results
        )
        
        # 5. 结果融合和去重
        final_nodes = self._fuse_and_deduplicate_results(
            all_retrieval_results,
            adaptive_weights,
            top_k
        )
        
        # 6. 对话上下文增强
        if chat_history:
            final_nodes = self._enhance_with_chat_context(final_nodes, chat_history, query)
        
        logger.info(f"✅ Optimized retrieval completed, returning {len(final_nodes)} nodes")
        return final_nodes
    
    def _extract_context_from_history(self, chat_history: List[Dict[str, str]]) -> str:
        """从对话历史中提取上下文"""
        if not chat_history:
            return ""
        
        # 提取最近几轮对话的关键信息
        recent_messages = chat_history[-4:]  # 最近2轮对话
        context_parts = []
        
        for msg in recent_messages:
            if msg["role"] == "user":
                context_parts.append(f"用户问题: {msg['content']}")
            elif msg["role"] == "assistant":
                # 只保留回答的前100个字符作为上下文
                content = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
                context_parts.append(f"助手回答: {content}")
        
        return "\n".join(context_parts)
    
    def _classify_query_type(self, query: str) -> str:
        """分类查询类型"""
        query_lower = query.lower()
        
        if any(keyword in query_lower for keyword in ["什么是", "定义", "概念"]):
            return "definition"
        elif any(keyword in query_lower for keyword in ["如何", "怎么", "方法"]):
            return "how_to"
        elif any(keyword in query_lower for keyword in ["为什么", "原因", "why"]):
            return "explanation"
        elif any(keyword in query_lower for keyword in ["比较", "区别", "差异"]):
            return "comparison"
        else:
            return "general"
    
    def _fuse_and_deduplicate_results(self,
                                     retrieval_results: Dict[str, List[NodeWithScore]],
                                     weights: Dict[str, float],
                                     top_k: int) -> List[NodeWithScore]:
        """融合和去重检索结果"""
        # 收集所有节点并计算加权分数
        node_scores = {}
        
        for path_name, nodes in retrieval_results.items():
            weight = weights.get(path_name, 0.0)
            
            for node in nodes:
                node_id = self._get_node_id(node)
                
                if node_id not in node_scores:
                    node_scores[node_id] = {
                        "node": node,
                        "weighted_score": 0.0,
                        "path_scores": {}
                    }
                
                # 累加加权分数
                node_scores[node_id]["weighted_score"] += node.score * weight
                node_scores[node_id]["path_scores"][path_name] = node.score
        
        # 按加权分数排序
        sorted_nodes = sorted(
            node_scores.values(),
            key=lambda x: x["weighted_score"],
            reverse=True
        )
        
        # 更新节点分数并返回
        final_nodes = []
        for item in sorted_nodes[:top_k]:
            node = item["node"]
            node.score = item["weighted_score"]
            
            # 添加融合信息到元数据
            if not hasattr(node, 'metadata') or node.metadata is None:
                node.metadata = {}
            node.metadata["fusion_score"] = item["weighted_score"]
            node.metadata["path_scores"] = item["path_scores"]
            
            final_nodes.append(node)
        
        return final_nodes
    
    def _get_node_id(self, node: NodeWithScore) -> str:
        """获取节点唯一标识"""
        # 使用节点文本的哈希作为唯一标识
        return str(hash(node.node.text))
    
    def _enhance_with_chat_context(self,
                                 nodes: List[NodeWithScore],
                                 chat_history: List[Dict[str, str]],
                                 current_query: str) -> List[NodeWithScore]:
        """基于对话上下文增强检索结果"""
        if not chat_history:
            return nodes
        
        # 提取对话中的关键词
        chat_keywords = set()
        for msg in chat_history[-6:]:  # 最近3轮对话
            content = msg["content"].lower()
            # 简单的关键词提取（可以使用更复杂的NLP方法）
            words = content.split()
            chat_keywords.update([word for word in words if len(word) > 2])
        
        # 基于对话上下文调整节点分数
        for node in nodes:
            context_boost = 0.0
            node_text = node.node.text.lower()
            
            # 计算与对话历史的相关性
            matching_keywords = sum(1 for keyword in chat_keywords if keyword in node_text)
            if matching_keywords > 0:
                context_boost = min(matching_keywords * 0.05, 0.2)  # 最多提升0.2分
            
            # 应用上下文增强
            node.score = min(node.score + context_boost, 1.0)
            
            # 记录增强信息
            if not hasattr(node, 'metadata') or node.metadata is None:
                node.metadata = {}
            node.metadata["context_boost"] = context_boost
            node.metadata["matching_keywords"] = matching_keywords
        
        # 重新排序
        nodes.sort(key=lambda x: x.score, reverse=True)
        return nodes
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """获取优化统计信息"""
        return {
            "available_strategies": list(self.strategies.keys()),
            "retrieval_paths": list(self.multi_path_retriever.retrieval_paths.keys()),
            "performance_history": {
                path: len(history) for path, history in self.weight_adjuster.performance_history.items()
            },
            "cache_stats": {
                "expansion_cache_size": len(self.query_expander.expansion_cache)
            }
        }