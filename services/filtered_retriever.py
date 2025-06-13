# -*- coding: utf-8 -*-
"""
带相似度过滤的检索器包装类
"""

import logging
from typing import List, Optional
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.schema import QueryBundle, NodeWithScore
from llama_index.core.callbacks.base import CallbackManager
from config import settings

logger = logging.getLogger(__name__)

class FilteredRetriever(BaseRetriever):
    """带相似度过滤的检索器包装类"""
    
    def __init__(
        self,
        base_retriever: BaseRetriever,
        similarity_threshold: float = None,
        callback_manager: Optional[CallbackManager] = None
    ):
        """初始化过滤检索器
        
        Args:
            base_retriever: 基础检索器
            similarity_threshold: 相似度阈值，低于此值的节点将被过滤
            callback_manager: 回调管理器
        """
        super().__init__(callback_manager=callback_manager)
        self._base_retriever = base_retriever
        self._similarity_threshold = similarity_threshold or settings.SIMILARITY_THRESHOLD
        
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """执行检索并过滤低相似度节点"""
        # 使用基础检索器获取节点
        nodes = self._base_retriever.retrieve(query_bundle)
        
        # 应用相似度过滤
        filtered_nodes = [
            node for node in nodes 
            if node.score >= self._similarity_threshold
        ]
        
        logger.info(
            f"Filtered {len(nodes) - len(filtered_nodes)} nodes below threshold {self._similarity_threshold}, "
            f"remaining: {len(filtered_nodes)}"
        )
        
        return filtered_nodes
    
    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """异步执行检索并过滤低相似度节点"""
        # 使用基础检索器获取节点
        nodes = await self._base_retriever.aretrieve(query_bundle)
        
        # 应用相似度过滤
        filtered_nodes = [
            node for node in nodes 
            if node.score >= self._similarity_threshold
        ]
        
        logger.info(
            f"Filtered {len(nodes) - len(filtered_nodes)} nodes below threshold {self._similarity_threshold}, "
            f"remaining: {len(filtered_nodes)}"
        )
        
        return filtered_nodes
    
    @property
    def similarity_threshold(self) -> float:
        """获取相似度阈值"""
        return self._similarity_threshold
    
    @similarity_threshold.setter
    def similarity_threshold(self, value: float):
        """设置相似度阈值"""
        self._similarity_threshold = value
        logger.info(f"Updated similarity threshold to {value}")