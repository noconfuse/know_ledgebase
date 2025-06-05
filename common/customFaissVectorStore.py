from typing import Dict
import numpy as np
from llama_index.vector_stores.faiss import FaissVectorStore
from typing import List,Any
from llama_index.core.schema import BaseNode
class CustomFaissVectorStore(FaissVectorStore):
    """支持精准删除的Faiss扩展类"""
    stores_text: bool = True
    def __init__(self, faiss_index: Any):
        super().__init__(faiss_index)
        self._id_mapping: Dict[str, int] = {}  # 节点ID到Faiss ID的映射
        self._current_max_id: int = 0
    
    def add(self, nodes: List[BaseNode], **kwargs) -> List[str]:
        ids = []
        embeddings = []
        for node in nodes:
            faiss_id = self._get_or_create_id(node.node_id)
            embedding = node.get_embedding()
            
            # 单次添加 ID 和向量
            ids.append(faiss_id)
            embeddings.append(embedding)
        
        # 转换为 NumPy 数组
        id_array = np.array(ids, dtype=np.int64)
        embedding_array = np.array(embeddings, dtype=np.float32)
        
        # 添加时校验维度
        assert len(id_array) == embedding_array.shape[0], "ID数量与向量数量必须一致"
        
        self._faiss_index.add_with_ids(embedding_array, id_array)
        return [str(id) for id in ids]

    def _get_or_create_id(self, node_id: str) -> int:
        """为节点生成唯一整数ID"""
        if node_id in self._id_mapping:
            return self._id_mapping[node_id]
        
        # 新ID生成策略（示例使用哈希，可自定义）
        new_id = self._generate_unique_id(node_id)
        self._id_mapping[node_id] = new_id
        return new_id

    def _generate_unique_id(self, node_id: str) -> int:
        """ID生成策略：哈希后取模"""
        hash_val = hash(node_id)
        return int(abs(hash_val % (2**63 - 1)))  # 兼容int64
    
    def delete_ids(self, node_ids: List[str]) -> None:
        """根据节点ID删除向量"""
        # 转换节点ID到Faiss ID
        faiss_ids = [self._id_mapping[id] for id in node_ids if id in self._id_mapping]
        
        if not faiss_ids:
            return

        # 执行删除
        id_array = np.array(faiss_ids, dtype=np.int64)
        self._faiss_index.remove_ids(id_array)
        
        # 清理映射表
        for id in node_ids:
            self._id_mapping.pop(id, None)