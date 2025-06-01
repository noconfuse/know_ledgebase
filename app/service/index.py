from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage,Document
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import BaseRetriever, QueryFusionRetriever, VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from app.common.customFaissVectorStore import CustomFaissVectorStore
from pathlib import Path
from typing import List,Optional,cast
from fastapi.logger import logger
from asyncio import Lock
import hashlib
import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from llama_index.core.schema import (
    QueryBundle,
    NodeWithScore,
    BaseNode,
)
from llama_index.core.vector_stores.types import MetadataFilter
from app.common.utils import tokenize
from faiss import IndexFlatL2, IndexIDMap2

logger = logging.getLogger(__name__)

class SentenceTransformerRetriever(BaseRetriever):
    def __init__(self, index, model):
        self.index = index
        self.model = model  # 直接使用传入的模型实例

    def _retrieve(self, query: QueryBundle, top_k: int = 3) -> List[NodeWithScore]:
        """基于语义相似度检索 top_k 文档"""
        # 计算查询句子的嵌入
        query_embedding = self.model.encode(query.query_str)
        doc_embeddings = []
        doc_texts = []
         # 获取所有文档的嵌入和文本
        for doc in self.index.docstore.docs.values():
            doc_embeddings.append(self.model.encode(doc.text))
            doc_texts.append(doc.text)
        # 计算余弦相似度
        similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
        # 按相似度排序并返回 top_k 文档
        top_k_indices = np.argsort(similarities)[-top_k:][::-1]
        top_k_scores = similarities[top_k_indices]

        nodes: List[NodeWithScore] = []
        
        # 按照相似度和索引返回对应的文档
        for idx, score in zip(top_k_indices, top_k_scores):
            node = cast(BaseNode, list(self.index.docstore.docs.values())[idx])  # 获取文档对象
            nodes.append(NodeWithScore(node=node, score=float(score)))  # 将文档和相似度分数封装为 NodeWithScore

        # 合并跨页的文档
        # merged_nodes = self._merge_cross_page_nodes(nodes)
        return nodes  # 返回合并后的 top_k 结果

    async def aretrieve(self, query: QueryBundle, top_k: int = 3) -> List[NodeWithScore]:
        """异步检索 top_k 文档"""
        return self._retrieve(query, top_k)

    # 如果你需要保留 retrieve 方法，可以这样写
    def retrieve(self, query: QueryBundle, top_k: int = 3) -> List[NodeWithScore]:
        return self._retrieve(query, top_k)

class IndexService:
    _instance = None
    fusion_retriever: Optional[QueryFusionRetriever] = None
    index: Optional[VectorStoreIndex] = None
    def __init__(self, store_dir: str,embed_model=None,llm=None):
        self._index_lock = Lock()
        self.vector_store: Optional[CustomFaissVectorStore] = None
        self.store_dir = store_dir
        self.embed_model = embed_model
        self.llm = llm
        self.load_existing_index()

    def load_existing_index(self):
        """Load existing index if it exists."""
        storage_dir = Path(self.store_dir)
        if not storage_dir.exists():
            logger.info(f"Creating storage directory at {storage_dir}")
            storage_dir.mkdir(parents=True, exist_ok=True)

        docstore_path = storage_dir / "docstore.json"
        if docstore_path.exists() and docstore_path.is_file():
            try:
                self.vector_store = CustomFaissVectorStore.from_persist_dir(persist_dir=str(storage_dir))
                storage_context = StorageContext.from_defaults(persist_dir=str(storage_dir),vector_store=self.vector_store)
                self.index = load_index_from_storage(storage_context, embed_model=self.embed_model)
                logger.info("Existing index loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load existing index: {e}")
        else:
            logger.info("No existing index found or docstore.json is missing.")

    def initialize_fusion_retriever(self):
        """Initialize the Fusion Retriever with BM25 and semantic retrievers."""
        if not self.index:
            raise ValueError("Index must be built before initializing Fusion Retriever.")
        bm25_retriever = BM25Retriever.from_defaults(docstore=self.index.docstore, similarity_top_k=3)
        # dense_retriever = VectorIndexRetriever(index=self.index,filters=MetadataFilter(), similarity_top_k=3)
        self.fusion_retriever = QueryFusionRetriever(
            retrievers=[ self.index.as_retriever(), bm25_retriever], similarity_top_k=3,  # 检索召回 top k 结果
            num_queries=3,  # 生成 query 数
            llm=self.llm,
            use_async=True)
        logger.info("Fusion Retriever initialized successfully.")

    

    def _generate_node_uuid(self, node: Document) -> str:
        """为节点生成唯一标识符"""
        # 使用元数据生成UUID
        metadata = node.get_metadata_str()
        unique_id = hashlib.md5(metadata.encode()).hexdigest()
        return str(unique_id)
        
    def is_doc_exists(self, unique_id):
        return any(d.metadata.get('unique_id') == unique_id for d in self.index.docstore.docs.values())
   
    async def build_or_update_index(self, nodes: List[BaseNode]):
        """Build or update the vector store index with the provided document files."""
        async with self._index_lock:
            if not self.index:
                base_index = IndexFlatL2(768) 
                faiss_index = IndexIDMap2(base_index)
                self.vector_store = CustomFaissVectorStore(faiss_index=faiss_index) 
                storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
                self.index = VectorStoreIndex([],embed_model=self.embed_model, storage_context=storage_context,show_progress=True)
        all_nodes = []
        for node in nodes:
            # 计算唯一ID
            uuid = self._generate_node_uuid(node)
            # 检查是否已存在
            if not self.is_doc_exists(uuid):
                node.metadata['uuid'] = uuid
                # node.text = tokenize(node.text)  # 对文本进行分词
                all_nodes.append(node)
                logger.info(f"Inserted node with unique ID: {uuid}")
            else:
                logger.info(f"Node with unique ID {uuid} already exists, skipping.")

        self.index.insert_nodes(all_nodes)

        async with self._index_lock:
            self.index.storage_context.persist(persist_dir=self.store_dir)
            logger.info("Index updated successfully.")

    async def delete_documents(self, source_path: str, page: Optional[int] = None, fragment: Optional[int] = None):
        """删除指定文件/页面/片段的文档"""
        async with self._index_lock:
            if not self.index:
                logger.warning("Index not initialized, nothing to delete")
                return

            storage_context = self.index.storage_context
            docstore = storage_context.docstore
            vector_store = storage_context.vector_store

            # 查找匹配文档
            to_delete = []
            for node_id in docstore.docs.keys():
                node = docstore.get_node(node_id)
                if not self._match_metadata(node.metadata, source_path, page, fragment):
                    continue
                to_delete.append(node_id)

            # 执行删除操作
            if to_delete:
                # 从文档存储删除
                for node_id in to_delete:
                    docstore.delete_document(node_id)
                # 处理向量存储
                if isinstance(vector_store, CustomFaissVectorStore):
                    # 仅当使用 IndexIDMap2 时支持删除
                    if self._supports_id_mapping(vector_store):
                        # 直接通过ID删除
                        vector_store.delete_ids(to_delete)
                    else:
                        # 回退到重建索引
                        self._rebuild_faiss_index(docstore, vector_store)
                        # 持久化存储
                        storage_context.persist(persist_dir=self.store_dir)
                        logger.info(f"成功删除 {len(to_delete)} 个文档")
            else:
                logger.info("未找到匹配文档")

    def _supports_id_mapping(self, vector_store: CustomFaissVectorStore) -> bool:
        """检查是否支持ID映射删除"""
        from faiss import IndexIDMap2
        return isinstance(vector_store.client, IndexIDMap2)

    def _match_metadata(self, metadata: dict, source_path: str, page: Optional[int], fragment: Optional[int]) -> bool:
        """元数据匹配逻辑"""
        if metadata.get('source') != source_path:
            return False
        if page is not None and metadata.get('page') != page:
            return False
        if fragment is not None and metadata.get('fragment') != fragment:
            return False
        return True

    def _rebuild_faiss_index(self, docstore, vector_store:CustomFaissVectorStore):
        """安全重建索引"""
    
        # 收集有效嵌入和ID
        valid_embeddings = []
        valid_ids = []
        for node_id in docstore.docs.keys():
            node = docstore.get_node(node_id)
            if node.embedding is not None:
                valid_embeddings.append(node.embedding)
                valid_ids.append(int(node.node_id))  # FAISS要求整数ID

        # 创建带ID映射的新索引
        dimension = 768
        new_index = IndexIDMap2(IndexFlatL2(dimension))
        if valid_embeddings:
            embeddings_array = np.array(valid_embeddings, dtype='float32')
            id_array = np.array(valid_ids, dtype=np.int64)
            new_index.add_with_ids(embeddings_array, id_array)

        # 通过安全方法替换索引
        vector_store.replace_index(new_index)

