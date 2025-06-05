# 针对法规文件解析，建立索引

from pathlib import Path
from pydantic import BaseModel,Field
from typing import List,Dict,Sequence
from llama_index.core import Document,VectorStoreIndex,StorageContext
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.readers.file import FlatReader
from llama_index.core.extractors import (    SummaryExtractor,  # 摘要提取器   
                                          QuestionsAnsweredExtractor,  # 问题回答提取器   
                                            TitleExtractor,  # 标题提取器   
                                              KeywordExtractor,  # 关键词提取器    
                                              BaseExtractor,
                                              DocumentContextExtractor,
                                              PydanticProgramExtractor,
                                              )
from llama_index.core.program import LLMTextCompletionProgram
from llama_index.core.ingestion import IngestionPipeline,IngestionCache
import torch
from llama_index.core.schema import NodeRelationship,RelatedNodeInfo,TextNode,BaseNode
from llama_index.core.storage.docstore import SimpleDocumentStore
from app.model.model_manager import model_manager
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.indices.loading import load_index_from_storage
import chromadb


EXTRACT_TEMPLATE_STR = """\
Here is the content of the section:
----------------
{context_str}
----------------
Given the contextual information, extract out a {class_name} object.\
"""

# 法律法规文件元数据
class LawMetadata(BaseModel):  
    """法律法规文件元数据"""
    document_title: str = Field(..., description="法律法规名称")
    document_number: str = Field(..., description="法律法规文件编号")
    summary: str = Field(..., description="法律法规文件摘要")
    keywords: List[str] = Field(..., description="法律法规文件关键词")
    issuing_authority: str = Field(..., description="法律法规文件颁布机构")
    # implementation_date: str = Field(..., description="法律法规文件实施日期") # 可能没有
    applicable_scope: str = Field(..., description="法律法规文件适用范围")
    exclusion_scope: str = Field(..., description="法律法规文件排除范围")
    document_type: str = Field(..., description="法律法规文件类型")



# 文档级元数据获取、建立关系
class CustomExtractor(BaseExtractor):
    async def aextract(self, nodes:Sequence[BaseNode]) -> List[Dict]:
        self_program = LLMTextCompletionProgram.from_defaults(    
            output_cls=LawMetadata,    
            prompt_template_str="{input}",
            verbose=True,
            llm=model_manager.ship_check_llm
        )
        program_extractor = PydanticProgramExtractor(   
            program=self_program, 
            input_key="input", 
            show_progress=True,
            extract_template_str=EXTRACT_TEMPLATE_STR,
            is_text_node_only=False
        )
        #  取前40个节点
        target_nodes = nodes[:40]  # 取前30个节点进行提取
        all_text = "\n\n".join([node.get_content() for node in target_nodes])
        meta_list = await program_extractor.aextract(nodes=[TextNode(text=all_text)])
        meta_data = meta_list[0]
        meta_data["keywords"] = ','.join(meta_data.get("keywords", []))
        path_to_node = {}
        new_meta_list = []
        for node in nodes:
            path = node.metadata.get("header_path", "")
            path_to_node[path] = node
            # 初始化 relationships 字典（如果不存在）
            if not node.relationships:
                node.relationships = {}

        for node in nodes:
            current_path = node.metadata.get("header_path", "")
            new_meta_list.append(meta_data)
            if not current_path:
                continue
            # 获取父路径（根据路径层级分割）
            path_parts = [p for p in current_path.split("/") if p]
            if len(path_parts) < 1:  # 根节点无需处理
                continue
            # 父路径比当前路径少最后一级
            parent_parts = path_parts[:-1]
            parent_path = "/" + "/".join(parent_parts) if parent_parts else "/"
            
            if parent_path in path_to_node:
                parent_node = path_to_node[parent_path]
                # 设置子节点关系（双向绑定）
                # 1. 当前节点的父关系
                node.relationships[NodeRelationship.PARENT] = RelatedNodeInfo(
                    node_id=parent_node.node_id,
                )
                
                # 2. 父节点的子关系
                if NodeRelationship.CHILD not in parent_node.relationships:
                    parent_node.relationships[NodeRelationship.CHILD] = []
                    
                parent_node.relationships[NodeRelationship.CHILD].append(
                    RelatedNodeInfo(
                        node_id=node.node_id,
                    )
                )
                
                
        return new_meta_list
    

class LawDocumentIndexService:
    def __init__(self, miner_ouput_dir: str):
        """初始化PDF解析器"""
        self.file_reader = FlatReader()
        self.miner_ouput_dir = miner_ouput_dir
        print(f"{self.miner_ouput_dir}/storage")
        self.storage_path = Path(f"{self.miner_ouput_dir}/storage")
        self.chroma_path = Path(f"{self.miner_ouput_dir}/chroma_db")

        self.md_node_parser = MarkdownNodeParser.from_defaults()
        
        self._init_components()

    def _init_components(self):
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # 初始化chroma客户端
        chroma_client = chromadb.PersistentClient(path=str(self.chroma_path))
        self.vector_store = ChromaVectorStore(
            chroma_collection=chroma_client.get_or_create_collection(
                name="law_documents",
                metadata={"hnsw:space": "cosine"}
            )
        )

        self.md_node_parser = MarkdownNodeParser.from_defaults()
        self.pipeline = IngestionPipeline(
            transformations=[
                self.md_node_parser,
                CustomExtractor(),
                model_manager.embed_model
            ],
            vector_store=self.vector_store,
            docstore=SimpleDocumentStore(),
            disable_cache=True
        )
        # 初始化或加载索引
        if self._pipeline_exists():
            self._init_index()

    def _pipeline_exists(self):
        return (self.storage_path / "docstore.json").exists()

    def _init_index(self):
        self.pipeline.load(self.storage_path)
        print('load index')
        self.index = VectorStoreIndex.from_vector_store(self.vector_store,model_manager.embed_model)

    def _load_markdown(self, file_name: str) -> List[Document]:
        doc_data = self.file_reader.load_data(Path(f"{self.miner_ouput_dir}/{file_name}_rebuilt.md"))
        return doc_data
    
    async def parse_and_build_index(self, file_name: str) -> list:
        doc_data = self._load_markdown(file_name)
        self.pipeline.cache.clear()
        # 解析文档
        nodes = await self.pipeline.arun(documents=doc_data, show_progress=True)
        for node in nodes:
            with open(f"{self.miner_ouput_dir}/{file_name}_nodes_list.txt","a", encoding="utf-8") as f:
                f.write(f"{node.get_content(metadata_mode='all')}\n=======================\n")

        # 持久化存储
        # storage_context = StorageContext.from_defaults(
        #     docstore=SimpleDocumentStore(),
        #     vector_store=self.vector_store
        # )
        # storage_context.persist(persist_dir=self.storage_path)
        self.pipeline.persist(persist_dir=self.storage_path)
        # 创建索引对象
        self.index = VectorStoreIndex.from_vector_store(self.pipeline.vector_store,model_manager.embed_model)
       
        # 保存到本地
        return self.index

