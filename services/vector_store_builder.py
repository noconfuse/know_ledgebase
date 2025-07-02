import asyncio
import os
from tabnanny import verbose
import uuid
import json
import time
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter, HTMLNodeParser
from llama_index.core.schema import BaseNode, TextNode
from llama_index.core.node_parser.interface import NodeParser
from common.custom_markdown_node_parser import (
    CustomMarkdownNodeParser as MarkdownNodeParser,
)
from llama_index.node_parser.docling import DoclingNodeParser
from services.smart_metadata_extractor import SmartMetadataExtractor
from llama_index.core.ingestion import IngestionPipeline
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.llms.openai_like import OpenAILike

# 使用已导入的SessionLocal
from dao.index_dao import IndexDAO

from config import settings
from models.database import get_db, SessionLocal
from services.document_docling_processor import DocumentDoclingProcessor
from services.document_html_processor import DocumentHTMLProcessor
from services.document_parser import document_parser, TaskStatus
from common.postgres_vector_store import create_postgres_vector_store_builder
from dao.task_dao import TaskDAO
from models.task_models import ParseTask, VectorStoreTask
from services.model_client_factory import ModelClientFactory

logger = logging.getLogger(__name__)


class DocumentPreservingTransformer(NodeParser):
    """自定义转换器，保留原始文档节点的同时创建切分节点"""
    
    def __init__(self, node_parser, **kwargs):
        super().__init__(**kwargs)
        self._node_parser = node_parser
    
    def _parse_nodes(self, nodes, show_progress=False, **kwargs):
        """解析节点，同时保留原始文档和创建切分节点"""
        result_nodes = []
        
        for node in nodes:
            if isinstance(node, Document):
                # 1. 创建原始文档节点
                original_node = TextNode(
                    text=node.text,
                    metadata={
                        **node.metadata,
                        "node_type": "original_document",
                        "is_complete_document": True
                    }
                )
                result_nodes.append(original_node)
                
                # 2. 创建切分节点
                if isinstance(self._node_parser, list):
                    # 如果是多个解析器，依次应用
                    current_nodes = [node]
                    for parser in self._node_parser:
                        if hasattr(parser, '_parse_nodes'):
                            current_nodes = parser._parse_nodes(current_nodes, show_progress=show_progress, **kwargs)
                        elif hasattr(parser, 'get_nodes_from_documents'):
                            # 对于某些解析器，使用get_nodes_from_documents方法
                            current_nodes = parser.get_nodes_from_documents(current_nodes, show_progress=show_progress)
                else:
                    # 单个解析器
                    if hasattr(self._node_parser, '_parse_nodes'):
                        current_nodes = self._node_parser._parse_nodes([node], show_progress=show_progress, **kwargs)
                    elif hasattr(self._node_parser, 'get_nodes_from_documents'):
                        current_nodes = self._node_parser.get_nodes_from_documents([node], show_progress=show_progress)
                
                # 为切分节点添加标记并继承原始文档的metadata
                for chunk_node in current_nodes:
                    if hasattr(chunk_node, 'metadata'):
                        # 继承原始文档的所有metadata
                        chunk_node.metadata.update(node.metadata)
                        # 添加切分节点特有的标记
                        chunk_node.metadata["node_type"] = "chunk"
                        chunk_node.metadata["is_complete_document"] = False
                
                result_nodes.extend(current_nodes)
            else:
                # 如果输入已经是节点，直接处理
                result_nodes.append(node)
        
        return result_nodes


class VectorStoreBuilder:
    """向量数据库构建器"""

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._setup_models()
            self.tasks: Dict[str, VectorStoreTask] = {}
            # self.executor = ThreadPoolExecutor(max_workers=2)  # 限制并发数 - 似乎未使用，暂时注释掉以解决进程挂起问题

            # 初始化任务DAO
            self.task_dao = TaskDAO()

            self._initialized = True
            logger.info("VectorStoreBuilder initialized")
    
    def _clear_gpu_memory_if_needed(self):
        """在需要时清理GPU内存"""
        try:
            import torch
            import gc
            
            if torch.cuda.is_available() and settings.USE_GPU:
                # 获取当前GPU内存使用情况
                device = torch.device(settings.GPU_DEVICE)
                total_memory = torch.cuda.get_device_properties(device).total_memory
                allocated_memory = torch.cuda.memory_allocated(device)
                cached_memory = torch.cuda.memory_reserved(device)
                
                # 计算内存使用率
                memory_usage_ratio = cached_memory / total_memory
                
                logger.info(f"GPU内存使用情况: {cached_memory / 1024**3:.2f}GB / {total_memory / 1024**3:.2f}GB ({memory_usage_ratio*100:.1f}%)")
                
                # 如果内存使用率超过70%，进行清理
                if memory_usage_ratio > 0.7:
                    logger.warning(f"GPU内存使用率过高 ({memory_usage_ratio*100:.1f}%)，开始清理...")
                    
                    # 清理PyTorch缓存
                    torch.cuda.empty_cache()
                    
                    # 强制垃圾回收
                    gc.collect()
                    
                    # 再次检查内存
                    new_cached_memory = torch.cuda.memory_reserved(device)
                    new_usage_ratio = new_cached_memory / total_memory
                    
                    logger.info(f"清理后GPU内存使用: {new_cached_memory / 1024**3:.2f}GB / {total_memory / 1024**3:.2f}GB ({new_usage_ratio*100:.1f}%)")
                    
                    if new_usage_ratio < memory_usage_ratio:
                        logger.info(f"✅ GPU内存清理成功，释放了 {(cached_memory - new_cached_memory) / 1024**3:.2f}GB")
                    else:
                        logger.warning("⚠️  GPU内存清理效果有限，建议降低批处理大小或重启应用")
                else:
                    logger.info("✅ GPU内存使用正常，无需清理")
        except Exception as e:
            logger.warning(f"GPU内存检查失败: {e}")

    def _setup_models(self):
        """设置模型"""
        try:
            # 初始化嵌入模型
            self.embed_model = ModelClientFactory.create_embedding_client(
                settings.embedding_model_settings
            )

            self.llm = ModelClientFactory.create_llm_client(settings.llm_model_settings)

        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            raise

    async def build_vector_store(self, task_id: str, config: Dict[str, Any]) -> str:
        """构建向量数据库

        Args:
            task_id: 解析任务ID
            config: 构建配置

        Returns:
            向量存储任务ID
        """
        return self.create_vector_store_task_from_parse_task(task_id, config)

    async def _execute_build_task(self, task: VectorStoreTask):
        """执行构建任务"""
        try:
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.utcnow()
            task.progress = 5

            # 更新数据库中的任务状态
            self._update_task_in_db(task)

            logger.info(f"Starting vector store build task {task.task_id}")

            # 直接执行异步构建
            result = await self._build_vector_store_sync(task)

            task.result = result
            task.status = TaskStatus.COMPLETED
            task.progress = 100
            task.completed_at = datetime.utcnow()

            # 更新数据库中的任务状态
            self._update_task_in_db(task)

            logger.info(
                f"Vector store build task {task.task_id} completed successfully"
            )

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = datetime.utcnow()

            # 更新数据库中的任务状态
            self._update_task_in_db(task)

            logger.error(f"Vector store build task {task.task_id} failed: {e}")

    def test_collection_documents(self, task_id: str):
        """测试收集文档"""
        task = self.task_dao.get_parse_task(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found")
        
        all_tasks = self._collect_all_tasks_recursively(task)
        
        # 转换为字典并处理datetime字段
        result = []
        for task in all_tasks:
            task_dict = task.to_dict()
            # 将datetime字段转换为字符串
            for key, value in task_dict.items():
                if hasattr(value, 'isoformat'):  # datetime对象有isoformat方法
                    task_dict[key] = value.isoformat() if value else None
            result.append(task_dict)
        
        return result

    async def _build_vector_store_sync(self, task: VectorStoreTask) -> Dict[str, Any]:
        """同步构建向量数据库"""
        try:
            # 1. 收集文档文件
            task.progress = 10
            documents_by_type, origin_file_path = self._collect_documents(task)
            total_files = sum(len(docs) for docs in documents_by_type.values())
            task.total_files = total_files

            all_nodes = []
            all_doc_ids = []

            # 3. 设置提取器
            task.progress = 30
            extractors = [
                SmartMetadataExtractor(
                    llm=self.llm,
                    min_chunk_size_for_extraction=task.config.get(
                        "min_chunk_size_for_extraction", 100
                    ),
                    min_chunk_size_for_summary=task.config.get(
                        "min_chunk_size_for_summary", 512
                    ),
                    min_chunk_size_for_qa=task.config.get(
                        "min_chunk_size_for_qa", 1024
                    ),
                    max_keywords=task.config.get("max_keywords", 5),
                    enable_persistent_cache=task.config.get("enable_persistent_cache", True),
                    cache_dir=task.config.get("cache_dir", "cache/metadata"),
                    # TODO num_questions=task.config.get("num_questions", 2),
                )
            ]

            # 4. 为每种文件类型创建专用处理管道
            task.progress = 40
            progress_step = 20 / total_files  # 在40-60%之间分配进度

            for file_type, type_docs in documents_by_type.items():
                logger.info(f"Processing {len(type_docs)} {file_type} files")

                # 获取适合该文件类型的节点解析器
                logger.info(f"Getting node parser for file type: {file_type}")
                node_parsers = self._get_node_parser_for_file_type(
                    file_type, task.config
                )

                # 确保node_parsers是列表格式
                if not isinstance(node_parsers, list):
                    node_parsers = [node_parsers]

                logger.info(
                    f"Node parsers configured: {[type(parser).__name__ for parser in node_parsers]}"
                )
                logger.info(
                    f"Extractors configured: {[type(extractor).__name__ for extractor in extractors]}"
                )
                logger.info(f"Embed model: {type(self.embed_model).__name__}")

                # 创建该文件类型的处理管道
                logger.info(f"Creating ingestion pipeline for {file_type}")
                pipeline = IngestionPipeline(
                    transformations=[*node_parsers, *extractors, self.embed_model]
                )

                start_time = time.time()
                logger.info(
                    f"Starting pipeline processing for {len(type_docs)} {file_type} documents"
                )

                # 记录每个文档的处理详情
                for i, doc in enumerate(type_docs):
                    logger.info(
                        f"Document {i+1}/{len(type_docs)}: {doc.metadata.get('file_name', 'unknown')} (size: {len(doc.text)} chars)"
                    )
                    all_doc_ids.append(doc.doc_id)

                try:
                    # 创建自定义的管道来逐步处理并记录每个阶段
                    logger.info("Starting transformation pipeline execution...")

                    # 逐个执行transformation步骤并记录时间
                    current_docs = type_docs
                    for step_idx, transformation in enumerate(pipeline.transformations):
                        step_start = time.time()
                        transformation_name = type(transformation).__name__
                        logger.info(f"\n{'='*60}")
                        logger.info(
                            f"Step {step_idx+1}/{len(pipeline.transformations)}: Starting {transformation_name}"
                        )
                        logger.info(
                            f"Input: {len(current_docs) if hasattr(current_docs, '__len__') else 'N/A'} items"
                        )

                        # 如果是LLM相关的extractor，记录更详细的信息
                        if "Extractor" in transformation_name:
                            logger.info(
                                f"🤖 LLM Extractor detected: {transformation_name}"
                            )
                            logger.info(
                                f"📝 This step will make LLM API calls - detailed logs will follow"
                            )
                            if hasattr(transformation, "llm"):
                                logger.info(
                                    f"🔧 LLM model: {type(transformation.llm).__name__}"
                                )

                        try:
                            if hasattr(transformation, "transform"):
                                # 对于node parser和其他transformer
                                if step_idx == 0:  # 第一步，输入是documents
                                    current_docs = transformation.transform(
                                        current_docs
                                    )
                                else:  # 后续步骤，输入是nodes
                                    current_docs = transformation.transform(
                                        current_docs
                                    )
                            elif hasattr(transformation, "__call__"):
                                # 对于embedding model等
                                current_docs = transformation(current_docs)
                            elif hasattr(transformation, "aextract"):
                                # 对于SmartMetadataExtractor，使用异步方法并传递进度回调
                                import asyncio
                                current_docs = await transformation.aextract(current_docs)

                            step_time = time.time() - step_start
                            logger.info(
                                f"✅ Step {step_idx+1} ({transformation_name}) completed successfully in {step_time:.2f}s"
                            )
                            logger.info(
                                f"📊 Output: {len(current_docs) if hasattr(current_docs, '__len__') else 'N/A'} items"
                            )

                            # 如果是LLM相关的extractor，记录成功信息
                            if "Extractor" in transformation_name:
                                logger.info(
                                    f"🎉 LLM Extractor {transformation_name} completed successfully"
                                )
                                logger.info(
                                    f"⏱️  Total LLM processing time: {step_time:.2f}s"
                                )

                        except Exception as e:
                            step_time = time.time() - step_start
                            logger.error(
                                f"❌ Step {step_idx+1} ({transformation_name}) FAILED after {step_time:.2f}s"
                            )
                            logger.error(f"🚨 Error in {transformation_name}: {str(e)}")
                            if "Extractor" in transformation_name:
                                logger.error(
                                    f"💥 LLM Extractor {transformation_name} failed - this is likely the timeout source!"
                                )
                                logger.error(
                                    f"🔍 Check the LLM API calls above for timeout or connection issues"
                                )
                            raise

                        logger.info(f"{'='*60}\n")

                    type_nodes = current_docs
                    processing_time = time.time() - start_time
                    logger.info(
                        f"Pipeline processing completed for {file_type} in {processing_time:.2f}s, generated {len(type_nodes)} nodes"
                    )

                    all_nodes.extend(type_nodes)
                    logger.info(f"Total nodes accumulated: {len(all_nodes)}")

                except Exception as e:
                    processing_time = time.time() - start_time
                    logger.error(
                        f"Pipeline processing failed for {file_type} after {processing_time:.2f}s: {e}"
                    )
                    logger.error(f"Error details: {str(e)}")
                    logger.error(
                        f"Error occurred during pipeline execution, last successful step info available in logs above"
                    )
                    raise

                task.progress += progress_step
                logger.info(f"Progress updated to {task.progress:.1f}%")

            # 5. 合并所有节点
            task.progress = 60
            task.total_nodes = len(all_nodes)
            nodes = all_nodes
            logger.info(f"Merging nodes completed, total nodes: {len(nodes)}")

            # 5.5 为节点添加页码信息
            nodes = self._map_page_info_to_nodes(nodes)
            logger.info(f"Page information mapped to nodes, total nodes: {len(nodes)}")

            # 5.6 输出所有节点文本到本地文件（用于调试）
            logger.info(f"🔍 准备保存节点调试信息，节点数量: {len(nodes)}")
            self._save_nodes_text_to_file(nodes, task.task_id)
            logger.info(f"✅ Nodes text saved to local file for debugging")

            # 6. 创建向量存储
            task.progress = 80
            logger.info(f"Starting vector store creation with {len(nodes)} nodes")
            
            # 在创建向量存储前清理GPU内存
            self._clear_gpu_memory_if_needed()
            
            start_time = time.time()

            try:
                # 依据parse_task_id 来创建向量索引或数据库
                vector_store, index, actual_index_id = (
                    self._create_or_update_vector_store(
                        nodes,
                        origin_file_path,
                        all_doc_ids,
                    )
                )
                creation_time = time.time() - start_time
                logger.info(
                    f"Vector store creation completed in {creation_time:.2f}s, using index_id: {actual_index_id}"
                )
            except Exception as e:
                creation_time = time.time() - start_time
                logger.error(
                    f"Vector store creation failed after {creation_time:.2f}s: {e}"
                )
                raise

            # 7. 保存索引
            task.progress = 95
            logger.info(
                f"Starting index save for task {task.task_id}, using index_id: {actual_index_id}"
            )

            # 8. 保存索引信息到数据库
            logger.info("Saving index information to database")
            start_time = time.time()
            try:
                # 自动生成索引描述
                logger.info("Generating automatic description for index")
                index_description = self._generate_auto_description(index, task.config)
                if not index_description:
                    logger.warning("Failed to generate automatic description")
                    index_description = "Auto-generated vector store index"
                else:
                    logger.info("Automatic description generated successfully")

                self._save_index_info_to_db(
                    origin_file_path,
                    actual_index_id,
                    index_description,
                    document_count=len(all_doc_ids),
                    node_count=len(nodes),
                    vector_dimension=settings.VECTOR_DIM,
                    processing_config=task.config,
                )
                db_time = time.time() - start_time
                logger.info(f"Index information saved to database in {db_time:.2f}s")
            except Exception as e:
                db_time = time.time() - start_time
                logger.error(
                    f"Failed to save index information to database after {db_time:.2f}s: {e}"
                )
                # 继续执行，不影响索引创建的主流程

            result = {
                "index_id": actual_index_id,
                "node_count": len(nodes),
                "vector_dimension": settings.VECTOR_DIM,
                "config": task.config,
            }

            return result

        except Exception as e:
            import traceback

            traceback.print_exc()
            logger.error(f"Error building vector store: {e}")
            raise

    def _process_document_task(self, doc_task) -> List[Document]:
        """根据任务类型处理单个文档任务并返回文档列表"""
        if doc_task.file_extension in [".html", ".htm"]:
            logger.info(f"Processing HTML task {doc_task.task_id} with HTMLProcessor")
            return DocumentHTMLProcessor().collect_document(doc_task)
        else:
            logger.info(
                f"Processing non-HTML task {doc_task.task_id} with DoclingProcessor"
            )
            return DocumentDoclingProcessor().collect_document(doc_task)

    def _collect_documents(
        self, task: VectorStoreTask
    ) -> Tuple[Dict[str, List[Document]], str]:
        """
        收集目录中的文档
        """
        with next(get_db()) as db:
            task_dao = TaskDAO(db)
            parse_task = task_dao.get_parse_task(task.parse_task_id)

            if not parse_task:
                raise ValueError(f"Parse task not found: {task.parse_task_id}")

            documents_by_type = {}

            # 递归收集所有层级的任务
            all_tasks = self._collect_all_tasks_recursively(parse_task)

            logger.warning(
                f"Found {len(all_tasks)} task(s) to process for document collection (including recursive subtasks)."
            )

            for task_item in all_tasks:
                docs = self._process_document_task(task_item)
                for doc in docs:
                    file_type = doc.metadata.get("file_type")  # 按file_type分类
                    documents_by_type.setdefault(file_type, []).append(doc)

            total_docs = sum(len(docs) for docs in documents_by_type.values())
            logger.info(
                f"Collected {total_docs} documents from task {task.parse_task_id}, grouped by file type."
            )
            return documents_by_type, parse_task.file_path

    def _collect_all_tasks_recursively(self, parse_task):
        """
        递归收集所有层级的任务

        由于 task_dao.get_parse_task 方法只能加载一层子任务，
        这个方法会通过多次查询数据库来获取所有层级的子任务。

        Args:
            parse_task: 根任务

        Returns:
            List: 包含所有层级任务的列表
        """
        all_tasks = []
        task_dao = self.task_dao

        def collect_tasks(task: ParseTask):
            """递归收集任务的内部函数"""
            # 基于KNOWLEDGE_BASE_DIR和task_id检查输出目录是否存在
            from config import settings
            expected_output_dir = os.path.join(settings.KNOWLEDGE_BASE_DIR, "outputs", task.task_id)
            if os.path.exists(expected_output_dir):
                all_tasks.append(task)
            
            # 如果任务有子任务，递归处理子任务
            if task.subtasks:
                for subtask in task.subtasks:
                    # 对于每个子任务，重新从数据库加载以获取其子任务
                    loaded_subtask = task_dao.get_parse_task(subtask.task_id)
                    if loaded_subtask:
                        collect_tasks(loaded_subtask)
                    else:
                        # 如果无法加载子任务，则使用当前任务
                        collect_tasks(subtask)
            elif not os.path.exists(expected_output_dir):
                # 如果没有子任务且没有输出目录，说明是叶子节点但没有输出，也添加到处理列表
                all_tasks.append(task)

        collect_tasks(parse_task)

        # 如果没有找到任何叶子节点任务，说明根任务本身就是叶子节点
        if not all_tasks:
            all_tasks.append(parse_task)

        logger.info(
            f"Recursively collected {len(all_tasks)} leaf tasks from parse_task {parse_task.task_id}"
        )
        return all_tasks

    def _get_node_parser_for_file_type(self, file_type: str, config: Dict[str, Any]):
        """根据文件类型获取合适的节点解析器，使用DocumentPreservingTransformer保留原始文档"""
        chunk_size = config.get("chunk_size", settings.CHUNK_SIZE)
        chunk_overlap = config.get("chunk_overlap", settings.CHUNK_OVERLAP)

        # 根据文件类型选择基础解析器
        if file_type == ".md":
            # Markdown文件使用MarkdownNodeParser，然后配合SentenceSplitter
            base_parsers = [
                MarkdownNodeParser.from_defaults(),
                SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap),
            ]
        elif file_type in [".html", ".htm"]:
            # HTML文件使用HTMLNodeParser，然后配合SentenceSplitter
            base_parsers = [
                HTMLNodeParser.from_defaults(),
                SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap),
            ]
        elif file_type == ".json":
            # JSON文件直接使用DoclingNodeParser
            base_parsers = [DoclingNodeParser()]
        else:
            # 文本文件直接使用SentenceSplitter
            base_parsers = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
        # 使用DocumentPreservingTransformer包装基础解析器
        return [DocumentPreservingTransformer(base_parsers)]

    def _determine_vector_store_strategy(
        self, origin_file_path: str
    ) -> Tuple[str, bool]:
        """
        根据任务ID确定向量存储策略：返回目标索引ID和是否需要更新

        Returns:
            Tuple[str, bool]: (target_index_id, should_update)
        """
        if origin_file_path:
            # 根据文件MD5检查是否已存在相同文件的索引
            try:
                db = SessionLocal()
                try:
                    existing_index = IndexDAO.get_index_by_origin_file_path(
                        db, origin_file_path
                    )
                    if existing_index:
                        # 找到现有索引，使用现有的index_id和对应的表
                        target_index_id = existing_index.index_id
                        logger.info(
                            f"Found existing index for origin_file_path: {origin_file_path}, using existing index_id: {target_index_id}"
                        )
                        return target_index_id, True
                    else:
                        # 没有找到现有索引，使用当前origin_file_path创建新的
                        logger.info(
                            f"No existing index found for origin_file_path: {origin_file_path}, creating new index with id: {origin_file_path}"
                        )
                        return str(uuid.uuid4()), False
                finally:
                    db.close()
            except Exception as e:
                logger.warning(
                    f"Error checking existing index by origin_file_path: {e}, using current origin_file_path: {origin_file_path}"
                )
                return str(uuid.uuid4()), False  # 保持返回origin_file_path
        else:
            # 如果没有文件信息，使用当前origin_file_path
            logger.info(
                f"No file MD5 available, using current origin_file_path: {origin_file_path}"
            )
            return str(uuid.uuid4()), False

    def _create_or_update_vector_store(
        self,
        nodes: List[Any],
        origin_file_path: str,
        all_doc_ids: List[str],
    ) -> Tuple[Any, Any, str]:
        """创建向量存储

        Returns:
            Tuple[Any, Any, str]: (vector_store, index, actual_index_id)
        """
        # 检查是否存在相同文件的索引，并确定使用的索引ID和表名
        target_index_id, should_update = self._determine_vector_store_strategy(
            origin_file_path
        )

        # 创建PostgreSQL向量存储
        logger.info(
            f"Creating PostgreSQL vector store with dimension {settings.VECTOR_DIM}"
        )
        start_time = time.time()
        # 创建PostgreSQL向量存储构建器（使用确定的索引ID）
        postgres_builder = create_postgres_vector_store_builder(
            host=settings.POSTGRES_HOST,
            port=settings.POSTGRES_PORT,
            database=settings.POSTGRES_DATABASE,
            user=settings.POSTGRES_USER,
            password=settings.POSTGRES_PASSWORD,
            table_name=f"{settings.POSTGRES_TABLE_NAME}_{target_index_id.replace('-', '_')}",
            embed_dim=settings.VECTOR_DIM,
        )

        if should_update:
            logger.info(f"Updating existing vector store with new data")
            index = postgres_builder.update_index_with_nodes(
                nodes, self.embed_model, all_doc_ids
            )
            vector_store = index.storage_context.vector_store
        else:
            logger.info(f"Creating new vector store")
            # 创建向量存储
            vector_store = postgres_builder.create_vector_store()
            store_time = time.time() - start_time
            logger.info(f"PostgreSQL vector store created in {store_time:.3f}s")

            # 创建存储上下文
            logger.info("Creating storage context")
            start_time = time.time()
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            context_time = time.time() - start_time
            logger.info(f"Storage context created in {context_time:.3f}s")

            # 创建索引
            logger.info(f"Creating vector store index with {len(nodes)} nodes")
            start_time = time.time()
            index = VectorStoreIndex(
                nodes=nodes,
                storage_context=storage_context,
                embed_model=self.embed_model,
            )

        index_time = time.time() - start_time
        logger.info(f"Vector store index processed in {index_time:.2f}s")

        # 返回实际使用的索引ID
        return vector_store, index, target_index_id

    def _save_index(self, index: VectorStoreIndex, index_id: str) -> str:
        """保存索引"""
        import time

        logger.info(f"Preparing index directory for {index_id}")
        index_dir = Path(settings.INDEX_STORE_PATH) / index_id
        index_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Index directory created: {index_dir}")

        # 保存索引
        logger.info(f"Starting index persistence to {index_dir}")
        start_time = time.time()
        index.storage_context.persist(persist_dir=str(index_dir))
        persist_time = time.time() - start_time

        logger.info(
            f"Index persistence completed in {persist_time:.2f}s to {index_dir}"
        )
        return str(index_dir)

    def _map_page_info_to_nodes(self, nodes: List[Any]) -> List[Any]:
        """将页码信息映射到文档切片节点

        这个方法尝试将content_list.json中的页码信息映射到每个文档切片的元数据中，
        使得在检索时可以知道每个切片来自原始文档的哪一页。

        Args:
            nodes: 文档切片节点列表

        Returns:
            添加了页码信息的节点列表
        """
        from pathlib import Path

        # 按文档分组节点
        nodes_by_doc = {}
        for node in nodes:
            if not hasattr(node, "metadata"):
                continue

            doc_path = node.metadata.get("original_file_path")  # 拿到向量化的源文件
            if not doc_path:
                continue

            if doc_path not in nodes_by_doc:
                nodes_by_doc[doc_path] = []
            nodes_by_doc[doc_path].append(node)

        # 处理每个文档的节点
        for doc_path, doc_nodes in nodes_by_doc.items():
            # 根据文档路径构建content_list.json路径
            doc_file = Path(doc_path)
            base_name = doc_file.stem
            content_list_path = doc_file.parent / f"{base_name}_content_list.json"

            if not content_list_path.exists():
                logger.warning(f"No content_list.json found for document: {doc_path}")
                continue

            try:
                with open(content_list_path, "r", encoding="utf-8") as f:
                    content_list = json.load(f)
            except Exception as e:
                logger.error(
                    f"Failed to load content_list from {content_list_path}: {e}"
                )
                continue

            if not isinstance(content_list, list) or not content_list:
                continue

            # 创建文本到页码的映射
            text_to_page = {}
            for item in content_list:
                if item.get("type") == "text" and "text" in item and "page_idx" in item:
                    text = item["text"]
                    page_idx = item["page_idx"]
                    text_to_page[text] = page_idx

            # 为每个节点分配页码
            for node in doc_nodes:
                if not hasattr(node, "text") or not node.text:
                    continue

                # 尝试直接匹配
                if node.text in text_to_page:
                    node.metadata["page_idx"] = text_to_page[node.text]
                    continue

                # 尝试部分匹配（查找节点文本中包含的最长content_list文本）
                best_match = None
                best_match_length = 0
                for text, page_idx in text_to_page.items():
                    if text in node.text and len(text) > best_match_length:
                        best_match = text
                        best_match_length = len(text)

                if best_match:
                    node.metadata["page_idx"] = text_to_page[best_match]
                    continue

                # 如果没有匹配，尝试反向匹配（查找包含节点文本的最短content_list文本）
                best_match = None
                best_match_length = float("inf")
                for text, page_idx in text_to_page.items():
                    if node.text in text and len(text) < best_match_length:
                        best_match = text
                        best_match_length = len(text)

                if best_match:
                    node.metadata["page_idx"] = text_to_page[best_match]

            # 统计添加了页码的节点数量
            nodes_with_page = sum(
                1 for node in doc_nodes if "page_idx" in node.metadata
            )
            logger.info(
                f"Added page information to {nodes_with_page}/{len(doc_nodes)} nodes for document: {doc_path}"
            )

        return nodes

    def _detect_global_circular_references(self, nodes: List[Any]):
        """检测节点间的全局循环引用"""
        logger.info(f"🔍 开始检测 {len(nodes)} 个节点的循环引用...")
        
        # 创建节点ID到节点的映射
        node_map = {}
        for i, node in enumerate(nodes):
            if hasattr(node, 'node_id'):
                node_map[node.node_id] = (i, node)
            else:
                logger.warning(f"节点 {i} 没有node_id属性")
        
        logger.info(f"📊 创建了 {len(node_map)} 个节点的ID映射")
        
        # 检查每个节点的元数据中是否引用了其他节点
        circular_refs = []
        node_references = []
        
        for i, node in enumerate(nodes):
            if not hasattr(node, 'metadata') or not node.metadata:
                continue
                
            current_node_id = getattr(node, 'node_id', f'node_{i}')
            
            # 递归检查元数据中的引用
            refs = self._find_node_references_in_object(node.metadata, node_map, current_node_id, path="metadata")
            if refs:
                node_references.extend(refs)
        
        # 分析引用关系
        if node_references:
            for ref in node_references:
                
                # 检查是否是循环引用
                if ref['source'] == ref['target']:
                    circular_refs.append(ref)
                    logger.error(f"🔄 发现自引用循环: {ref['source']} -> {ref['target']}")
        
        if circular_refs:
            logger.error(f"❌ 发现 {len(circular_refs)} 个循环引用!")
            for ref in circular_refs:
                logger.error(f"  🔄 循环引用: {ref['source']} (路径: {ref['path']})")
        else:
            logger.info("✅ 未发现直接的循环引用")
        
        return circular_refs, node_references
    
    def _find_node_references_in_object(self, obj, node_map: dict, source_node_id: str, path: str = "", visited=None):
        """递归查找对象中的节点引用"""
        if visited is None:
            visited = set()
        
        # 防止无限递归
        obj_id = id(obj)
        if obj_id in visited:
            return []
        visited.add(obj_id)
        
        references = []
        
        try:
            # 检查是否是节点对象
            if hasattr(obj, 'node_id') and obj.node_id in node_map:
                references.append({
                    'source': source_node_id,
                    'target': obj.node_id,
                    'path': path,
                    'object_type': type(obj).__name__
                })
            
            # 递归检查字典
            elif isinstance(obj, dict):
                for key, value in obj.items():
                    new_path = f"{path}.{key}" if path else key
                    refs = self._find_node_references_in_object(value, node_map, source_node_id, new_path, visited.copy())
                    references.extend(refs)
            
            # 递归检查列表
            elif isinstance(obj, (list, tuple)):
                for i, item in enumerate(obj):
                    new_path = f"{path}[{i}]" if path else f"[{i}]"
                    refs = self._find_node_references_in_object(item, node_map, source_node_id, new_path, visited.copy())
                    references.extend(refs)
            
            # 检查对象属性
            elif hasattr(obj, '__dict__'):
                for attr_name, attr_value in obj.__dict__.items():
                    new_path = f"{path}.{attr_name}" if path else attr_name
                    refs = self._find_node_references_in_object(attr_value, node_map, source_node_id, new_path, visited.copy())
                    references.extend(refs)
        
        except Exception as e:
            # 忽略检查过程中的错误，避免影响主流程
            pass
        
        return references

    def _save_nodes_text_to_file(self, nodes: List[Any], task_id: str):
        """将所有节点的文本内容保存到本地文件用于调试"""
        
        # 预先检查节点间的循环引用
        circular_refs, node_references = self._detect_global_circular_references(nodes)
        
        try:
            # 创建调试输出目录
            debug_dir = Path("debug_nodes")
            debug_dir.mkdir(exist_ok=True)

            # 生成文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = debug_dir / f"nodes_text_{task_id}_{timestamp}.txt"

            with open(output_file, "w", encoding="utf-8") as f:
                f.write(f"节点文本调试输出\n")
                f.write(f"任务ID: {task_id}\n")
                f.write(f"生成时间: {datetime.now().isoformat()}\n")
                f.write(f"总节点数: {len(nodes)}\n")
                
                # 写入循环引用检测结果
                f.write("\n🔍 循环引用检测结果:\n")
                f.write("-" * 40 + "\n")
                if circular_refs:
                    f.write(f"❌ 发现 {len(circular_refs)} 个循环引用:\n")
                    for ref in circular_refs:
                        f.write(f"  🔄 {ref['source']} -> {ref['target']} (路径: {ref['path']})\n")
                else:
                    f.write("✅ 未发现循环引用\n")
                
                if node_references:
                    f.write(f"\n🔗 节点引用关系 (共 {len(node_references)} 个):\n")
                    for ref in node_references:
                        f.write(f"  - {ref['source']} -> {ref['target']} (路径: {ref['path']}, 类型: {ref['object_type']})\n")
                else:
                    f.write("\n🔗 未发现节点间引用关系\n")
                
                f.write("\n" + "=" * 80 + "\n\n")
                f.flush()  # 强制刷新缓冲区

                for i, node in enumerate(nodes):
                    try:
                        logger.debug(f"📄 处理节点 {i+1}/{len(nodes)}")
                        f.write(f"节点 {i+1}/{len(nodes)}:\n")
                        f.write("-" * 40 + "\n")

                        # 输出节点ID
                        if hasattr(node, "node_id"):
                            f.write(f"节点ID: {node.node_id}\n")
                            logger.debug(f"节点ID: {node.node_id}")
                        else:
                            f.write("节点ID: [无ID属性]\n")
                            logger.debug("节点无ID属性")

                        # 输出节点类型
                        f.write(f"节点类型: {type(node).__name__}\n")
                        logger.debug(f"节点类型: {type(node).__name__}")

                        # 输出元数据（增强循环引用检测）
                        if hasattr(node, "metadata") and node.metadata:
                            try:
                                # 尝试直接序列化
                                metadata_str = json.dumps(node.metadata, ensure_ascii=False, indent=2)
                                f.write(f"元数据: {metadata_str}\n")
                                logger.debug(f"元数据长度: {len(metadata_str)}字符")
                            except Exception as meta_e:
                                f.write(f"元数据: [序列化失败: {meta_e}]\n")
                                
                                # 详细分析元数据结构
                                f.write("🔍 元数据详细分析:\n")
                                
                                try:
                                    # 分析元数据的键值对
                                    f.write(f"  - 元数据类型: {type(node.metadata).__name__}\n")
                                    f.write(f"  - 元数据键数量: {len(node.metadata) if hasattr(node.metadata, '__len__') else 'N/A'}\n")
                                    
                                    if isinstance(node.metadata, dict):
                                        f.write("  - 元数据键列表:\n")
                                        for key in node.metadata.keys():
                                            f.write(f"    * {key}: {type(node.metadata[key]).__name__}\n")
                                            
                                            # 检查每个值是否可序列化
                                            try:
                                                json.dumps(node.metadata[key], ensure_ascii=False)
                                                f.write(f"      ✅ 可序列化\n")
                                            except Exception as key_e:
                                                f.write(f"      ❌ 不可序列化: {key_e}\n")
                                                
                                                # 进一步分析不可序列化的对象
                                                obj = node.metadata[key]
                                                f.write(f"        - 对象类型: {type(obj).__name__}\n")
                                                f.write(f"        - 对象模块: {type(obj).__module__}\n")
                                                
                                                # 检查是否是节点对象
                                                if hasattr(obj, 'node_id'):
                                                    f.write(f"        - 🔗 检测到节点引用: {obj.node_id}\n")
                                                
                                                # 检查是否有循环引用
                                                if obj is node:
                                                    f.write(f"        - 🔄 检测到自引用循环!\n")
                                                    logger.error(f"发现自引用循环: {key}")
                                                elif hasattr(obj, '__dict__'):
                                                    f.write(f"        - 对象属性数量: {len(obj.__dict__)}\n")
                                                    for attr_name, attr_value in obj.__dict__.items():
                                                        if attr_value is node:
                                                            f.write(f"        - 🔄 检测到循环引用: {attr_name} -> 当前节点\n")
                                                            logger.error(f"发现循环引用: {key}.{attr_name} -> 当前节点")
                                                        elif hasattr(attr_value, 'node_id') and hasattr(node, 'node_id') and attr_value.node_id == node.node_id:
                                                            f.write(f"        - 🔄 检测到节点ID循环引用: {attr_name}\n")
                                                            logger.error(f"发现节点ID循环引用: {key}.{attr_name}")
                                    
                                    # 尝试创建安全的元数据副本
                                    f.write("  - 尝试创建安全的元数据副本:\n")
                                    safe_metadata = {}
                                    for key, value in node.metadata.items():
                                        try:
                                            json.dumps(value, ensure_ascii=False)
                                            safe_metadata[key] = value
                                            f.write(f"    ✅ {key}: 已包含\n")
                                        except:
                                            safe_metadata[key] = f"<不可序列化的{type(value).__name__}对象>"
                                            f.write(f"    ⚠️ {key}: 已替换为占位符\n")
                                    
                                    safe_metadata_str = json.dumps(safe_metadata, ensure_ascii=False, indent=2)
                                    f.write(f"  - 安全元数据: {safe_metadata_str}\n")
                                    
                                except Exception as analysis_e:
                                    f.write(f"  - 元数据分析失败: {analysis_e}\n")
                                    logger.error(f"元数据分析失败: {analysis_e}")
                        else:
                            f.write("元数据: [无元数据或为空]\n")
                            logger.debug("节点无元数据")

                        # 输出文本内容
                        text_content = None
                        if hasattr(node, "text"):
                            text_content = node.text if node.text else "[空文本]"
                            f.write(f"文本内容 (长度: {len(text_content)}字符):\n")
                            f.write(f"{text_content}\n")
                            logger.debug(f"文本内容长度: {len(text_content)}字符")
                        elif hasattr(node, "get_content"):
                            try:
                                text_content = node.get_content()
                                text_content = text_content if text_content else "[空文本]"
                                f.write(f"文本内容 (通过get_content获取，长度: {len(text_content)}字符):\n")
                                f.write(f"{text_content}\n")
                                logger.debug(f"通过get_content获取文本，长度: {len(text_content)}字符")
                            except Exception as content_e:
                                f.write(f"文本内容: [get_content调用失败: {content_e}]\n")
                        else:
                            f.write("文本内容: [无文本属性]\n")
                            logger.debug("节点无文本属性")

                        # 输出嵌入向量信息（如果有）
                        if hasattr(node, "embedding") and node.embedding:
                            f.write(f"嵌入向量: 已生成 (维度: {len(node.embedding)})\n")
                            logger.debug(f"嵌入向量维度: {len(node.embedding)}")
                        else:
                            f.write("嵌入向量: 未生成\n")
                            logger.debug("节点无嵌入向量")

                        f.write("\n" + "=" * 80 + "\n\n")
                        
                        # 每10个节点刷新一次缓冲区
                        if (i + 1) % 10 == 0:
                            f.flush()
                            logger.debug(f"已处理 {i+1} 个节点，缓冲区已刷新")
                            
                    except Exception as node_e:
                        error_msg = f"处理节点 {i+1} 时出错: {node_e}"
                        f.write(f"错误: {error_msg}\n")
                        f.write("\n" + "=" * 80 + "\n\n")
                        logger.error(error_msg)
                        continue

                f.flush()  # 最终刷新

            
            # 验证文件是否正确写入
            if output_file.exists():
                file_size = output_file.stat().st_size
            else:
                logger.error("❌ 调试文件未成功创建")

        except Exception as e:
            logger.error(f"❌ 保存节点文本到文件失败: {e}")
            import traceback
            logger.error(f"详细错误信息: {traceback.format_exc()}")

    def _save_index_info_to_db(
        self,
        origin_file_path: str,
        index_id: str,
        index_description: Optional[str] = None,
        document_count: Optional[int] = None,
        node_count: Optional[int] = None,
        vector_dimension: Optional[int] = None,
        processing_config: Optional[Dict[str, Any]] = None,
    ):
        """保存索引信息到数据库"""
        try:

            # 创建数据库会话
            db = SessionLocal()
            try:
                existing_index = IndexDAO.get_index_by_origin_file_path(
                    db, origin_file_path
                )

                if existing_index:
                    # 更新现有索引信息
                    if index_description is not None:
                        existing_index.index_description = index_description
                    if document_count is not None:
                        existing_index.document_count = document_count
                    if node_count is not None:
                        existing_index.node_count = node_count
                    if vector_dimension is not None:
                        existing_index.vector_dimension = vector_dimension
                    if processing_config is not None:
                        existing_index.processing_config = processing_config

                    # 更新索引ID（如果不同）
                    if existing_index.index_id != index_id:
                        existing_index.index_id = index_id

                    db.commit()
                    db.refresh(existing_index)
                    logger.info(
                        f"Updated existing index info for origin_file_path: {origin_file_path}, index ID: {index_id}"
                    )
                else:
                    # 创建新索引信息
                    create_params = {
                        "index_id": index_id,
                        "index_description": index_description,
                        "document_count": document_count,
                        "node_count": node_count,
                        "vector_dimension": vector_dimension,
                        "processing_config": processing_config,
                        "origin_file_path": origin_file_path,
                    }

                    IndexDAO.create_index(db, **create_params)
                    logger.info(f"Created new index info in database: {index_id}")
            finally:
                db.close()
        except Exception as e:
            logger.error(f"Error saving index info to database: {e}")
            raise

    def _generate_auto_description(
        self, index: Any, config: Dict[str, Any]
    ) -> Optional[str]:
        """自动生成索引描述"""
        try:
            from .auto_index_description_generator import AutoIndexDescriptionGenerator

            # 创建描述生成器
            description_generator = AutoIndexDescriptionGenerator(llm=self.llm)

            # 生成描述
            sample_size = config.get("description_sample_size", 50)
            description = description_generator.generate_description(index, sample_size)

            logger.info("Automatic index description generated successfully")
            return description

        except Exception as e:
            logger.error(f"Error generating automatic index description: {str(e)}")
            return None

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务状态"""
        task = self.tasks.get(task_id)
        return task.to_dict() if task else None

    def get_all_tasks(self) -> List[Dict[str, Any]]:
        """获取所有任务状态"""
        return [task.to_dict() for task in self.tasks.values()]

    def create_vector_store_task_from_parse_task(
        self, task_id: str, config: Dict[str, Any]
    ) -> str:
        """从解析任务创建向量存储构建任务"""
        # 获取解析任务信息
        parse_task = self.task_dao.get_parse_task(task_id)
        if not parse_task:
            raise ValueError(f"Parse task not found: {task_id}")

        if parse_task.status != TaskStatus.COMPLETED:
            raise ValueError(
                f"Parse task {task_id} is not completed. Current status: {parse_task.status}"
            )

        # 判断是否为主任务（parent_task_id为None表示主任务）
        is_main_task = parse_task.parent_task_id is None
        output_dir = None

        if is_main_task:
            # 主任务（目录解析）无需校验output_directory
            # 主任务通过子任务来处理具体文件，自身不直接产生输出目录
            logger.info(
                f"Main task {task_id} detected, skipping output_directory validation"
            )
        else:
            # 子任务（单文件解析）基于KNOWLEDGE_BASE_DIR和task_id构建输出目录
            from config import settings
            output_dir = os.path.join(settings.KNOWLEDGE_BASE_DIR, "outputs", task_id)

            # 检查输出目录是否存在
            if not os.path.exists(output_dir):
                raise ValueError(f"Output directory does not exist: {output_dir}")

            logger.info(f"Subtask {task_id} validated, output_directory: {output_dir}")

        # 创建向量存储任务ID
        vector_task_id = str(uuid.uuid4())

        # 创建任务数据
        task_data = {
            "task_id": vector_task_id,
            "parse_task_id": task_id,
            "status": "PENDING",
            "progress": 0,
            "config": config,
            "processed_files": [],
            "total_files": 0,
        }

        # 保存任务到数据库并获取任务对象
        task = self.task_dao.create_vector_store_task(task_data)
        if task:
            self.tasks[vector_task_id] = task

        # 异步执行构建
        asyncio.create_task(self._execute_build_task(task))

        logger.info(
            f"Created vector store build task {vector_task_id} from parse task {task_id}, output dir: {output_dir}"
        )
        return vector_task_id

    def _update_task_in_db(self, task: VectorStoreTask):
        """更新数据库中的任务状态"""
        try:
            update_data = {
                "status": task.status,
                "progress": task.progress,
                "result": task.result,
                "error": task.error,
                "processed_files": task.processed_files,
                "total_files": task.total_files,
                "total_nodes": task.total_nodes,
                "config": task.config,
                "index_id": task.index_id,
            }

            if task.started_at:
                update_data["started_at"] = task.started_at
            if task.completed_at:
                update_data["completed_at"] = task.completed_at

            self.task_dao.update_vector_store_task(task.task_id, update_data)
            logger.debug(f"Vector store task {task.task_id} status updated in database")
        except Exception as e:
            logger.error(
                f"Failed to update vector store task {task.task_id} in database: {e}"
            )

    def cleanup_expired_tasks(self):
        """清理过期任务"""
        current_time = datetime.utcnow()
        expired_tasks = [
            task_id
            for task_id, task in self.tasks.items()
            if task.created_at
            and (current_time - task.created_at).total_seconds()
            > settings.TASK_EXPIRE_TIME
        ]

        for task_id in expired_tasks:
            del self.tasks[task_id]
            logger.info(f"Cleaned up expired vector store task: {task_id}")


# 全局构建器实例
vector_store_builder = VectorStoreBuilder()
