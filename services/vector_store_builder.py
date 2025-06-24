import asyncio
import os
from tabnanny import verbose
import uuid
import time
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter, HTMLNodeParser
from common.custom_markdown_node_parser import CustomMarkdownNodeParser as MarkdownNodeParser
from llama_index.core.extractors import (
    TitleExtractor,
    KeywordExtractor,
    SummaryExtractor,
    QuestionsAnsweredExtractor
)
from llama_index.core.ingestion import IngestionPipeline
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.llms.openai_like import OpenAILike
import faiss

from config import settings
from services.document_parser import document_parser, TaskStatus
from common.postgres_vector_store import create_postgres_vector_store_builder
from dao.task_dao import TaskDAO
from models.task_models import VectorStoreTask
from services.model_client_factory import ModelClientFactory

logger = logging.getLogger(__name__)

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
            self.executor = ThreadPoolExecutor(max_workers=2)  # 限制并发数
            
            # 初始化任务DAO
            self.task_dao = TaskDAO()
            
            self._initialized = True
            logger.info("VectorStoreBuilder initialized")
    
    def _setup_models(self):
        """设置模型"""
        try:
            # 初始化嵌入模型
            self.embed_model = ModelClientFactory.create_embedding_client(settings.embedding_model_settings)
            
            self.llm = ModelClientFactory.create_llm_client(settings.llm_model_settings)
            
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            raise
    
    async def build_vector_store(
        self, 
        task_id: str, 
        config: Optional[Dict[str, Any]] = None
    ) -> str:
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
            
            # 在线程池中执行构建
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor, 
                self._build_vector_store_sync, 
                task
            )
            
            task.result = result
            task.status = TaskStatus.COMPLETED
            task.progress = 100
            task.completed_at = datetime.utcnow()
            
            # 更新数据库中的任务状态
            self._update_task_in_db(task)
            
            logger.info(f"Vector store build task {task.task_id} completed successfully")
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = datetime.utcnow()
            
            # 更新数据库中的任务状态
            self._update_task_in_db(task)
            
            logger.error(f"Vector store build task {task.task_id} failed: {e}")
    
    def _build_vector_store_sync(self, task: VectorStoreTask) -> Dict[str, Any]:
        """同步构建向量数据库"""
        try:
            # 1. 收集文档文件
            task.progress = 10
            documents = self._collect_documents(task)
            task.total_files = len(documents)
            
            if not documents:
                raise ValueError("No valid documents found in directory")
            
            # 2. 按文件类型分组处理文档
            task.progress = 20
            all_nodes = []
            
            # 按文件类型分组
            docs_by_type = {}
            for doc in documents:
                logger.info(doc.metadata)
                file_type = doc.metadata.get('file_type', '.txt')
                if file_type not in docs_by_type:
                    docs_by_type[file_type] = []
                docs_by_type[file_type].append(doc)
            
            # 3. 设置提取器
            task.progress = 30
            extractors = self._setup_extractors(task.config)
            
            # 4. 为每种文件类型创建专用处理管道
            task.progress = 40
            progress_step = 20 / len(docs_by_type)  # 在40-60%之间分配进度
            
            for file_type, type_docs in docs_by_type.items():
                 logger.info(f"Processing {len(type_docs)} {file_type} files")
                 
                 # 获取适合该文件类型的节点解析器
                 logger.info(f"Getting node parser for file type: {file_type}")
                 node_parsers = self._get_node_parser_for_file_type(file_type, task.config)
                 
                 # 确保node_parsers是列表格式
                 if not isinstance(node_parsers, list):
                     node_parsers = [node_parsers]
                 
                 logger.info(f"Node parsers configured: {[type(parser).__name__ for parser in node_parsers]}")
                 logger.info(f"Extractors configured: {[type(extractor).__name__ for extractor in extractors]}")
                 logger.info(f"Embed model: {type(self.embed_model).__name__}")
                 
                 # 创建该文件类型的处理管道
                 logger.info(f"Creating ingestion pipeline for {file_type}")
                 pipeline = IngestionPipeline(
                     transformations=[
                         *node_parsers,
                         *extractors,
                         self.embed_model
                     ]
                 )
                 
                 # 处理该类型的文档
                 import time
                 start_time = time.time()
                 logger.info(f"Starting pipeline processing for {len(type_docs)} {file_type} documents")
                 
                 # 记录每个文档的处理详情
                 for i, doc in enumerate(type_docs):
                     logger.info(f"Document {i+1}/{len(type_docs)}: {doc.metadata.get('file_name', 'unknown')} (size: {len(doc.text)} chars)")
                 
                 try:
                     # 创建自定义的管道来逐步处理并记录每个阶段
                     logger.info("Starting transformation pipeline execution...")
                     
                     # 逐个执行transformation步骤并记录时间
                     current_docs = type_docs
                     for step_idx, transformation in enumerate(pipeline.transformations):
                         step_start = time.time()
                         transformation_name = type(transformation).__name__
                         logger.info(f"\n{'='*60}")
                         logger.info(f"Step {step_idx+1}/{len(pipeline.transformations)}: Starting {transformation_name}")
                         logger.info(f"Input: {len(current_docs) if hasattr(current_docs, '__len__') else 'N/A'} items")
                         
                         # 如果是LLM相关的extractor，记录更详细的信息
                         if 'Extractor' in transformation_name:
                             logger.info(f"🤖 LLM Extractor detected: {transformation_name}")
                             logger.info(f"📝 This step will make LLM API calls - detailed logs will follow")
                             if hasattr(transformation, 'llm'):
                                 logger.info(f"🔧 LLM model: {type(transformation.llm).__name__}")
                         
                         try:
                             if hasattr(transformation, 'transform'):
                                 # 对于node parser和其他transformer
                                 if step_idx == 0:  # 第一步，输入是documents
                                     current_docs = transformation.transform(current_docs)
                                 else:  # 后续步骤，输入是nodes
                                     current_docs = transformation.transform(current_docs)
                             elif hasattr(transformation, '__call__'):
                                 # 对于embedding model等
                                 current_docs = transformation(current_docs)
                             
                             step_time = time.time() - step_start
                             logger.info(f"✅ Step {step_idx+1} ({transformation_name}) completed successfully in {step_time:.2f}s")
                             logger.info(f"📊 Output: {len(current_docs) if hasattr(current_docs, '__len__') else 'N/A'} items")
                             
                             # 如果是LLM相关的extractor，记录成功信息
                             if 'Extractor' in transformation_name:
                                 logger.info(f"🎉 LLM Extractor {transformation_name} completed successfully")
                                 logger.info(f"⏱️  Total LLM processing time: {step_time:.2f}s")
                         
                         except Exception as e:
                             step_time = time.time() - step_start
                             logger.error(f"❌ Step {step_idx+1} ({transformation_name}) FAILED after {step_time:.2f}s")
                             logger.error(f"🚨 Error in {transformation_name}: {str(e)}")
                             if 'Extractor' in transformation_name:
                                 logger.error(f"💥 LLM Extractor {transformation_name} failed - this is likely the timeout source!")
                                 logger.error(f"🔍 Check the LLM API calls above for timeout or connection issues")
                             raise
                         
                         logger.info(f"{'='*60}\n")
                     
                     type_nodes = current_docs
                     processing_time = time.time() - start_time
                     logger.info(f"Pipeline processing completed for {file_type} in {processing_time:.2f}s, generated {len(type_nodes)} nodes")
                     
                     all_nodes.extend(type_nodes)
                     logger.info(f"Total nodes accumulated: {len(all_nodes)}")
                     
                 except Exception as e:
                     processing_time = time.time() - start_time
                     logger.error(f"Pipeline processing failed for {file_type} after {processing_time:.2f}s: {e}")
                     logger.error(f"Error details: {str(e)}")
                     logger.error(f"Error occurred during pipeline execution, last successful step info available in logs above")
                     raise
                 
                 task.progress += progress_step
                 logger.info(f"Progress updated to {task.progress:.1f}%")
            
            # 5. 合并所有节点
            task.progress = 60
            nodes = all_nodes
            logger.info(f"Merging nodes completed, total nodes: {len(nodes)}")
            
            # 5.5 为节点添加页码信息
            nodes = self._map_page_info_to_nodes(nodes)
            logger.info(f"Page information mapped to nodes, total nodes: {len(nodes)}")
            
            # 6. 创建向量存储
            task.progress = 80
            logger.info(f"Starting vector store creation with {len(nodes)} nodes")
            start_time = time.time()
            
            try:
                # 提取文件信息用于向量存储创建
                file_info = self._extract_file_info_from_task(task, documents)
                vector_store, index, actual_index_id = self._create_vector_store(nodes, task.task_id, file_info)
                creation_time = time.time() - start_time
                logger.info(f"Vector store creation completed in {creation_time:.2f}s, using index_id: {actual_index_id}")
            except Exception as e:
                creation_time = time.time() - start_time
                logger.error(f"Vector store creation failed after {creation_time:.2f}s: {e}")
                raise
            
            # 7. 保存索引
            task.progress = 95
            logger.info(f"Starting index save for task {task.task_id}, using index_id: {actual_index_id}")
            start_time = time.time()
            
            try:
                index_path = self._save_index(index, actual_index_id)
                save_time = time.time() - start_time
                logger.info(f"Index saved successfully in {save_time:.2f}s to: {index_path}")
            except Exception as e:
                save_time = time.time() - start_time
                logger.error(f"Index save failed after {save_time:.2f}s: {e}")
                raise
            
            # 8. 生成统计信息
            logger.info("Generating statistics")
            start_time = time.time()
            stats = self._generate_stats(documents, nodes, task)
            stats_time = time.time() - start_time
            logger.info(f"Statistics generated in {stats_time:.2f}s")
            
            # 9. 保存索引信息到数据库
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
                
                # 收集文件信息用于保存到数据库
                file_info = self._extract_file_info_from_task(task, documents)
                self._save_index_info_to_db(
                    actual_index_id, 
                    index_description,
                    file_info=file_info,
                    document_count=len(documents),
                    node_count=len(nodes),
                    vector_dimension=settings.VECTOR_DIM,
                    processing_config=task.config
                )
                db_time = time.time() - start_time
                logger.info(f"Index information saved to database in {db_time:.2f}s")
            except Exception as e:
                db_time = time.time() - start_time
                logger.error(f"Failed to save index information to database after {db_time:.2f}s: {e}")
                # 继续执行，不影响索引创建的主流程
            
            result = {
                "index_id": actual_index_id,
                "index_path": index_path,
                "document_count": len(documents),
                "node_count": len(nodes),
                "vector_dimension": settings.VECTOR_DIM,
                "stats": stats,
                "config": task.config
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error building vector store: {e}")
            raise
    
    def _collect_documents(self, task: VectorStoreTask) -> List[Document]:
        """
        收集目录中的文档
        - 目录主任务：递归收集子任务文档，不校验output_directory
        - 单文件解析任务：必须有output_directory
        - 针对目录中的html文件，直接转为Document对象
        """
        documents = []
        # 从解析任务获取输出目录 
        parse_task = self.task_dao.get_parse_task(task.parse_task_id)
        if not parse_task:
            raise ValueError(f"Parse task not found: {task.parse_task_id}")

        is_directory = bool(parse_task.config and parse_task.config.get("is_directory"))
        if is_directory:
            # 目录主任务：递归收集子任务文档，不校验output_directory
            return self._collect_documents_from_directory_parse_task(task, parse_task)
        else:
            # 单文件解析任务：必须有output_directory
            directory_path = None
            if parse_task.output_directory:
                directory_path = parse_task.output_directory
            elif parse_task.result and parse_task.result.get('output_directory'):
                directory_path = parse_task.result.get('output_directory')
            if not directory_path:
                raise ValueError(f"No output directory found for parse task {task.parse_task_id}")
            directory = Path(directory_path)
            # 优先处理knowledge/outputs格式
            if self._is_knowledge_outputs_format(directory):
                return self._collect_from_knowledge_outputs(task, directory)
            # 新增：收集html文件
            html_docs = self._collect_html_documents_from_dir(task, directory)
            if html_docs:
                return html_docs
            return documents
    
    def _is_knowledge_outputs_format(self, directory: Path) -> bool:
        """检查是否是knowledge/outputs格式的目录"""
        # 检查目录名是否是UUID格式
        try:
            import uuid
            uuid.UUID(directory.name)
            # 检查是否包含markdown文件和content_list.json文件
            # 查找所有.md文件，因为我们不知道原始文件名
            md_files = list(directory.glob('*.md'))
            json_files = list(directory.glob('*_content_list.json'))
            return len(md_files) > 0 and len(json_files) > 0
        except (ValueError, AttributeError):
            return False
    
    def _get_original_file_stem(self, task: VectorStoreTask, directory: Path) -> str:
        """获取原始文档的文件名（不含扩展名）"""
        # 首先尝试从关联的解析任务获取原始文件名
        if task.parse_task_id:
            try:
                from .document_parser import document_parser
                parse_task = document_parser.get_task_status(task.parse_task_id)
                if parse_task and parse_task.get('file_path'):
                    original_file_path = parse_task['file_path']
                    return Path(original_file_path).stem
            except Exception as e:
                logger.warning(f"无法从解析任务获取原始文件名: {e}")
        
        # 如果无法从解析任务获取，则查找目录中的.md文件
        md_files = list(directory.glob('*.md'))
        if md_files:
            # 返回第一个找到的.md文件的文件名（不含扩展名）
            return md_files[0].stem
        
        # 最后回退到使用目录名
        return directory.name


    def _collect_html_documents_from_dir(self, task: VectorStoreTask, directory: Path) -> list:
        """收集目录下所有html/htm文件，转为Document对象"""
        documents = []
        html_files = list(directory.glob('*.html')) + list(directory.glob('*.htm'))
        for html_file in html_files:
            try:
                content = html_file.read_text(encoding='utf-8')
                metadata = {
                    # "file_path": str(html_file),  # 移除 file_path，避免元数据过大
                    "file_type": ".html" if html_file.suffix == ".html" else ".htm",
                    "source_file": html_file.stem
                }
                # 尝试补充 original_file_path/original_file_name
                if task.parse_task_id:
                    from .document_parser import document_parser
                    parse_task = document_parser.get_task_status(task.parse_task_id)
                    if parse_task:
                        original_file_path = parse_task.get('file_path')
                        if original_file_path:
                            metadata['original_file_path'] = original_file_path
                            metadata['original_file_name'] = Path(original_file_path).name
                doc = Document(text=content, metadata=metadata)
                documents.append(doc)
                task.processed_files.append(str(html_file))
            except Exception as e:
                logger.warning(f"Failed to process html file {html_file}: {e}")
        return documents
    
    def _collect_documents_from_directory_parse_task(self, task: VectorStoreTask, parse_task: Any) -> List[Document]:
        """
        从目录解析任务的子任务中收集文档。
        - 优先 knowledge/outputs 格式
        - 否则收集 html 文件
        - 统一复用 _collect_from_knowledge_outputs 和 _collect_html_documents_from_dir
        """
        documents = []
        logger.info(f"Collecting documents from directory parse task: {parse_task.task_id}")
        if not parse_task.config or "subtasks" not in parse_task.config:
            logger.warning(f"Directory parse task {parse_task.task_id} has no subtasks defined.")
            return []
        subtasks_info = parse_task.config["subtasks"]
        for subtask_info in subtasks_info:
            subtask_id = subtask_info.get("task_id")
            if not subtask_id:
                logger.warning(f"Subtask info missing task_id: {subtask_info}")
                continue
            sub_parse_task = self.task_dao.get_parse_task(subtask_id)
            if not (sub_parse_task and sub_parse_task.status == TaskStatus.COMPLETED):
                logger.warning(f"Subtask {subtask_id} is not completed or not found. Status: {sub_parse_task.status if sub_parse_task else 'Not Found'}")
                continue
            sub_directory_path = sub_parse_task.output_directory or (sub_parse_task.result and sub_parse_task.result.get('output_directory'))
            if not sub_directory_path:
                logger.warning(f"No output directory found for subtask {subtask_id}.")
                continue
            sub_directory = Path(sub_directory_path)
            # 优先 knowledge/outputs 格式
            if self._is_knowledge_outputs_format(sub_directory):
                logger.info(f"Collecting documents from subtask output directory: {sub_directory_path}")
                documents.extend(self._collect_from_knowledge_outputs(task, sub_directory))
            else:
                # 新增：收集html文件
                html_docs = self._collect_html_documents_from_dir(task, sub_directory)
                if html_docs:
                    logger.info(f"Collecting html documents from subtask output directory: {sub_directory_path}")
                    documents.extend(html_docs)
                else:
                    logger.warning(f"Subtask output directory {sub_directory_path} is not in expected format and contains no html files.")
        logger.info(f"Finished collecting documents from directory parse task. Total documents: {len(documents)}")
        return documents
    
    def _collect_from_knowledge_outputs(self, task: VectorStoreTask, directory: Path) -> List[Document]:
        """从knowledge/outputs格式的目录收集文档"""
        documents = []
        
        try:
            # 获取原始文档名称
            base_name = self._get_original_file_stem(task, directory)
            content_file = directory / f"{base_name}.md"
            content_list_file = directory / f"{base_name}_content_list.json"
            
            if content_file.exists() and content_list_file.exists():
                # 读取markdown内容
                content = content_file.read_text(encoding='utf-8')
                
                # 读取content_list.json元数据
                with open(content_list_file, 'r', encoding='utf-8') as f:
                    content_list_data = json.load(f)
                
                if content.strip():
                    # 构建基础元数据
                    metadata = {
                        # "file_path": str(content_file),  # 移除 file_path，避免元数据过大
                        "file_type": ".md",
                        "source_file": base_name  # 原始文件名
                    }
                    
                    # 如果有关联的解析任务，添加更多信息
                    if task.parse_task_id:
                        from .document_parser import document_parser
                        parse_task = document_parser.get_task_status(task.parse_task_id)
                        if parse_task:
                            # 添加原始文件信息
                            original_file_path = parse_task.get('file_path')
                            if original_file_path:
                                metadata['original_file_path'] = original_file_path
                                metadata['original_file_name'] = Path(original_file_path).name
                            
                            # 添加文件信息
                            file_info = parse_task.get('file_info', {})
                            if file_info:
                                metadata['file_size'] = file_info.get('size')
                                metadata['file_extension'] = file_info.get('extension')
                                metadata['mime_type'] = file_info.get('mime_type')
                            
                            # 添加解析器类型
                            parser_type = parse_task.get('parser_type')
                            if parser_type:
                                metadata['parser_type'] = parser_type
                    
                    # 从content_list.json提取有用信息
                    if isinstance(content_list_data, list) and content_list_data:
                        # 提取页码信息
                        page_indices = set()
                        for item in content_list_data:
                            if 'page_idx' in item:
                                page_indices.add(item['page_idx'])
                        
                        # 只在有页码信息时才添加页码相关元数据
                        if page_indices:
                            metadata['page_count'] = len(page_indices)
                            metadata['first_page'] = min(page_indices)
                            metadata['last_page'] = max(page_indices)
                        
                        # 只在有标题时才添加标题元数据
                        title_items = [item for item in content_list_data if item.get('type') == 'text' and item.get('text_level') == 1]
                        if title_items and title_items[0].get('text'):
                            metadata['title'] = title_items[0]['text']
                        
                        # 提取表格和图片信息
                        tables = [item for item in content_list_data if item.get('type') == 'table']
                        images = [item for item in content_list_data if item.get('type') == 'image']
                        
                        # 只在有表格或图片时才添加相关元数据
                        if tables:
                            metadata['has_tables'] = True
                            metadata['tables_count'] = len(tables)
                        
                        if images:
                            metadata['has_images'] = True
                            metadata['images_count'] = len(images)
                    
                    doc = Document(
                        text=content,
                        metadata=metadata
                    )
                    documents.append(doc)
                    task.processed_files.append(str(content_file))
                    
                    logger.info(f"Successfully processed knowledge output: {directory.name}")
                    
        except Exception as e:
            logger.warning(f"Failed to process knowledge output directory {directory}: {e}")
        
        return documents
    
    def _get_node_parser_for_file_type(self, file_type: str, config: Dict[str, Any]):
        """根据文件类型获取合适的节点解析器"""
        chunk_size = config.get("chunk_size", settings.CHUNK_SIZE)
        chunk_overlap = config.get("chunk_overlap", settings.CHUNK_OVERLAP)
        
        if file_type == ".md":
            # Markdown文件使用MarkdownNodeParser，然后配合SentenceSplitter
            # MarkdownNodeParser主要用于解析markdown结构，然后用SentenceSplitter进行分块
            return [
                MarkdownNodeParser.from_defaults(),
                SentenceSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
            ]
        elif file_type in [".html", ".htm"]:
            # HTML文件使用HTMLNodeParser，然后配合SentenceSplitter
            return [
                HTMLNodeParser.from_defaults(),
                SentenceSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
            ]
        else:
            # 文本文件直接使用SentenceSplitter
            return SentenceSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
    
    def _setup_extractors(self, config: Dict[str, Any]) -> List[Any]:
        """设置提取器"""
        extractors = []
        
        # 使用新的智能元数据提取器
        logger.info("Using SmartMetadataExtractor with intelligent chunk-level processing")
        
        from .smart_metadata_extractor import SmartMetadataExtractor
        
        # 创建带有详细日志记录的LLM包装器
        def create_logging_llm(llm, extractor_name):
            """
            创建带有详细日志记录的LLM包装器，使用LlamaIndex的CallbackManager
            """
            from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
            import llama_index.core
            
            # 设置全局调试处理器以获取详细的LLM调用信息
            llama_index.core.set_global_handler("simple")
            
            # 创建自定义的调试处理器
            debug_handler = LlamaDebugHandler(print_trace_on_end=True)
            callback_manager = CallbackManager([debug_handler])
            
            # 为LLM设置callback manager
            if hasattr(llm, 'callback_manager'):
                llm.callback_manager = callback_manager
            
            logger.info(f"[{extractor_name}] LLM logging configured for: {type(llm).__name__} with detailed request/response tracking")
            return llm
        
        smart_llm = create_logging_llm(self.llm, "SmartMetadataExtractor")
        
        # 确定提取模式
        extract_mode = config.get("extract_mode", "enhanced")
        
        # 添加智能元数据提取器
        extractors.append(SmartMetadataExtractor(
            llm=smart_llm,
            min_chunk_size_for_summary=config.get("min_chunk_size_for_summary", 500),
            min_chunk_size_for_qa=config.get("min_chunk_size_for_qa", 300),
            max_keywords=config.get("max_keywords", 5),
            num_questions=config.get("num_questions", 3),
            show_progress=True,
            extract_mode=extract_mode
        ))
        
        logger.info(f"SmartMetadataExtractor configured successfully with mode={extract_mode}")
        logger.info(f"Configuration: min_summary_size={config.get('min_chunk_size_for_summary', 500)}, min_qa_size={config.get('min_chunk_size_for_qa', 300)}")
        
        return extractors
    
    def _determine_vector_store_strategy(self, current_index_id: str, file_info: Optional[Dict[str, Any]] = None) -> Tuple[str, bool]:
        """确定向量存储策略：返回目标索引ID和是否需要更新
        
        Returns:
            Tuple[str, bool]: (target_index_id, should_update)
        """
        if file_info and file_info.get('file_md5'):
            # 根据文件MD5检查是否已存在相同文件的索引
            try:
                from models.database import SessionLocal
                from dao.index_dao import IndexDAO
                
                db = SessionLocal()
                try:
                    existing_index = IndexDAO.get_index_by_file_md5(db, file_info.get('file_md5'))
                    if existing_index:
                        # 找到现有索引，使用现有的index_id和对应的表
                        target_index_id = existing_index.index_id
                        logger.info(f"Found existing index for file MD5: {file_info.get('file_md5')}, using existing index_id: {target_index_id}")
                        return target_index_id, True
                    else:
                        # 没有找到现有索引，使用当前index_id创建新的
                        logger.info(f"No existing index found for file MD5: {file_info.get('file_md5')}, creating new index with id: {current_index_id}")
                        return current_index_id, False
                finally:
                    db.close()
            except Exception as e:
                logger.warning(f"Error checking existing index by MD5: {e}, using current index_id: {current_index_id}")
                return current_index_id, False
        else:
            # 如果没有文件信息，使用当前index_id
            logger.info(f"No file MD5 available, using current index_id: {current_index_id}")
            return current_index_id, False
    
    def _create_vector_store(self, nodes: List[Any], index_id: str, file_info: Optional[Dict[str, Any]] = None) -> Tuple[Any, Any, str]:
        """创建向量存储
        
        Returns:
            Tuple[Any, Any, str]: (vector_store, index, actual_index_id)
        """
        import time
        
        if settings.VECTOR_STORE_TYPE == "postgres":
            # 创建PostgreSQL向量存储
            logger.info(f"Creating PostgreSQL vector store with dimension {settings.VECTOR_DIM}")
            start_time = time.time()
            
            # 检查是否存在相同文件的索引，并确定使用的索引ID和表名
            target_index_id, should_update = self._determine_vector_store_strategy(index_id, file_info)
            
            # 创建PostgreSQL向量存储构建器（使用确定的索引ID）
            postgres_builder = create_postgres_vector_store_builder(
                host=settings.POSTGRES_HOST,
                port=settings.POSTGRES_PORT,
                database=settings.POSTGRES_DATABASE,
                user=settings.POSTGRES_USER,
                password=settings.POSTGRES_PASSWORD,
                table_name=f"{settings.POSTGRES_TABLE_NAME}_{target_index_id.replace('-', '_')}",
                embed_dim=settings.VECTOR_DIM
            )
            
            if should_update:
                logger.info(f"Updating existing vector store with new data")
                # 使用更新方法
                index = postgres_builder.update_index_with_nodes(nodes, self.embed_model)
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
                    embed_model=self.embed_model
                )
            
            index_time = time.time() - start_time
            logger.info(f"Vector store index processed in {index_time:.2f}s")
            
            # 返回实际使用的索引ID
            return vector_store, index, target_index_id
            
        else:
            # 创建FAISS索引（默认行为）
            logger.info(f"Creating FAISS index with dimension {settings.VECTOR_DIM}")
            start_time = time.time()
            faiss_index = faiss.IndexFlatL2(settings.VECTOR_DIM)
            faiss_time = time.time() - start_time
            logger.info(f"FAISS index created in {faiss_time:.3f}s")
            
            # 创建向量存储
            logger.info("Creating FAISS vector store")
            start_time = time.time()
            vector_store = FaissVectorStore(faiss_index=faiss_index)
            store_time = time.time() - start_time
            logger.info(f"Vector store created in {store_time:.3f}s")
            
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
                embed_model=self.embed_model
            )
            index_time = time.time() - start_time
            logger.info(f"Vector store index created in {index_time:.2f}s")
            
            # 对于FAISS，使用传入的index_id
            return vector_store, index, index_id
    
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
        
        logger.info(f"Index persistence completed in {persist_time:.2f}s to {index_dir}")
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
            if not hasattr(node, 'metadata'):
                continue
            
            # 优先用 original_file_path 分组，否则 fallback 到 file_path
            doc_path = node.metadata.get('original_file_path') or node.metadata.get('file_path')
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
                
            # 读取content_list.json
            try:
                with open(content_list_path, 'r', encoding='utf-8') as f:
                    content_list = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load content_list from {content_list_path}: {e}")
                continue
                
            if not isinstance(content_list, list) or not content_list:
                continue
                
            # 创建文本到页码的映射
            text_to_page = {}
            for item in content_list:
                if item.get('type') == 'text' and 'text' in item and 'page_idx' in item:
                    text = item['text']
                    page_idx = item['page_idx']
                    text_to_page[text] = page_idx
            
            # 为每个节点分配页码
            for node in doc_nodes:
                if not hasattr(node, 'text') or not node.text:
                    continue
                
                # 尝试直接匹配
                if node.text in text_to_page:
                    node.metadata['page_idx'] = text_to_page[node.text]
                    continue
                    
                # 尝试部分匹配（查找节点文本中包含的最长content_list文本）
                best_match = None
                best_match_length = 0
                for text, page_idx in text_to_page.items():
                    if text in node.text and len(text) > best_match_length:
                        best_match = text
                        best_match_length = len(text)
                
                if best_match:
                    node.metadata['page_idx'] = text_to_page[best_match]
                    continue
                
                # 如果没有匹配，尝试反向匹配（查找包含节点文本的最短content_list文本）
                best_match = None
                best_match_length = float('inf')
                for text, page_idx in text_to_page.items():
                    if node.text in text and len(text) < best_match_length:
                        best_match = text
                        best_match_length = len(text)
                
                if best_match:
                    node.metadata['page_idx'] = text_to_page[best_match]
                    
            # 统计添加了页码的节点数量
            nodes_with_page = sum(1 for node in doc_nodes if 'page_idx' in node.metadata)
            logger.info(f"Added page information to {nodes_with_page}/{len(doc_nodes)} nodes for document: {doc_path}")
        
        return nodes
    
    def _generate_stats(self, documents: List[Document], nodes: List[Any], task: VectorStoreTask) -> Dict[str, Any]:
        """生成统计信息"""
        total_chars = sum(len(doc.text) for doc in documents)
        avg_chunk_size = sum(len(getattr(node, 'text', '')) for node in nodes) / len(nodes) if nodes else 0
        
        # 统计文件类型
        file_types = {}
        for file_path in task.processed_files:
            ext = Path(file_path).suffix.lower()
            file_types[ext] = file_types.get(ext, 0) + 1
        
        # 统计有页码信息的节点数量
        nodes_with_page = sum(1 for node in nodes if hasattr(node, 'metadata') and 'page_idx' in node.metadata)
        
        return {
            "total_characters": total_chars,
            "average_chunk_size": avg_chunk_size,
            "file_types": file_types,
            "nodes_with_page_info": nodes_with_page,
            "nodes_with_page_percentage": f"{nodes_with_page/len(nodes)*100:.2f}%" if nodes else "0%",
            "processing_time": time.time() - task.started_at.timestamp() if task.started_at else 0
        }
    
    def _save_index_info_to_db(
        self, 
        index_id: str, 
        index_description: Optional[str] = None,
        file_info: Optional[Dict[str, Any]] = None,
        document_count: Optional[int] = None,
        node_count: Optional[int] = None,
        vector_dimension: Optional[int] = None,
        processing_config: Optional[Dict[str, Any]] = None
    ):
        """保存索引信息到数据库"""
        try:
            from models.database import SessionLocal
            from dao.index_dao import IndexDAO
            
            # 创建数据库会话
            db = SessionLocal()
            try:
                existing_index = None
                
                # 如果有文件信息，优先根据文件MD5查找现有索引
                if file_info and file_info.get('file_md5'):
                    existing_index = IndexDAO.get_index_by_file_md5(db, file_info.get('file_md5'))
                    
                # 如果根据MD5没找到，再根据索引ID查找
                if not existing_index:
                    existing_index = IndexDAO.get_index_by_id(db, index_id)
                
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
                    logger.info(f"Updated existing index info for file MD5: {file_info.get('file_md5') if file_info else 'N/A'}, index ID: {index_id}")
                else:
                    # 创建新索引信息
                    create_params = {
                        'index_id': index_id,
                        'index_description': index_description,
                        'document_count': document_count,
                        'node_count': node_count,
                        'vector_dimension': vector_dimension,
                        'processing_config': processing_config
                    }
                    
                    # 添加文件信息（如果有）
                    if file_info:
                        create_params.update({
                            'file_md5': file_info.get('file_md5'),
                            'file_path': file_info.get('file_path'),
                            'file_name': file_info.get('file_name'),
                            'file_size': file_info.get('file_size'),
                            'file_extension': file_info.get('file_extension'),
                            'mime_type': file_info.get('mime_type')
                        })
                    
                    IndexDAO.create_index(db, **create_params)
                    logger.info(f"Created new index info in database: {index_id}")
            finally:
                db.close()
        except Exception as e:
            logger.error(f"Error saving index info to database: {e}")
            raise
    
    def _extract_file_info_from_task(self, task: VectorStoreTask, documents: List[Any]) -> Optional[Dict[str, Any]]:
        """从任务和文档中提取文件信息"""
        try:
            import hashlib
            from pathlib import Path
            
            # 尝试从解析任务获取原始文件信息
            if task.parse_task_id:
                from .document_parser import document_parser
                parse_task = document_parser.get_task_status(task.parse_task_id)
                if parse_task:
                    original_file_path = parse_task.get('file_path')
                    if original_file_path and Path(original_file_path).exists():
                        file_path = Path(original_file_path)
                        
                        # 计算文件MD5
                        file_md5 = None
                        try:
                            with open(file_path, 'rb') as f:
                                file_content = f.read()
                                file_md5 = hashlib.md5(file_content).hexdigest()
                        except Exception as e:
                            logger.warning(f"Failed to calculate MD5 for {file_path}: {e}")
                        
                        # 获取文件信息
                        file_info = parse_task.get('file_info', {})
                        
                        return {
                            'file_md5': file_md5,
                            'file_path': str(file_path),
                            'file_name': file_path.name,
                            'file_size': file_info.get('size') or file_path.stat().st_size,
                            'file_extension': file_path.suffix,
                            'mime_type': file_info.get('mime_type')
                        }
            
            # 如果无法从解析任务获取，尝试从文档元数据获取
            if documents:
                first_doc = documents[0]
                if hasattr(first_doc, 'metadata') and first_doc.metadata:
                    metadata = first_doc.metadata
                    original_file_path = metadata.get('original_file_path')
                    if original_file_path and Path(original_file_path).exists():
                        file_path = Path(original_file_path)
                        
                        # 计算文件MD5
                        file_md5 = None
                        try:
                            with open(file_path, 'rb') as f:
                                file_content = f.read()
                                file_md5 = hashlib.md5(file_content).hexdigest()
                        except Exception as e:
                            logger.warning(f"Failed to calculate MD5 for {file_path}: {e}")
                        
                        return {
                            'file_md5': file_md5,
                            'file_path': str(file_path),
                            'file_name': metadata.get('original_file_name') or file_path.name,
                            'file_size': metadata.get('file_size'),
                            'file_extension': metadata.get('file_extension') or file_path.suffix,
                            'mime_type': metadata.get('mime_type')
                        }
            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting file info from task: {e}")
            return None
    
    def _generate_auto_description(self, index: Any, config: Dict[str, Any]) -> Optional[str]:
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
    
    def create_vector_store_task_from_parse_task(self, task_id: str, config: Dict[str, Any] = None) -> str:
        """从解析任务创建向量存储构建任务"""
        # 获取解析任务信息
        parse_task = self.task_dao.get_parse_task(task_id)
        if not parse_task:
            raise ValueError(f"Parse task not found: {task_id}")
        
        if parse_task.status != 'COMPLETED':
            raise ValueError(f"Parse task {task_id} is not completed. Current status: {parse_task.status}")
        
        # 判断是否为目录解析主任务
        is_directory = bool(parse_task.config and parse_task.config.get("is_directory"))
        output_dir = None
        if not is_directory:
            # 单文件解析任务必须有output_directory
            if parse_task.output_directory:
                output_dir = parse_task.output_directory
            elif parse_task.result and parse_task.result.get('output_directory'):
                output_dir = parse_task.result.get('output_directory')
            if not output_dir:
                raise ValueError(f"No output directory found for parse task {task_id}")
            # 检查输出目录是否存在
            if not os.path.exists(output_dir):
                raise ValueError(f"Output directory does not exist: {output_dir}")
        # 目录主任务无需校验output_directory
        
        # 创建向量存储任务ID
        vector_task_id = str(uuid.uuid4())
        
        # 设置默认配置
        task_config = config or {
            "extract_keywords": True,
            "extract_summary": True,
            "generate_qa": True,
            "chunk_size": settings.CHUNK_SIZE,
            "chunk_overlap": settings.CHUNK_OVERLAP
        }
        
        # 添加解析任务关联
        task_config["parse_task_id"] = task_id
        
        # 创建任务数据
        task_data = {
            'task_id': vector_task_id,
            'parse_task_id': task_id,
            'status': 'PENDING',
            'progress': 0,
            'config': task_config,
            'processed_files': [],
            'total_files': 0
        }
        
        # 保存任务到数据库并获取任务对象
        task = self.task_dao.create_vector_store_task(task_data)
        if task:
            self.tasks[vector_task_id] = task
        
        # 异步执行构建
        asyncio.create_task(self._execute_build_task(task))
        
        logger.info(f"Created vector store build task {vector_task_id} from parse task {task_id}, output dir: {output_dir}")
        return vector_task_id
    
    def _update_task_in_db(self, task: VectorStoreTask):
        """更新数据库中的任务状态"""
        try:
            update_data = {
                'status': task.status,
                'progress': task.progress,
                'result': task.result,
                'error': task.error,
                'processed_files': task.processed_files,
                'total_files': task.total_files
            }
            
            if task.started_at:
                update_data['started_at'] = task.started_at
            if task.completed_at:
                update_data['completed_at'] = task.completed_at
            
            # 如果任务完成，保存索引信息
            if task.status == TaskStatus.COMPLETED and task.result:
                update_data['index_id'] = task.result.get('index_id')
                update_data['total_documents'] = task.result.get('stats', {}).get('total_documents', 0)
                update_data['total_nodes'] = task.result.get('stats', {}).get('total_nodes', 0)
                
            self.task_dao.update_vector_store_task(task.task_id, update_data)
            logger.debug(f"Vector store task {task.task_id} status updated in database")
        except Exception as e:
            logger.error(f"Failed to update vector store task {task.task_id} in database: {e}")
    
    def cleanup_expired_tasks(self):
        """清理过期任务"""
        current_time = datetime.utcnow()
        expired_tasks = [
            task_id for task_id, task in self.tasks.items()
            if task.created_at and (current_time - task.created_at).total_seconds() > settings.TASK_EXPIRE_TIME
        ]
        
        for task_id in expired_tasks:
            del self.tasks[task_id]
            logger.info(f"Cleaned up expired vector store task: {task_id}")

# 全局构建器实例
vector_store_builder = VectorStoreBuilder()