import asyncio
import json
from tabnanny import verbose
import uuid
import time
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor

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

logger = logging.getLogger(__name__)

class VectorStoreTask:
    """向量数据库构建任务"""
    def __init__(self, task_id: str, directory_path: str, config: Dict[str, Any]):
        self.task_id = task_id
        self.directory_path = directory_path
        self.config = config
        self.status = TaskStatus.PENDING
        self.progress = 0
        self.result = None
        self.error = None
        self.created_at = time.time()
        self.started_at = None
        self.completed_at = None
        self.processed_files = []
        self.total_files = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "directory_path": self.directory_path,
            "status": self.status,
            "progress": self.progress,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "processed_files": self.processed_files,
            "total_files": self.total_files,
            "index_description": self.config.get("index_description")
        }

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
            self._initialized = True
            logger.info("VectorStoreBuilder initialized")
    
    def _setup_models(self):
        """设置模型"""
        try:
            # 初始化嵌入模型
            self.embed_model = HuggingFaceEmbedding(
                model_name=settings.EMBED_MODEL_PATH,
                device=settings.GPU_DEVICE if settings.USE_GPU else "cpu",
                trust_remote_code=True
            )
            
            # 初始化LLM（用于摘要和问答生成）
            if settings.LLM_API_BASE and settings.LLM_API_KEY:
                # 使用第三方API
                self.llm = OpenAILike(
                    api_base=settings.LLM_API_BASE,
                    api_key=settings.LLM_API_KEY,
                    api_version=settings.LLM_API_VERSION,
                    model=settings.LLM_MODEL_NAME,
                    max_tokens=settings.LLM_MAX_TOKENS,
                    temperature=settings.LLM_TEMPERATURE,
                    is_chat_model=True,
                )
                logger.info(f"Using external LLM API: {settings.LLM_API_BASE}")
            else:
                # 使用本地模型
                self.llm = HuggingFaceLLM(
                    model_name=settings.LLM_MODEL_PATH,
                    device_map="auto" if settings.USE_GPU else None,
                    model_kwargs={
                        "torch_dtype": "auto",
                        "trust_remote_code": True
                    },
                    tokenizer_kwargs={
                        "trust_remote_code": True
                    },
                    max_new_tokens=settings.LLM_MAX_TOKENS,
                    temperature=settings.LLM_TEMPERATURE
                )
                logger.info(f"Using local LLM: {settings.LLM_MODEL_PATH}")
            
            logger.info("Models initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            raise
    
    async def build_vector_store(
        self, 
        directory_path: str, 
        config: Optional[Dict[str, Any]] = None,
        index_description: Optional[str] = None
    ) -> str:
        """构建向量数据库"""
        
        # 验证目录
        if not Path(directory_path).exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        if not Path(directory_path).is_dir():
            raise ValueError(f"Path is not a directory: {directory_path}")
        
        # 创建任务
        task_id = str(uuid.uuid4())
        task_config = config or {
            "extract_keywords": True,
            "extract_summary": True,
            "generate_qa": True,
            "chunk_size": settings.CHUNK_SIZE,
            "chunk_overlap": settings.CHUNK_OVERLAP
        }
        
        # 添加索引描述
        if index_description:
            task_config["index_description"] = index_description
        
        task = VectorStoreTask(task_id, directory_path, task_config)
        self.tasks[task_id] = task
        
        # 异步执行构建
        asyncio.create_task(self._execute_build_task(task))
        
        logger.info(f"Created vector store build task {task_id} for directory: {directory_path}")
        return task_id
    
    async def _execute_build_task(self, task: VectorStoreTask):
        """执行构建任务"""
        try:
            task.status = TaskStatus.RUNNING
            task.started_at = time.time()
            task.progress = 5
            
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
            task.completed_at = time.time()
            
            logger.info(f"Vector store build task {task.task_id} completed successfully")
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = time.time()
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
                vector_store, index = self._create_vector_store(nodes, task.task_id)
                creation_time = time.time() - start_time
                logger.info(f"Vector store creation completed in {creation_time:.2f}s")
            except Exception as e:
                creation_time = time.time() - start_time
                logger.error(f"Vector store creation failed after {creation_time:.2f}s: {e}")
                raise
            
            # 7. 保存索引
            task.progress = 95
            logger.info(f"Starting index save for task {task.task_id}")
            start_time = time.time()
            
            try:
                index_path = self._save_index(index, task.task_id)
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
                self._save_index_info_to_db(task.task_id, task.config.get("index_description"))
                db_time = time.time() - start_time
                logger.info(f"Index information saved to database in {db_time:.2f}s")
            except Exception as e:
                db_time = time.time() - start_time
                logger.error(f"Failed to save index information to database after {db_time:.2f}s: {e}")
                # 继续执行，不影响索引创建的主流程
            
            result = {
                "index_id": task.task_id,
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
        """收集目录中的文档"""
        documents = []
        directory = Path(task.directory_path)
        
        # 检查是否是knowledge/outputs格式的目录
        if self._is_knowledge_outputs_format(directory):
            return self._collect_from_knowledge_outputs(task, directory)
        else:
            return documents
    
    def _is_knowledge_outputs_format(self, directory: Path) -> bool:
        """检查是否是knowledge/outputs格式的目录"""
        # 检查目录名是否是UUID格式
        try:
            import uuid
            uuid.UUID(directory.name)
            # 检查是否包含content.md和content_list.json文件
            content_file = directory / "content.md"
            content_list_file = directory / "content_list.json"
            return content_file.exists() and content_list_file.exists()
        except (ValueError, AttributeError):
            return False
    
    def _collect_from_knowledge_outputs(self, task: VectorStoreTask, directory: Path) -> List[Document]:
        """从knowledge/outputs格式的目录收集文档"""
        documents = []
        
        try:
            # 读取content.md文件
            content_file = directory / "content.md"
            content_list_file = directory / "content_list.json"
            
            if content_file.exists() and content_list_file.exists():
                # 读取markdown内容
                content = content_file.read_text(encoding='utf-8')
                
                # 读取content_list.json元数据
                import json
                with open(content_list_file, 'r', encoding='utf-8') as f:
                    content_list_data = json.load(f)
                
                if content.strip():
                    # 构建元数据
                    # 只保留对检索有价值的基础元数据
                    metadata = {
                        "file_path": str(content_file),
                        "file_type": ".md"
                    }
                    
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
    
    def _load_metadata(self, file_path: Path) -> Dict[str, Any]:
        """加载对应的JSON元数据文件"""
        metadata = {}
        
        # 查找可能的JSON元数据文件
        json_patterns = [
            f"{file_path.stem}_content_list.json",
            f"{file_path.stem}_middle.json",
            f"{file_path.stem}_metadata.json"
        ]
        
        for pattern in json_patterns:
            json_path = file_path.parent / pattern
            if json_path.exists():
                try:
                    import json
                    with open(json_path, 'r', encoding='utf-8') as f:
                        json_data = json.load(f)
                    
                    # 提取有用的元数据
                    if isinstance(json_data, list) and json_data:
                        # 如果是content_list格式，提取结构化信息
                        text_items = [item for item in json_data if item.get('type') == 'text']
                        if text_items:
                            metadata['total_text_blocks'] = len(text_items)
                            metadata['has_structured_content'] = True
                            
                            # 提取页面信息
                            pages = set(item.get('page_idx', 0) for item in json_data if 'page_idx' in item)
                            metadata['total_pages'] = len(pages)
                            
                            # 提取文本层级信息
                            text_levels = [item.get('text_level') for item in text_items if 'text_level' in item]
                            if text_levels:
                                metadata['has_hierarchical_structure'] = True
                                metadata['max_text_level'] = max(text_levels)
                    
                    elif isinstance(json_data, dict):
                        # 如果是字典格式，直接使用部分字段
                        for key in ['title', 'author', 'subject', 'creator', 'producer']:
                            if key in json_data:
                                metadata[key] = json_data[key]
                    
                    metadata['metadata_source'] = pattern
                    break
                    
                except Exception as e:
                    logger.warning(f"Failed to load metadata from {json_path}: {e}")
                    continue
        
        return metadata
    
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
        
        # 定义中文提示词模板
        chinese_title_node_template = (
            "请根据以下法律条文内容，提取一个简洁准确的标题。请使用与原文相同的语言回答。\n"
            "请提取条文编号和条文主题，格式如：第X条 关于XXX的规定。\n"
            "文本内容：\n"
            "{context_str}\n"
            "标题："
        )
        
        chinese_title_combine_template = (
            "以下是从文档不同部分提取的标题候选：{context_str}\n"
            "请根据这些候选标题，生成一个最能概括整个文档内容的标题。请使用与原文相同的语言回答。\n"
            "最终标题："
        )
        
        chinese_keyword_template = (
            "请从以下法律条文中提取 {max_keywords} 个最重要的关键词。请使用与原文相同的语言回答。\n"
            "重点提取：法律概念、适用范围、责任主体、处罚措施、法律术语等。\n"
            "避免使用停用词。\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "请按以下格式提供关键词：'关键词: <关键词1>, <关键词2>, <关键词3>...'\n"
        )
        
        chinese_summary_template = (
            "请为以下法律条文写一个简洁的摘要。请使用与原文相同的语言回答。\n"
            "请概括条文的适用情形、法律后果、关键要素。\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "摘要："
        )
        
        chinese_qa_template = (
            "请根据以下法律条文内容，生成 {num_questions} 个这段文本可以回答的问题。请使用与原文相同的语言回答。\n"
            "重点生成以下类型问题：什么情况下适用、违反后果是什么、适用主体是谁、如何执行等。\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "请按以下格式提供问题：\n"
            "1. <问题1>\n"
            "2. <问题2>\n"
            "3. <问题3>\n"
            "4. <问题4>\n"
            "5. <问题5>\n"
        )
        
        # 标题提取器
        logger.info("Setting up TitleExtractor with Chinese prompts")
        title_llm = create_logging_llm(self.llm, "TitleExtractor")
        extractors.append(TitleExtractor(
            llm=title_llm,
            node_template=chinese_title_node_template,
            combine_template=chinese_title_combine_template
        ))
        
        # 关键词提取器
        if config.get("extract_keywords", True):
            logger.info("Setting up KeywordExtractor with Chinese prompts (8 keywords for legal docs)")
            keyword_llm = create_logging_llm(self.llm, "KeywordExtractor")
            extractors.append(
                KeywordExtractor(
                    llm=keyword_llm,
                    keywords=5,  # 提取8个关键词（法律文档优化）
                    prompt_template=chinese_keyword_template
                )
            )
        
        # 摘要提取器
        if config.get("extract_summary", True):
            logger.info("Setting up SummaryExtractor with Chinese prompts (self summary for legal docs)")
            summary_llm = create_logging_llm(self.llm, "SummaryExtractor")
            extractors.append(
                SummaryExtractor(
                    llm=summary_llm,
                    summaries=["self"],  # 法律条文独立性强，只提取当前文本摘要
                    prompt_template=chinese_summary_template
                )
            )
        
        # 问答对生成器
        if config.get("generate_qa", True):
            logger.info("Setting up QuestionsAnsweredExtractor with Chinese prompts (5 questions for legal docs)")
            qa_llm = create_logging_llm(self.llm, "QuestionsAnsweredExtractor")
            extractors.append(
                QuestionsAnsweredExtractor(
                    llm=qa_llm,
                    questions=3,  # 生成3问答对（法律文档优化）
                    prompt_template=chinese_qa_template
                )
            )
        
        return extractors
    
    def _create_vector_store(self, nodes: List[Any], index_id: str) -> Tuple[Any, Any]:
        """创建向量存储"""
        import time
        
        if settings.VECTOR_STORE_TYPE == "postgres":
            # 创建PostgreSQL向量存储
            logger.info(f"Creating PostgreSQL vector store with dimension {settings.VECTOR_DIM}")
            start_time = time.time()
            
            # 创建PostgreSQL向量存储构建器
            postgres_builder = create_postgres_vector_store_builder(
                host=settings.POSTGRES_HOST,
                port=settings.POSTGRES_PORT,
                database=settings.POSTGRES_DATABASE,
                user=settings.POSTGRES_USER,
                password=settings.POSTGRES_PASSWORD,
                table_name=f"{settings.POSTGRES_TABLE_NAME}_{index_id.replace('-', '_')}",
                embed_dim=settings.VECTOR_DIM
            )
            
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
            logger.info(f"Vector store index created in {index_time:.2f}s")
            
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
        
        return vector_store, index
    
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
        import json
        from pathlib import Path
        
        # 按文档分组节点
        nodes_by_doc = {}
        for node in nodes:
            if not hasattr(node, 'metadata'):
                continue
                
            # 获取原始文档路径
            doc_path = node.metadata.get('file_path')
            if not doc_path:
                continue
                
            if doc_path not in nodes_by_doc:
                nodes_by_doc[doc_path] = []
            nodes_by_doc[doc_path].append(node)
        
        # 处理每个文档的节点
        for doc_path, doc_nodes in nodes_by_doc.items():
            # 根据文档路径构建content_list.json路径
            doc_file = Path(doc_path)
            content_list_path = doc_file.parent / "content_list.json"
            
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
            "processing_time": time.time() - task.started_at if task.started_at else 0
        }
    
    def _save_index_info_to_db(self, index_id: str, index_description: Optional[str] = None) -> None:
        """保存索引信息到数据库"""
        try:
            from models.database import SessionLocal
            from dao.index_dao import IndexDAO
            
            # 创建数据库会话
            db = SessionLocal()
            try:
                # 检查索引是否已存在
                existing_index = IndexDAO.get_index_by_id(db, index_id)
                
                if existing_index:
                    # 更新索引描述
                    if index_description is not None:
                        IndexDAO.update_index_description(db, index_id, index_description)
                        logger.info(f"Updated description for index: {index_id}")
                else:
                    # 创建新索引信息
                    IndexDAO.create_index(db, index_id, index_description)
                    logger.info(f"Created new index info in database: {index_id}")
            finally:
                db.close()
        except Exception as e:
            logger.error(f"Error saving index info to database: {e}")
            raise
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务状态"""
        task = self.tasks.get(task_id)
        return task.to_dict() if task else None
    
    def get_all_tasks(self) -> List[Dict[str, Any]]:
        """获取所有任务状态"""
        return [task.to_dict() for task in self.tasks.values()]
    
    def cleanup_expired_tasks(self):
        """清理过期任务"""
        current_time = time.time()
        expired_tasks = [
            task_id for task_id, task in self.tasks.items()
            if current_time - task.created_at > settings.TASK_EXPIRE_TIME
        ]
        
        for task_id in expired_tasks:
            del self.tasks[task_id]
            logger.info(f"Cleaned up expired vector store task: {task_id}")

# 全局构建器实例
vector_store_builder = VectorStoreBuilder()