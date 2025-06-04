import os
from typing import Optional, List
from pydantic import Field
from pydantic_settings import BaseSettings
from pathlib import Path

# 设置Hugging Face离线模式环境变量
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

os.environ["EASYOCR_MODULE_PATH"] = "/home/ubuntu/.EasyOCR/model"
os.environ["EASYOCR_DOWNLOAD"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

class Settings(BaseSettings):
    """应用配置"""
    
    # 基础路径配置
    BASE_DIR: Path = Path(__file__).parent
    MODEL_BASE_DIR: str = os.getenv("MODEL_BASE_DIR", "/home/ubuntu/workspace/models_dir/models/")
    KNOWLEDGE_BASE_DIR: str = os.getenv("KNOWLEDGE_BASE_DIR", "/home/ubuntu/workspace/knowledge/")
    
    # 服务配置
    DOC_PARSE_HOST: str = "0.0.0.0"
    DOC_PARSE_PORT: int = 8001
    RAG_SERVICE_HOST: str = "0.0.0.0"
    RAG_SERVICE_PORT: int = 8002
    
    # 模型路径配置
    EMBED_MODEL_PATH: str = Field(default_factory=lambda: os.path.join(os.getenv("MODEL_BASE_DIR", "/home/ubuntu/workspace/models_dir/models/"), "BAAI_bge-large-zh-v1.5"))
    RERANK_MODEL_PATH: str = Field(default_factory=lambda: os.path.join(os.getenv("MODEL_BASE_DIR", "/home/ubuntu/workspace/models_dir/models/"), "BAAI_bge-reranker-v2-m3"))
    LLM_MODEL_PATH: str = Field(default_factory=lambda: os.path.join(os.getenv("MODEL_BASE_DIR", "/home/ubuntu/workspace/models_dir/test_models/"), "hf-internal-testing_tiny-random-bert"))
    DOCLING_MODEL_PATH: str = Field(default_factory=lambda: os.path.join(os.getenv("MODEL_BASE_DIR", "/home/ubuntu/workspace/models_dir/models/"), "docling-models"))
    EASY_ORC_MODEL_PATH: str = Field(default_factory=lambda: os.path.join(os.getenv("MODEL_BASE_DIR", "/home/ubuntu/workspace/models_dir/models/"), "easyocr"))
    
    # GPU配置
    USE_GPU: bool = True
    GPU_DEVICE: str = "cuda:0"
    
    # 文档解析配置
    MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 100MB
    SUPPORTED_FORMATS: List[str] = [".pdf", ".docx", ".doc", ".txt", ".md", ".html", ".pptx", ".xlsx"]
    PARSE_TIMEOUT: int = 300  # 5分钟
    
    # 解析器类型配置
    DEFAULT_PARSER_TYPE: str = "docling"  # 默认解析器类型: "docling" 或 "mineru"
    
    MINERU_MAX_WORKERS: int = 1  # MinerU并发处理数量
    
    # OCR配置
    OCR_ENABLED: bool = True
    OCR_LANGUAGES: List[str] = ["ch_sim", "en"]
    OCR_GPU_MEMORY: int = 2048  # MB
    
    # 向量数据库配置
    VECTOR_STORE_TYPE: str = "postgres"  # faiss, chroma, postgres
    VECTOR_DIM: int = 1024  # BAAI_bge-large-zh-v1.5 模型的向量维度
    INDEX_STORE_PATH: str = Field(default_factory=lambda: os.path.join(os.getenv("KNOWLEDGE_BASE_DIR", "/home/ubuntu/workspace/knowledge/"), "vector_stores"))
    
    # PostgreSQL配置
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DATABASE: str = "knowledge_base"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_TABLE_NAME: str = "vector_store"
    
    # 文本分块配置
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    
    # 检索配置
    RETRIEVAL_TOP_K: int = 10
    RERANK_TOP_K: int = 5
    SIMILARITY_THRESHOLD: float = 0.7
    
    # LLM API配置
    LLM_API_BASE: Optional[str] = "https://api.siliconflow.cn/v1/"
    LLM_API_KEY: Optional[str] = "sk-pjzovtzymbtowvsidefrxhxwxwpgzrdpfirhxzikqpxlovbl"
    LLM_API_VERSION: Optional[str] = ""
    LLM_MODEL_NAME: str = "THUDM/glm-4-9b-chat"
    LLM_MAX_TOKENS: int = 4096
    LLM_TEMPERATURE: float = 0.7
    LLM_CONTEXT_WINDOW: int = 8192  # LLM上下文窗口大小
    LLM_NUM_OUTPUT: int = 1024      # LLM输出token数量
    
    # 任务管理配置
    TASK_EXPIRE_TIME: int = 3600  # 1小时
    MAX_CONCURRENT_TASKS: int = 3
    MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 100MB
    
    # 日志配置
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "app.log"
    LOG_DIR: str = Field(default_factory=lambda: os.path.join(os.getenv("KNOWLEDGE_BASE_DIR", "/home/ubuntu/workspace/knowledge/"), "logs"))
    ENABLE_FILE_LOGGING: bool = True
    ENABLE_DOCLING_LOGGING: bool = True
    LOG_ROTATION_SIZE: str = "10MB"
    LOG_RETENTION_DAYS: int = 30
    
    # 存储配置
    UPLOAD_DIR: str = Field(default_factory=lambda: os.path.join(os.getenv("KNOWLEDGE_BASE_DIR", "/home/ubuntu/workspace/knowledge/"), "uploads"))
    OUTPUT_DIR: str = Field(default_factory=lambda: os.path.join(os.getenv("KNOWLEDGE_BASE_DIR", "/home/ubuntu/workspace/knowledge/"), "outputs"))
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

# 全局配置实例
settings = Settings()

# 确保必要目录存在
for directory in [settings.INDEX_STORE_PATH, settings.UPLOAD_DIR, settings.OUTPUT_DIR, settings.LOG_DIR]:
    Path(directory).mkdir(parents=True, exist_ok=True)