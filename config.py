import os
from typing import Optional, List
from pydantic import Field, BaseModel
from pydantic_settings import BaseSettings
from pathlib import Path

# 设置Hugging Face离线模式环境变量
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

os.environ["EASYOCR_DOWNLOAD"] = "0"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

class EmbeddingModelSettings(BaseModel):
    """嵌入模型配置"""
    PROVIDER_NAME: str = "siliconflow"
    MODEL_NAME: str = "BAAI/bge-large-zh-v1.5"
    API_BASE_URL: str = "https://api.siliconflow.com"
    API_KEY: str = "sk-pjzovtzymbtowvsidefrxhxwxwpgzrdpfirhxzikqpxlovbl"
    DIMENSIONS: int = 1024

class RerankModelSettings(BaseModel):
    """重排模型配置"""
    PROVIDER_NAME: str = "siliconflow"
    MODEL_NAME: str = "BAAI/bge-reranker-v2-m3"
    API_BASE_URL: str = "https://api.siliconflow.com"
    API_KEY: str = "sk-pjzovtzymbtowvsidefrxhxwxwpgzrdpfirhxzikqpxlovbl"

class LLMModelSettings(BaseModel):
    """LLM模型配置"""
    PROVIDER_NAME: str = "siliconflow"
    MODEL_NAME: str = "THUDM/GLM-4-9B-0414"
    API_BASE_URL: str = "https://api.siliconflow.com"
    API_KEY: str = "sk-pjzovtzymbtowvsidefrxhxwxwpgzrdpfirhxzikqpxlovbl"
    TEMPERATURE: float = 0.5
    MAX_TOKENS: int = 2048
    MAX_RETRIES: int = 3
    SYSTEM_PROMPT: str = "你是一个专业的法律问答助手，你的回答必须简洁明了，不要答非所问，不要编造答案，不要回答法律问题之外的内容。"


class Settings(BaseSettings):
    """应用配置"""
    
    # 基础路径配置
    BASE_DIR: Path = Path(__file__).parent
    KNOWLEDGE_BASE_DIR: str = os.getenv("KNOWLEDGE_BASE_DIR", str(Path(__file__).parent))
    
    # 服务配置
    DOC_PARSE_HOST: str = "0.0.0.0"
    DOC_PARSE_PORT: int = 8001
    RAG_SERVICE_HOST: str = "0.0.0.0"
    RAG_SERVICE_PORT: int = 8002
    
    # CORS配置
    ALLOWED_ORIGINS: List[str] = ["*"]
    
    
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
    MINERU_API_TIMEOUT: int = 300 # seconds
    MINERU_RETRY_ATTEMPTS: int = 3
    MINERU_RETRY_DELAY: int = 5 # seconds

    TASK_MONITOR_INTERVAL_SECONDS: int = 5 # 任务监控间隔，单位秒
    MAX_CONCURRENT_PARSES_PER_DIRECTORY: int = Field(default=1, description="目录解析的最大并发文件数")
    
    # OCR配置
    OCR_ENABLED: bool = False
    OCR_LANGUAGES: List[str] = ["chi_sim", "eng"]
    OCR_GPU_MEMORY: int = 2048  # MB
    OCR_BACKEND: str = "rapidorc"  # 支持 easyocr, tesseract, rapidocr
    
    # 向量数据库配置
    VECTOR_STORE_TYPE: str = "postgres"  # faiss, chroma, postgres
    VECTOR_DIM: int = 1024  # BAAI_bge-large-zh-v1.5 模型的向量维度
    INDEX_STORE_PATH: str = Field(default_factory=lambda: os.path.join(os.getenv("KNOWLEDGE_BASE_DIR", str(Path(__file__).parent)), "vector_stores"))
    
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

    embedding_model_settings: EmbeddingModelSettings = EmbeddingModelSettings()
    rerank_model_settings: RerankModelSettings = RerankModelSettings()
    llm_model_settings: LLMModelSettings = LLMModelSettings()
    
    # 任务管理配置
    TASK_EXPIRE_TIME: int = 3600  # 1小时
    MAX_CONCURRENT_TASKS: int = 3
    MAX_FILE_SIZE: int = 800 * 1024 * 1024  # 800MB
    
    # 会话管理配置
    SESSION_SOFT_DELETE_DAYS: int = 30  # 30天后软删除
    SESSION_HARD_DELETE_DAYS: int = 180  # 180天后硬删除
    
    # 日志配置
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "app.log"
    LOG_DIR: str = Field(default_factory=lambda: os.path.join(os.getenv("KNOWLEDGE_BASE_DIR", str(Path(__file__).parent)), "logs"))
    ENABLE_FILE_LOGGING: bool = True
    ENABLE_DOCLING_LOGGING: bool = True
    LOG_ROTATION_SIZE: str = "10MB"
    LOG_RETENTION_DAYS: int = 30
    
    # 存储配置
    UPLOAD_DIR: str = Field(default_factory=lambda: os.path.join(os.getenv("KNOWLEDGE_BASE_DIR", str(Path(__file__).parent)), "uploads"))
    OUTPUT_DIR: str = Field(default_factory=lambda: os.path.join(os.getenv("KNOWLEDGE_BASE_DIR", str(Path(__file__).parent)), "outputs"))

    DOCLING_MODEL_PATH: str = Field(default_factory=lambda: os.path.join(os.getenv("KNOWLEDGE_BASE_DIR", str(Path(__file__).parent)), "models_dir/docling"))

    EASY_OCR_MODEL_PATH: str = Field(default_factory=lambda: os.path.join(os.getenv("KNOWLEDGE_BASE_DIR", str(Path(__file__).parent)), "models_dir/easyocr"))

    RAPID_OCR_MODEL_PATH: str = Field(default_factory=lambda: os.path.join(os.getenv("KNOWLEDGE_BASE_DIR", str(Path(__file__).parent)), "models_dir/rapidocr"))
    
    # JWT认证配置
    JWT_SECRET_KEY: str = "your-secret-key-change-in-production-please-use-a-strong-random-key"
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_HOURS: int = 24
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    FREE_LOGIN_TOKEN: str = "free-login-token-please-change-in-production"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

# 全局配置实例
settings = Settings()

# 确保必要目录存在
for directory in [settings.INDEX_STORE_PATH, settings.UPLOAD_DIR, settings.OUTPUT_DIR, settings.LOG_DIR, settings.DOCLING_MODEL_PATH,settings.EASY_OCR_MODEL_PATH]:
    Path(directory).mkdir(parents=True, exist_ok=True)