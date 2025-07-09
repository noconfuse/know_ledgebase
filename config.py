import os
from typing import Optional, List
from pydantic import Field, BaseModel
from pydantic_settings import BaseSettings
from pathlib import Path

# 加载环境变量
try:
    from utils.env_loader import load_env_vars
    load_env_vars()
except ImportError:
    # 如果导入失败，尝试直接使用dotenv
    try:
        from dotenv import load_dotenv
        from pathlib import Path
        project_root = Path(__file__).parent.absolute()
        env_file = project_root / ".env"
        if env_file.exists():
            load_dotenv(env_file)
    except ImportError:
        pass  # 如果dotenv未安装，继续使用默认值

# 设置Hugging Face离线模式环境变量

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

class EmbeddingModelSettings(BaseModel):
    """嵌入模型配置"""
    PROVIDER_NAME: str = os.getenv("EMBEDDING_PROVIDER_NAME", "siliconflow")
    MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
    API_BASE_URL: str = os.getenv("EMBEDDING_API_BASE_URL", "https://api.siliconflow.cn/v1/embeddings")
    API_KEY: str = os.getenv("EMBEDDING_API_KEY", "")
    DIMENSIONS: int = int(os.getenv("EMBEDDING_DIMENSIONS", "1024"))
    LOCAL_PATH: str = os.getenv("EMBEDDING_LOCAL_PATH", "")

class RerankModelSettings(BaseModel):
    """重排模型配置"""
    PROVIDER_NAME: str = os.getenv("RERANK_PROVIDER_NAME", "siliconflow")
    MODEL_NAME: str = os.getenv("RERANK_MODEL_NAME", "BAAI/bge-reranker-v2-m3")
    API_BASE_URL: str = os.getenv("RERANK_API_BASE_URL", "https://api.siliconflow.cn/v1/rerank")
    API_KEY: str = os.getenv("RERANK_API_KEY", "")

class LLMModelSettings(BaseModel):
    """LLM模型配置"""
    PROVIDER_NAME: str = os.getenv("LLM_PROVIDER_NAME", "zhipu")
    MODEL_NAME: str = os.getenv("LLM_MODEL_NAME", "glm-4-flash")
    API_BASE_URL: str = os.getenv("LLM_API_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
    API_KEY: str = os.getenv("LLM_API_KEY", "")
    TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.5"))
    MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "2048"))
    MAX_RETRIES: int = int(os.getenv("LLM_MAX_RETRIES", "3"))
    API_REQUEST_INTERVAL: float = float(os.getenv("LLM_API_REQUEST_INTERVAL", "1"))
    SYSTEM_PROMPT: str = os.getenv("LLM_SYSTEM_PROMPT", "你是一个专业的法律问答助手，你的回答必须简洁明了，不要答非所问，不要编造答案，不要回答法律问题之外的内容。")


class Settings(BaseSettings):
    """应用配置"""
    
    # 基础路径配置
    BASE_DIR: Path = Path(__file__).parent
    KNOWLEDGE_BASE_DIR: str = os.getenv("KNOWLEDGE_BASE_DIR", str(Path(__file__).parent))
    
    # 嵌入模型配置字段
    EMBEDDING_PROVIDER_NAME: str = os.getenv("EMBEDDING_PROVIDER_NAME", "siliconflow")
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
    EMBEDDING_API_BASE_URL: str = os.getenv("EMBEDDING_API_BASE_URL", "https://api.siliconflow.com")
    EMBEDDING_API_KEY: str = os.getenv("EMBEDDING_API_KEY", "")
    EMBEDDING_DIMENSIONS: str = os.getenv("EMBEDDING_DIMENSIONS", "1024")
    
    # 重排模型配置字段
    RERANK_PROVIDER_NAME: str = os.getenv("RERANK_PROVIDER_NAME", "siliconflow")
    RERANK_MODEL_NAME: str = os.getenv("RERANK_MODEL_NAME", "BAAI/bge-reranker-v2-m3")
    RERANK_API_BASE_URL: str = os.getenv("RERANK_API_BASE_URL", "https://api.siliconflow.com")
    RERANK_API_KEY: str = os.getenv("RERANK_API_KEY", "")
    
    # LLM模型配置字段
    LLM_PROVIDER_NAME: str = os.getenv("LLM_PROVIDER_NAME", "zhipu")
    LLM_MODEL_NAME: str = os.getenv("LLM_MODEL_NAME", "glm-4-flash-250414")
    LLM_API_BASE_URL: str = os.getenv("LLM_API_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
    LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")
    LLM_TEMPERATURE: str = os.getenv("LLM_TEMPERATURE", "0.5")
    LLM_MAX_TOKENS: str = os.getenv("LLM_MAX_TOKENS", "2048")
    LLM_MAX_RETRIES: str = os.getenv("LLM_MAX_RETRIES", "3")
    LLM_API_REQUEST_INTERVAL: str = os.getenv("LLM_API_REQUEST_INTERVAL", "1")
    LLM_SYSTEM_PROMPT: str = os.getenv("LLM_SYSTEM_PROMPT", "你是一个专业的法律问答助手，你的回答必须简洁明了，不要答非所问，不要编造答案，不要回答法律问题之外的内容。")
    
    # Hugging Face配置字段
    HF_HUB_OFFLINE: str = os.getenv("HF_HUB_OFFLINE", "0")
    TRANSFORMERS_OFFLINE: str = os.getenv("TRANSFORMERS_OFFLINE", "0")
    EASYOCR_DOWNLOAD: str = os.getenv("EASYOCR_DOWNLOAD", "0")
    HF_ENDPOINT: str = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")
    
    # 服务配置
    DOC_PARSE_HOST: str = os.getenv("DOC_PARSE_HOST", "0.0.0.0")
    DOC_PARSE_PORT: int = int(os.getenv("DOC_PARSE_PORT", "8001"))
    RAG_SERVICE_HOST: str = os.getenv("RAG_SERVICE_HOST", "0.0.0.0")
    RAG_SERVICE_PORT: int = int(os.getenv("RAG_SERVICE_PORT", "8002"))
    
    # CORS配置
    ALLOWED_ORIGINS: List[str] = ["*"]
    
    
    # GPU配置
    USE_GPU: bool = True
    GPU_DEVICE: str = "cuda:0"
    EMBEDDING_BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "2"))  # 嵌入模型批处理大小，降低以减少GPU内存使用
    
    # 文档解析配置
    MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 100MB
    SUPPORTED_FORMATS: List[str] = [".pdf", ".docx", ".doc", ".txt", ".md", ".html", ".pptx", ".xlsx", '.xls']
    SUPPORTED_PARSER_TYPES: List[str] = ["docling", "mineru"]
    DEFAULT_PARSER_TYPE: str = "docling"  # 默认解析器类型: "docling" 或 "mineru"
    PARSE_TIMEOUT: int = 300  # 5分钟
    
    # 解析器类型配置
    
    MINERU_MAX_WORKERS: int = 1  # MinerU并发处理数量
    MINERU_API_TIMEOUT: int = 300 # seconds
    MINERU_RETRY_ATTEMPTS: int = 3
    MINERU_RETRY_DELAY: int = 5 # seconds

    TASK_MONITOR_INTERVAL_SECONDS: int = 5 # 任务监控间隔，单位秒
    MAX_CONCURRENT_PARSES_PER_DIRECTORY: int = Field(default=1, description="目录解析的最大并发文件数")
    
    # OCR配置
    OCR_ENABLED: bool = False
    OCR_LANGUAGES: List[str] = ["ch_sim", "en"]  # EasyOCR支持的语言代码
    OCR_GPU_MEMORY: int = 2048  # MB
    OCR_TYPE: str = "rapidocr"  # 支持 easyocr, tesseract, rapidocr
    
    # 向量数据库配置
    VECTOR_STORE_TYPE: str = "postgres"  # faiss, chroma, postgres
    VECTOR_DIM: int = 1024  # BAAI_bge-large-zh-v1.5 模型的向量维度
    INDEX_STORE_PATH: str = Field(default_factory=lambda: os.path.join(os.getenv("KNOWLEDGE_BASE_DIR", str(Path(__file__).parent)), "vector_stores"))
    
    # PostgreSQL配置
    POSTGRES_HOST: str = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT: int = int(os.getenv("POSTGRES_PORT", "5432"))
    POSTGRES_DATABASE: str = os.getenv("POSTGRES_DATABASE", "knowledge_base")
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "postgres")
    POSTGRES_TABLE_NAME: str = os.getenv("POSTGRES_TABLE_NAME", "vector_store")
    
    
    # 不同文档类型的分块配置
    LEGAL_CHUNK_SIZE: int = int(os.getenv("LEGAL_CHUNK_SIZE", "1000"))  # 法律文档使用更大的分块
    LEGAL_CHUNK_OVERLAP: int = int(os.getenv("LEGAL_CHUNK_OVERLAP", "50"))  # 法律文档重叠稍小
    POLICY_CHUNK_SIZE: int = int(os.getenv("POLICY_CHUNK_SIZE", "1000"))  # 政策文档中等分块
    POLICY_CHUNK_OVERLAP: int = int(os.getenv("POLICY_CHUNK_OVERLAP", "100"))  # 政策文档重叠中等
    GENERAL_CHUNK_SIZE: int = int(os.getenv("GENERAL_CHUNK_SIZE", "800"))  # 一般文档标准分块
    GENERAL_CHUNK_OVERLAP: int = int(os.getenv("GENERAL_CHUNK_OVERLAP", "100"))  # 一般文档标准重叠
    
    # 元数据提取配置
    MIN_CHUNK_SIZE_FOR_EXTRACTION: int = int(os.getenv("MIN_CHUNK_SIZE_FOR_EXTRACTION", "20"))  # 进行元数据提取的最小chunk大小
    MIN_CHUNK_SIZE_FOR_SUMMARY: int = int(os.getenv("MIN_CHUNK_SIZE_FOR_SUMMARY", "512"))  # 生成摘要的最小chunk大小
    MIN_CHUNK_SIZE_FOR_QA: int = int(os.getenv("MIN_CHUNK_SIZE_FOR_QA", "1024"))  # 生成问答对的最小chunk大小
    MAX_KEYWORDS: int = int(os.getenv("MAX_KEYWORDS", "5"))  # 要提取的最大关键词数
    
    # 文档内容优化配置
    DOC_CONTENT_OPTIMIZATION_THRESHOLD: int = int(os.getenv("DOC_CONTENT_OPTIMIZATION_THRESHOLD", "10000"))  # 启用内容优化的文档长度阈值
    DOC_LARGE_CONTENT_THRESHOLD: int = int(os.getenv("DOC_LARGE_CONTENT_THRESHOLD", "50000"))  # 超大文档阈值，使用摘要策略
    DOC_MAX_SECTION_LENGTH: int = int(os.getenv("DOC_MAX_SECTION_LENGTH", "1000"))  # 单个章节最大长度
    DOC_MAX_OPTIMIZED_LENGTH: int = int(os.getenv("DOC_MAX_OPTIMIZED_LENGTH", "8000"))  # 优化后内容最大长度
    DOC_SUMMARY_MAX_LENGTH: int = int(os.getenv("DOC_SUMMARY_MAX_LENGTH", "6000"))  # 文档摘要最大长度
    
    # 检索配置
    RETRIEVAL_TOP_K: int = 10
    RERANK_TOP_K: int = 10
    SIMILARITY_THRESHOLD: float = 0.3

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

    EMBEDDING_LOCAL_PATH: str = os.getenv("EMBEDDING_LOCAL_PATH", "/home/ubuntu/workspace/know_ledgebase/models_dir/bge_m3")
    
    # JWT认证配置
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production-please-use-a-strong-random-key")
    JWT_ALGORITHM: str = os.getenv("JWT_ALGORITHM", "HS256")
    JWT_ACCESS_TOKEN_EXPIRE_HOURS: int = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_HOURS", "24"))
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = int(os.getenv("JWT_REFRESH_TOKEN_EXPIRE_DAYS", "7"))

    FREE_LOGIN_TOKEN: str = os.getenv("FREE_LOGIN_TOKEN", "free-login-token-please-change-in-production")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

# 全局配置实例
settings = Settings()

# 确保必要目录存在
for directory in [settings.INDEX_STORE_PATH, settings.UPLOAD_DIR, settings.OUTPUT_DIR, settings.LOG_DIR, settings.DOCLING_MODEL_PATH,settings.EASY_OCR_MODEL_PATH]:
    Path(directory).mkdir(parents=True, exist_ok=True)