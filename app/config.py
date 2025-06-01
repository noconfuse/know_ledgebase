# 模型相关配置
import os
from pydantic_settings import BaseSettings

#"/cloud/cloud-ssd1/myrag/models_dir/"  的绝对路径 
MODEL_BASE_DIR: str = os.path.abspath("/cloud/cloud-ssd1/myrag/models_dir/")
KONWLEDGE_BASE_DIR: str = os.path.abspath("/cloud/cloud-ssd1/knowledge_build_admin/")
class MYSettings(BaseSettings):
    PROJECTS_DIR: str = os.path.join(KONWLEDGE_BASE_DIR, "projects")
    EMBED_MODEL_PATH: str = os.path.join(MODEL_BASE_DIR, "gte-base-zh")
    RANK_MODEL_PATH: str = os.path.join(MODEL_BASE_DIR, "bge-reranker-large")
    SEMATIC_RETRIEVER_MODEL_PATH:str = os.path.join(MODEL_BASE_DIR, "paraphrase-multilingual-MiniLM-L12-v2")

    # EMBED_MODEL_PATH: str = "models_dir/bge-large-zh-v1.5"
    # RANK_MODEL_PATH: str = "models_dir/bge-reranker-large"
    # SEMATIC_RETRIEVER_MODEL_PATH: str = "models_dir/paraphrase-multilingual-MiniLM-L12-v2"
    # BIG_LLM_MODEL_PATH: str = "models_dir/Qwen2.5-7B"
    SHIP_CHECK_LLM_MODEL_PATH: str = os.path.join(MODEL_BASE_DIR, "internlm2_5-1_8b-chat")

    
class Config:
    env_file = ".env"
    env_file_encoding = "utf-8"
    secret_key = "changhai_ai"
    algorithm = "HS256"

    

modelsettings = MYSettings()

config = Config()