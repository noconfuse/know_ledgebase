"""环境变量加载模块"""

import os
from pathlib import Path
from dotenv import load_dotenv
import logging

def load_env_vars():
    """加载.env文件中的环境变量
    
    优先级：
    1. 系统环境变量
    2. .env文件中的环境变量
    """
    # 获取项目根目录
    project_root = Path(__file__).parent.parent.absolute()
    
    # .env文件路径
    env_file = project_root / ".env"
    
    # 如果.env文件存在，则加载
    if env_file.exists():
        logging.info(f"Loading environment variables from {env_file}")
        load_dotenv(env_file)
        return True
    else:
        logging.warning(f".env file not found at {env_file}")
        return False