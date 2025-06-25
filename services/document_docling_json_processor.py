# -*- coding: utf-8 -*-
import os
import re
import json
from typing import Dict, List, Any
from pathlib import Path
import logging
import uuid
import time

from docling.datamodel.document import DoclingDocument

from utils.logging_config import get_logger
from services.base_document_processor import BaseDocumentProcessor

logger = get_logger(__name__)

class DocumentDoclingJsonProcessor(BaseDocumentProcessor):
    """文档JSON处理器，用于处理Docling导出的JSON数据并进行增强"""
    
    def __init__(self):
        """初始化文档JSON处理器"""
        logger.info("DocumentJsonProcessor初始化完成")

    def process_document(docling_document: DoclingDocument, task_id: str, output_dir: str, file_name: str) -> Dict[str, Any]:
        """处理Docling的Document对象，并导出增强后的JSON
        
        Args:
            document: Docling的Document对象
            task_id: 任务ID，用于构建输出路径
            output_dir: 基础输出目录
            file_name: 文件名（不含扩展名）
            
        Returns:
            Dict: 处理结果信息
        """
        full_output_dir = os.path.join(output_dir, task_id)
        os.makedirs(full_output_dir, exist_ok=True)

        processed_result = {
            "title": getattr(docling_document, 'title', file_name),
            "page_count": len(docling_document.pages) if hasattr(docling_document, 'pages') else 1,
            "parsed_at": time.time(),
            "output_directory": full_output_dir,
        }

        original_json_content = docling_document.export_to_dict()
        # TODO: 实现JSON增强处理逻辑
        processed_json_content = original_json_content # 暂时直接使用原始JSON
        
        json_export_path = os.path.join(full_output_dir, f"{file_name}.json")
        with open(json_export_path, 'w', encoding='utf-8') as f:
            json.dump(processed_json_content, f, ensure_ascii=False, indent=2)
        logger.info(f"文档已导出为json: {json_export_path}")

        # 提取表格和图片信息
        # TODO: 从document对象中提取表格和图片，并更新processed_result
        # 例如：
        # tables = document_utils.extract_tables(document) # 假设有一个document_utils模块
        # images = document_utils.extract_images(document)
        # processed_result["tables"] = tables
        # processed_result["images"] = images
        # processed_result["has_tables"] = len(tables) > 0
        # processed_result["has_images"] = len(images) > 0

        return processed_result