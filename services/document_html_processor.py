import os
import requests
from pathlib import Path
from typing import Dict, Any
from bs4 import BeautifulSoup
from llama_index.core import Document

from config import settings
from models.task_models import ParseTask
from services.docling_parser import DoclingParser
from utils.logging_config import get_logger
from services.base_document_processor import BaseDocumentProcessor

logger = get_logger(__name__)

class DocumentHTMLProcessor(BaseDocumentProcessor):
    """处理器，用于解析HTML文件及其中的链接"""

    SUPPORTED_MIME_TYPES = {
        "application/pdf": ".pdf",
        "application/msword": ".doc",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
        "application/vnd.ms-powerpoint": ".ppt",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
        "image/jpeg": ".jpeg",
        "image/png": ".png",
        "image/gif": ".gif",
        "image/bmp": ".bmp",
        "image/tiff": ".tiff",
    }
    SUPPORTED_EXTENSIONS = list(SUPPORTED_MIME_TYPES.values())

    def process_document(self, task: ParseTask) -> Dict[str, Any]:
        logger.info(f"使用BeautifulSoup处理HTML文件: {task.file_path}")
        with open(task.file_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        soup = BeautifulSoup(html_content, "html.parser")

        output_dir = Path(settings.OUTPUT_DIR) / task.task_id
        output_dir.mkdir(exist_ok=True)

        for a in soup.find_all("a", href=True):
            href = a["href"]
            local_filename = None
            should_process = False

            if href.startswith("http"):
                try:
                    head_response = requests.head(href, timeout=5, allow_redirects=True)
                    head_response.raise_for_status()
                    content_type = head_response.headers.get("Content-Type", "").split(";")[0]
                    if content_type in self.SUPPORTED_MIME_TYPES:
                        file_ext = self.SUPPORTED_MIME_TYPES[content_type]
                        local_filename = output_dir / (Path(href).name or f"temp_file{file_ext}")
                        
                        r = requests.get(href, timeout=20)
                        r.raise_for_status()
                        with open(local_filename, "wb") as f:
                            f.write(r.content)
                        logger.info(f"已下载远程文件: {href} -> {local_filename}")
                        should_process = True
                    else:
                        logger.debug(f"跳过不支持的远程文件类型: {href} (Content-Type: {content_type})")
                except requests.exceptions.RequestException as e:
                    logger.warning(f"HEAD或GET请求远程文件失败: {href}, {e}")
                except Exception as e:
                    logger.warning(f"处理远程文件时发生未知错误: {href}, {e}")
            else:
                potential_local_path = Path(task.file_path).parent / href
                if potential_local_path.is_file() and potential_local_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                    local_filename = potential_local_path
                    should_process = True
                else:
                    logger.debug(f"跳过不支持的本地文件或不存在: {potential_local_path}")

            if should_process and local_filename:
                try:
                    sub_md_content = self._parse_file_to_markdown(str(local_filename), task.config)
                    sub_md_path = output_dir / f"{Path(local_filename).stem}.md"
                    with open(sub_md_path, "w", encoding="utf-8") as f:
                        f.write(sub_md_content)
                    a["href"] = str(sub_md_path.relative_to(output_dir))
                except Exception as e:
                    logger.warning(f"docling解析超链接文件失败: {local_filename}, {e}")
                    continue

        base_file_name, _ = os.path.splitext(task.file_name)
        html_out_path = output_dir / f"{base_file_name}.html"
        with open(html_out_path, "w", encoding="utf-8") as f:
            f.write(str(soup))
        logger.info(f"HTML处理完成，输出: {html_out_path}")

        task.output_directory = str(output_dir)
        return {"title": task.file_name, "html_path": str(html_out_path)}

    def _parse_file_to_markdown(self, file_path: str, config: dict = None) -> str:
        """辅助函数：用docling解析单个文件为markdown，返回markdown内容"""
        docling_parser = DoclingParser()
        docling_parser.setup_converter(config)
        result = docling_parser.parse_file(file_path)
        document = result.document
        markdown_content = document.export_to_markdown()
        return markdown_content

    def collect_document(self, parse_task: ParseTask):
        """收集解析任务的文档（只处理子任务）
        
        Args:
            parse_task: 解析任务（子任务）
            
        Returns:
            List[Document]: 文档列表
        """
        from models.parse_task import TaskStatus
        
        # 校验任务状态
        if parse_task.status != TaskStatus.COMPLETED:
            raise ValueError(f"Parse task {parse_task.task_id} is not completed.")

        # 只处理子任务
        if parse_task.subtasks:
            raise ValueError(f"HTMLProcessor.collect_document should only process subtasks, not main tasks")
        
        documents = []
        logger.info(f"Processing HTML subtask {parse_task.task_id}")
        
        # 校验解析任务是否有输出目录
        output_dir = parse_task.output_directory
        if not output_dir:
            raise ValueError(f"No output directory found for parse task {parse_task.task_id}")
        
        output_dir = Path(output_dir)
        
        try:
            # 收集主HTML文件
            base_file_name, _ = os.path.splitext(parse_task.file_name)
            main_html_path = output_dir / f"{base_file_name}.html"
            if main_html_path.exists():
                # LlamaIndex Document for the main HTML
                with open(main_html_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                metadata = {
                    "source_file": str(main_html_path),
                    "original_file_path": parse_task.file_path,
                    "file_size": main_html_path.stat().st_size,
                    'file_type': '.html',
                    'mime_type': parse_task.mime_type
                }
                doc = Document(text=html_content, metadata=metadata)
                documents.append(doc)

            # 收集所有生成的Markdown文件
            for md_file in output_dir.glob("*.md"):
                content = md_file.read_text(encoding='utf-8')
                metadata = {
                    "source_file": str(md_file),
                    "original_file_path": parse_task.file_path, # 指向原始的HTML文件
                    "file_size": md_file.stat().st_size,
                    'file_type': '.md',
                    'mime_type': parse_task.mime_type
                }
                doc = Document(text=content, metadata=metadata)
                documents.append(doc)
                
        except Exception as e:
            logger.error(f"收集HTML文档出错：{str(e)}")
            raise

        return documents