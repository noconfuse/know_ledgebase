import asyncio
import hashlib
import os
import requests
from pathlib import Path
from typing import Dict, Any
from bs4 import BeautifulSoup
from llama_index.core import Document

from config import settings
from models.task_models import ParseTask
from utils.logging_config import get_logger
from utils.document_utils import truncate_filename
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

        output_dir = Path(settings.KNOWLEDGE_BASE_DIR) / "outputs" / task.task_id
        output_dir.mkdir(exist_ok=True)

        # 先不处理附件
        # for a in soup.find_all("a", href=True):
        #     href = a["href"]
        #     local_filename = None
        #     should_process = False

        #     if href.startswith("http"):
        #         try:
        #             head_response = requests.head(href, timeout=5, allow_redirects=True)
        #             head_response.raise_for_status()
        #             content_type = head_response.headers.get("Content-Type", "").split(";")[0]
        #             if content_type in self.SUPPORTED_MIME_TYPES:
        #                 file_ext = self.SUPPORTED_MIME_TYPES[content_type]
        #                 local_filename = output_dir / (Path(href).name or f"temp_file{file_ext}")
                        
        #                 r = requests.get(href, timeout=20)
        #                 r.raise_for_status()
        #                 with open(local_filename, "wb") as f:
        #                     f.write(r.content)
        #                 logger.info(f"已下载远程文件: {href} -> {local_filename}")
        #                 should_process = True
        #             else:
        #                 logger.debug(f"跳过不支持的远程文件类型: {href} (Content-Type: {content_type})")
        #         except requests.exceptions.RequestException as e:
        #             logger.warning(f"HEAD或GET请求远程文件失败: {href}, {e}")
        #         except Exception as e:
        #             logger.warning(f"处理远程文件时发生未知错误: {href}, {e}")
        #     else:
        #         potential_local_path = Path(task.file_path).parent / href
        #         if potential_local_path.is_file() and potential_local_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
        #             local_filename = potential_local_path
        #             should_process = True
        #         else:
        #             logger.debug(f"跳过不支持的本地文件或不存在: {potential_local_path}")

        #     if should_process and local_filename:
        #         try:
        #             # 解析链接文件并获取markdown文件路径
        #             sub_md_path = self._parse_local_file(str(local_filename), task)
        #             if sub_md_path:
        #                 # 计算相对于当前HTML输出目录的相对路径
        #                 sub_md_path_obj = Path(sub_md_path)
        #                 if sub_md_path_obj.exists():
        #                     # 如果markdown文件在子任务的输出目录中，需要计算相对路径
        #                     try:
        #                         # 获取父目录(KNOWLEDGE_BASE_DIR/outputs)
        #                         parent_dir = output_dir.parent
        #                         # 计算从父目录开始的相对路径
        #                         relative_path = sub_md_path_obj.relative_to(parent_dir)
        #                         a["href"] = f"../{relative_path}"
        #                     except ValueError:
        #                         # 如果不在同一目录树下，使用文件名
        #                         a["href"] = sub_md_path_obj.name
        #                 else:
        #                     logger.warning(f"解析生成的markdown文件不存在: {sub_md_path}")
        #             else:
        #                 logger.warning(f"未能获取解析后的markdown文件路径: {local_filename}")
        #         except Exception as e:
        #             logger.warning(f"docling解析超链接文件失败: {local_filename}, {e}")
        #             continue

        base_file_name, _ = os.path.splitext(task.file_name)
        truncated_base_name = truncate_filename(base_file_name, max_length=60, preserve_extension=False)
        html_out_path = output_dir / f"{truncated_base_name}.html"
        with open(html_out_path, "w", encoding="utf-8") as f:
            f.write(str(soup))
        logger.info(f"HTML处理完成，输出: {html_out_path}")

        task.output_directory = str(output_dir)
        return {"title": task.file_name, "html_path": str(html_out_path)}

    def _parse_local_file(self, file_path: str, task: ParseTask) -> str:
        """辅助函数：用docling解析单个文件，返回生成的输出文件路径
        
        注意：此方法在线程池中同步执行，不需要再次使用线程池
        
        Returns:
            str: 生成的输出文件的绝对路径，如果解析失败则返回None
        """
        try:
            from services.document_parse_task import DocumentParseTask
            # 检查递归深度限制，最多允许3层（0, 1, 2）
            if task.depth >= 2:
                logger.warning(f"任务深度已达到限制 (当前深度: {task.depth})，跳过文件 {file_path} 的递归解析")
                return None
            
            # 构建子任务来处理
            sub_task = DocumentParseTask.create_parse_task(file_path, task.config, task)
            logger.info(f"创建子任务 {sub_task.task_id}，深度: {sub_task.depth}，处理文件: {file_path}")
            
            document_parse_task = DocumentParseTask(sub_task.task_id)
            
            # 直接同步执行解析任务，因为当前已经在线程池中执行
            document_parse_task.execute_parse_task()
            
            # 根据统一的路径规则构建输出目录路径
            sub_output_dir = os.path.join(settings.KNOWLEDGE_BASE_DIR, "outputs", sub_task.task_id)
            
            # 如果结果中没有对应路径，尝试从输出目录查找
            if os.path.exists(sub_output_dir):
                # 根据文件扩展名确定输出文件扩展名
                output_ext = '.html' if sub_task.file_extension in ['.html', '.htm'] else '.md'
                
                # 构建输出文件路径，使用截断函数避免文件名过长
                base_name = os.path.splitext(sub_task.file_name)[0]
                truncated_base_name = truncate_filename(base_name, max_length=60, preserve_extension=False)
                output_path = os.path.join(sub_output_dir, f"{truncated_base_name}{output_ext}")
                
                if os.path.exists(output_path):
                    return output_path
            
            # 如果没有找到，记录警告并返回None
            logger.warning(f"未能找到文件 {file_path} 解析生成的输出文件")
            return None
                
        except Exception as e:
            logger.error(f"解析文件 {file_path} 时发生错误: {str(e)}")
            return None

    def collect_document(self, parse_task: ParseTask):
        """收集解析任务的文档
        
        Args:
            parse_task: 解析任务（子任务）
            
        Returns:
            List[Document]: 文档列表
        """
        from models.parse_task import TaskStatus
        
        # 校验任务状态
        if parse_task.status != TaskStatus.COMPLETED:
            raise ValueError(f"Parse task {parse_task.task_id} is not completed.")
        
        documents = []
        logger.info(f"Processing HTML subtask {parse_task.task_id}")
        
        # 基于KNOWLEDGE_BASE_DIR和task_id构建输出目录路径
        from config import settings
        output_dir = os.path.join(settings.KNOWLEDGE_BASE_DIR, "outputs", parse_task.task_id)
        
        # 校验输出目录是否存在
        if not os.path.exists(output_dir):
            raise ValueError(f"Output directory does not exist: {output_dir}")
        
        output_dir = Path(output_dir)
        
        try:
            # 收集主HTML文件
            base_file_name, _ = os.path.splitext(parse_task.file_name)
            truncated_base_name = truncate_filename(base_file_name, max_length=60, preserve_extension=False)
            main_html_path = output_dir / f"{truncated_base_name}.html"
            if main_html_path.exists():
                # LlamaIndex Document for the main HTML
                with open(main_html_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                    content_md5 = hashlib.md5(html_content.encode('utf-8')).hexdigest()
                metadata = {
                    "original_file_path": parse_task.file_path,
                    "file_size": main_html_path.stat().st_size,
                    'file_type': '.html',
                    'mime_type': parse_task.mime_type
                }
                doc = Document(text=html_content, metadata=metadata,doc_id=content_md5)
                documents.append(doc)

            # 收集所有生成的Markdown文件
            for md_file in output_dir.glob("*.md"):
                content = md_file.read_text(encoding='utf-8')
                content_md5 = hashlib.md5(content.encode('utf-8')).hexdigest()
                metadata = {
                    "original_file_path": parse_task.file_path, # 指向原始的HTML文件
                    "file_size": md_file.stat().st_size,
                    'file_type': '.md',
                    'mime_type': parse_task.mime_type
                }

                doc = Document(text=content, metadata=metadata,doc_id=content_md5)
                documents.append(doc)
                
        except Exception as e:
            logger.error(f"收集HTML文档出错：{str(e)}")
            raise

        return documents