from abc import ABC, abstractmethod
from typing import Dict, Any

from docling.datamodel.document import DoclingDocument

from models.task_models import ParseTask

class BaseDocumentProcessor(ABC):
    """文档处理器的抽象基类"""

    @abstractmethod
    def process_document(docling_document: DoclingDocument, task_id: str, output_dir: str, file_name: str) -> Dict[str, Any]:
        """处理Docling文档的抽象方法"""
        pass


    @abstractmethod
    def collect_document(parse_task: ParseTask) -> DoclingDocument:
        """收集处理后的文档的抽象方法"""
        pass


