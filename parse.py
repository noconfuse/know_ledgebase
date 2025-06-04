from docling.document_converter import DocumentConverter
import logging
import time
from pathlib import Path
from typing import Optional, Callable


class DocumentParser:
    """文档解析器单例类"""
    _instance = None
    _converter = None
    _logger = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DocumentParser, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, enable_logging: bool = True, log_level: int = logging.INFO):
        if self._converter is None:
            self._converter = DocumentConverter()
            
        if self._logger is None and enable_logging:
            self._setup_logging(log_level)
    
    def _setup_logging(self, log_level: int = logging.INFO):
        """设置日志记录"""
        self._logger = logging.getLogger(__name__)
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self._logger.addHandler(handler)
            self._logger.setLevel(log_level)
    
    @property
    def converter(self):
        """获取DocumentConverter实例"""
        return self._converter
    
    def parse(self, source, progress_callback: Optional[Callable[[str], None]] = None):
        """解析文档
        
        Args:
            source (str): 文档路径或URL
            progress_callback (Optional[Callable]): 进度回调函数
            
        Returns:
            解析结果对象
        """
        return self._parse_with_progress(source, progress_callback)
    
    def parse_to_markdown(self, source, progress_callback: Optional[Callable[[str], None]] = None):
        """解析文档并导出为Markdown格式
        
        Args:
            source (str): 文档路径或URL
            progress_callback (Optional[Callable]): 进度回调函数
            
        Returns:
            str: Markdown格式的文档内容
        """
        result = self._parse_with_progress(source, progress_callback)
        
        if progress_callback:
            progress_callback("正在导出为Markdown格式...")
        if self._logger:
            self._logger.info("正在导出为Markdown格式...")
            
        markdown_content = result.document.export_to_markdown()
        
        if progress_callback:
            progress_callback("Markdown导出完成")
        if self._logger:
            self._logger.info("Markdown导出完成")
            
        return markdown_content
    
    def _parse_with_progress(self, source, progress_callback: Optional[Callable[[str], None]] = None):
        """带进度监控的解析方法"""
        # 获取文档信息
        if isinstance(source, str):
            if source.startswith(('http://', 'https://')):
                doc_name = source.split('/')[-1] or 'remote_document'
            else:
                doc_name = Path(source).name
        else:
            doc_name = str(source)
        
        # 开始解析
        start_time = time.time()
        
        if progress_callback:
            progress_callback(f"开始解析文档: {doc_name}")
        if self._logger:
            self._logger.info(f"开始解析文档: {doc_name}")
        
        if progress_callback:
            progress_callback("正在初始化文档转换器...")
        if self._logger:
            self._logger.info("正在初始化文档转换器...")
        
        if progress_callback:
            progress_callback("正在分析文档结构...")
        if self._logger:
            self._logger.info("正在分析文档结构...")
        
        # 执行转换
        try:
            result = self._converter.convert(source)
            
            # 计算耗时
            end_time = time.time()
            duration = end_time - start_time
            
            # 获取页面信息
            page_count = len(result.pages) if hasattr(result, 'pages') else 0
            
            success_msg = f"文档解析完成! 耗时: {duration:.2f}秒, 页数: {page_count}"
            
            if progress_callback:
                progress_callback(success_msg)
            if self._logger:
                self._logger.info(success_msg)
                
            # 显示转换状态
            if hasattr(result, 'status'):
                status_msg = f"转换状态: {result.status}"
                if progress_callback:
                    progress_callback(status_msg)
                if self._logger:
                    self._logger.info(status_msg)
            
            # 显示错误信息（如果有）
            if hasattr(result, 'errors') and result.errors:
                error_msg = f"发现 {len(result.errors)} 个错误/警告"
                if progress_callback:
                    progress_callback(error_msg)
                if self._logger:
                    self._logger.warning(error_msg)
                    for error in result.errors:
                        self._logger.warning(f"  - {error}")
            
            return result
            
        except Exception as e:
            error_msg = f"文档解析失败: {str(e)}"
            if progress_callback:
                progress_callback(error_msg)
            if self._logger:
                self._logger.error(error_msg)
            raise
    
    def parse_with_detailed_progress(self, source):
        """带详细进度输出的解析方法"""
        def detailed_progress_callback(message):
            print(f"[进度] {message}")
            
        return self.parse(source, progress_callback=detailed_progress_callback)


# 创建全局单例实例
parse_instance = DocumentParser()


# 使用示例
if __name__ == "__main__":
    # 示例用法
    source = "https://arxiv.org/pdf/2408.09869"  # document per local path or URL
    
    # 方式1：直接使用全局实例
    result = parse_instance.parse(source)
    print(result.document.export_to_markdown())  # output: "## Docling Technical Report[...]"
    
    # 方式2：使用便捷方法
    markdown_content = parse_instance.parse_to_markdown(source)
    print(markdown_content)
    
    # 验证单例模式
    parser1 = DocumentParser()
    parser2 = DocumentParser()
    print(f"单例验证: {parser1 is parser2}")  # 应该输出 True