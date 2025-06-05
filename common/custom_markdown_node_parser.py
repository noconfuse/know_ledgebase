from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core.schema import BaseNode, TextNode
from typing import List
import re

class CustomMarkdownNodeParser(MarkdownNodeParser):
    """自定义Markdown解析器，header_path包含当前chunk的header"""
    
    def _build_node_from_split(
        self,
        text_split: str,
        node: BaseNode,
        header_path: str,
    ) -> TextNode:
        """构建节点，包含当前chunk的header"""
        from llama_index.core.node_parser.node_utils import build_nodes_from_splits
        
        new_node = build_nodes_from_splits([text_split], node, id_func=self.id_func)[0]
        
        if self.include_metadata:
            separator = self.header_path_separator
            
            # 提取当前chunk的header
            current_header = self._extract_current_header(text_split)
            
            # 构建完整的header_path（包含当前header）
            if current_header:
                if header_path:
                    full_header_path = f"{separator}{header_path}{separator}{current_header}{separator}"
                else:
                    full_header_path = f"{separator}{current_header}{separator}"
            else:
                full_header_path = f"{separator}{header_path}{separator}" if header_path else separator
            
            new_node.metadata["header_path"] = full_header_path
            
            # 同时添加当前header作为单独的元数据字段
            if current_header:
                new_node.metadata["current_header"] = current_header
        
        return new_node
    
    def _extract_current_header(self, text: str) -> str:
        """从文本中提取当前的header"""
        lines = text.split('\n')
        for line in lines:
            header_match = re.match(r'^(#+)\s(.*)', line.strip())
            if header_match:
                return header_match.group(2)
        return ""