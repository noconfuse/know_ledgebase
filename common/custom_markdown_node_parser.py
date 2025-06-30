from typing import List, Tuple
import re
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core.schema import BaseNode, TextNode, MetadataMode
from llama_index.core.node_parser.node_utils import build_nodes_from_splits

class CustomMarkdownNodeParser(MarkdownNodeParser):
    """自定义Markdown解析器，实现以下功能:
    1. 从节点文本中移除Markdown标题行 (e.g., '## Title')。
    2. 在元数据 'header_path' 中包含当前节点自身的标题。
    """

    def get_nodes_from_node(self, node: BaseNode) -> List[TextNode]:
        """通过重写核心解析逻辑来获取节点。"""
        text = node.get_content(metadata_mode=MetadataMode.NONE)
        lines = text.split("\n")
        markdown_nodes: List[TextNode] = []
        current_section_lines: List[str] = []
        header_stack: List[Tuple[int, str]] = []
        code_block = False

        for line in lines:
            if line.lstrip().startswith("```"):
                code_block = not code_block
                current_section_lines.append(line)
                continue

            if not code_block:
                header_match = re.match(r"^(#+)\s(.*)", line.strip())
                if header_match:
                    # Found a header, so the previous section is complete.
                    if current_section_lines:
                        section_text = "\n".join(current_section_lines).strip()
                        if section_text:
                            # The header_stack is correct for this completed section.
                            header_path = self.header_path_separator.join(
                                [h[1] for h in header_stack]
                            )
                            markdown_nodes.append(
                                self._build_node_from_split(
                                    section_text, node, header_path
                                )
                            )

                    # Now, start the new section.
                    # Update header stack.
                    header_level = len(header_match.group(1))
                    header_text = header_match.group(2).strip()
                    while header_stack and header_stack[-1][0] >= header_level:
                        header_stack.pop()
                    header_stack.append((header_level, header_text))

                    # Start the new section's content with the title text.
                    current_section_lines = [header_text]
                    
                    # We've processed this line, so skip to the next.
                    continue

            current_section_lines.append(line)

        # 添加最后一节的内容
        if current_section_lines:
            current_section_text = "\n".join(current_section_lines).strip()
            if current_section_text:
                header_path = self.header_path_separator.join([h[1] for h in header_stack])
                markdown_nodes.append(
                    self._build_node_from_split(
                        current_section_text,
                        node,
                        header_path,
                    )
                )

        return markdown_nodes

    def _build_node_from_split(
        self,
        text_split: str,
        node: BaseNode,
        header_path: str,
    ) -> TextNode:
        """构建节点，并附加完整的header_path元数据。"""
        new_node = build_nodes_from_splits([text_split], node, id_func=self.id_func)[0]

        if self.include_metadata:
            separator = self.header_path_separator
            # header_path现在已经包含了所有层级的标题
            full_header_path = f"{separator}{header_path}{separator}" if header_path else separator
            new_node.metadata["header_path"] = full_header_path
            
            # 也可以选择性地保留当前标题
            if header_path:
                new_node.metadata["current_header"] = header_path.split(separator)[-1]

        return new_node

