# -*- coding: utf-8 -*-
import os
import re
import json
import os
import re
import json
from typing import Dict, List, Tuple, Any
from pathlib import Path
import logging

# 导入Docling的Document类型
from docling.datamodel.document import DoclingDocument
from docling_core.types.doc.document import RefItem
from llama_index.core import Document

from models.parse_task import TaskStatus
from models.task_models import ParseTask
from utils.logging_config import get_logger
from services.base_document_processor import BaseDocumentProcessor

logger = get_logger(__name__)

class DocumentDoclingProcessor(BaseDocumentProcessor):
    """文档Docling处理器，用于增强解析后文件的结构化"""

    
    def __init__(self):
        """初始化文档后处理器"""
        self.hierarchy_patterns = [
            r"^(第[一二三四五六七八九十\-\d\s]*章)",
            r"^(第[一二三四五六七八九十\-\d\s]*条)", # 第一条
            r"^(第[\d\s]*节)",
            r"^(\d+\.\d+\.\d+)\s*",  # 3.1.2
            r"^(\d+\.\d+\.\d+\.\d+)\s*",  # 3.1.2.1
            r"^(\d+\.\d+)\s*",  # 3.1
            r"^([一二三四五六七八九十]+)\s*",
        ]
        self.default_level_names = ["章", "节", "条", "项", "款", "目", "段"]
        logger.info("DocumentPostProcessor初始化完成")
    
    def has_chinese_punctuation(self, text: str) -> bool:
        """判断文本是否包含中文标点符号"""
        # 定义中文标点符号集合（可根据实际文档特征扩展）
        CHINESE_PUNCTUATION = {'，', '。', '；', '：', '？', '！', '"', '"', '、', '《', '》'}
        return any(char in CHINESE_PUNCTUATION for char in text)
    
    def _convert_table_cells_to_markdown(self, table_cells: List[Dict[str, Any]]) -> str:
        """将table_cells数据转换为markdown表格格式
        
        Args:
            table_cells: 表格单元格数据列表，每个单元格包含位置和内容信息
            
        Returns:
            str: markdown格式的表格文本
        """
        if not table_cells:
            return "[空表格]"
        
        try:
            # 构建表格矩阵
            max_row = 0
            max_col = 0
            cell_matrix = {}
            
            # 遍历所有单元格，确定表格尺寸和填充数据
            for cell in table_cells:
                if not isinstance(cell, dict):
                    continue
                logger.info(f"处理单元格: {cell}")
                # 获取单元格位置信息
                row_span = cell.get('row_span', 1)
                col_span = cell.get('col_span', 1)
                start_row_offset_idx = cell.get('start_row_offset_idx', 0)
                end_row_offset_idx = cell.get('end_row_offset_idx', start_row_offset_idx)
                start_col_offset_idx = cell.get('start_col_offset_idx', 0)
                end_col_offset_idx = cell.get('end_col_offset_idx', start_col_offset_idx)
                
                # 获取单元格文本内容
                text = cell.get('text', '').strip()
                
                # 更新最大行列数
                max_row = max(max_row, start_row_offset_idx + row_span)
                max_col = max(max_col, start_col_offset_idx + col_span)
                
                # 填充单元格数据（处理合并单元格）
                for r in range(start_row_offset_idx, start_row_offset_idx + row_span):
                    for c in range(start_col_offset_idx, start_col_offset_idx + col_span):
                        if (r, c) not in cell_matrix:
                            # 只在起始位置放置文本，其他位置标记为合并单元格
                            if r == start_row_offset_idx and c == start_col_offset_idx:
                                cell_matrix[(r, c)] = text
                            else:
                                cell_matrix[(r, c)] = ''  # 合并单元格的其他部分
            
            # 构建markdown表格
            markdown_lines = []
            
            for row in range(max_row):
                row_cells = []
                for col in range(max_col):
                    cell_text = cell_matrix.get((row, col), '')
                    # 处理markdown表格中的特殊字符
                    cell_text = cell_text.replace('|', '\\|').replace('\n', ' ')
                    row_cells.append(cell_text)
                
                # 添加表格行
                markdown_lines.append('| ' + ' | '.join(row_cells) + ' |')
                
                # 在第一行后添加分隔符
                if row == 0:
                    separator = '| ' + ' | '.join(['---'] * max_col) + ' |'
                    markdown_lines.append(separator)
            
            return '\n'.join(markdown_lines)
            
        except Exception as e:
            logger.warning(f"转换表格为markdown时出错: {e}")
            return f"[表格: {len(table_cells)} 个单元格，转换失败]"
    
    def _detect_pattern(self, text: str) -> Tuple[int, str]:
        """检测文本匹配的模式索引"""
        for i, pattern in enumerate(self.hierarchy_patterns):
            match = re.match(pattern, text)
            if match:
                return i, match.group(1)
        return -1, ""
    
    
    def process_document(self, docling_document: DoclingDocument, output_dir: str, file_name: str, enhanced: bool = True) -> Dict[str, Any]:
        """处理Docling的Document对象，并导出JSON、Markdown和中间content_list.json
        
        Args:
            docling_document: Docling的Document对象
            output_dir: 基础输出目录
            file_name: 文件名（不含扩展名）
            enhanced: 是否增强处理
            
        Returns:
            Dict: 处理结果信息
        """
        os.makedirs(output_dir, exist_ok=True)

        # 移除文件名中的扩展名，以避免重复后缀
        base_file_name, _ = os.path.splitext(file_name)

        json_export_path = os.path.join(output_dir, f"{base_file_name}.json")
        markdown_export_path = os.path.join(output_dir, f"{base_file_name}.md")
        content_list_path = os.path.join(output_dir, f"{base_file_name}_content_list.json")

        # 导出原始JSON和markdown
        with open(json_export_path, 'w', encoding='utf-8') as f:
            json.dump(docling_document.export_to_dict(), f, ensure_ascii=False, indent=4)
        logger.info(f"原始Docling JSON已导出: {json_export_path}")
        
        original_markdown = docling_document.export_to_markdown()
        with open(markdown_export_path, 'w', encoding='utf-8') as f:
            f.write(original_markdown)
        logger.info(f"原始Docling markdown已导出: {markdown_export_path}")

        # 生成中间content_list.json（基于docling导出的json）
        content_list = self._extract_content_list_from_docling(docling_document)
        with open(content_list_path, 'w', encoding='utf-8') as f:
            json.dump(content_list, f, ensure_ascii=False, indent=2)
        logger.info(f"中间content_list.json已导出: {content_list_path}")

        # 如果进行了增强处理，只对markdown进行增强处理
        if enhanced:
            # 备份原始markdown文件
            if os.path.exists(markdown_export_path):
                os.rename(markdown_export_path, markdown_export_path + ".bak")

            # 基于content_list对markdown进行增强处理
            enhanced_markdown = self._enhance_markdown_from_content_list(original_markdown, content_list)

            # 保存增强后的Markdown
            with open(markdown_export_path, 'w', encoding='utf-8') as f:
                f.write(enhanced_markdown)
            logger.info(f"增强后的Markdown已导出: {markdown_export_path}")

        processed_result = {
            "title": getattr(docling_document, 'title', file_name),
            "page_count": len(docling_document.pages) if hasattr(docling_document, 'pages') else 1,
            "json_path": json_export_path,
            "markdown_path": markdown_export_path
        }

        return processed_result


    def _extract_content_list_from_docling(self, docling_document: DoclingDocument) -> List[Dict[str, Any]]:
        """从DoclingDocument中提取content_list，包含文本、level、type、page_no信息
        
        Args:
            docling_document: DoclingDocument对象
            
        Returns:
            List[Dict]: 包含text、level、type、page_idx的内容列表
        """
        content_list = []
        
        def resolve_ref(ref: str, doc_dict: Dict[str, Any]) -> Any:
            """解析$ref引用"""
            try:
                parts = ref.strip('#/').split('/')
                obj = doc_dict
                for part in parts:
                    if isinstance(obj, dict):
                        obj = obj.get(part)
                    elif isinstance(obj, list):
                        obj = obj[int(part)]
                return obj
            except (KeyError, IndexError, AttributeError) as e:
                logger.warning(f"Could not resolve reference: {ref}, error: {e}")
                return None

        def extract_from_children(children: List[RefItem], doc_dict: Dict[str, Any]):
            """递归提取children中的内容"""
            for child_ref in children:
                # RefItem对象有cref属性，别名为$ref
                if hasattr(child_ref, 'cref'):
                    ref_str = child_ref.cref
                elif isinstance(child_ref, dict) and '$ref' in child_ref:
                    ref_str = child_ref['$ref']
                else:
                    continue
                resolved_item = resolve_ref(ref_str, doc_dict)
                if not resolved_item:
                    continue

                # 从$ref中提取类型信息
                ref_parts = ref_str.strip('#/').split('/')
                item_type_key = ref_parts[0] if ref_parts else ''
                
                # 获取页码信息
                page_idx = None
                if 'prov' in resolved_item and resolved_item['prov']:
                    for prov in resolved_item['prov']:
                        if 'page_no' in prov:
                            page_idx = prov['page_no']
                            break
                        elif 'page' in prov:
                            page_idx = prov['page']
                            break

                if item_type_key == 'texts':
                    text_content = resolved_item.get("text", "").strip()
                    if text_content:  # 只添加非空文本
                        content_item = {
                            "text": text_content,
                            "level": resolved_item.get("level", 0),
                            "type": "text",
                            "page_idx": page_idx
                        }
                        content_list.append(content_item)
                        
                elif item_type_key == 'pictures':
                    content_item = {
                        "text": f"[图片: {resolved_item.get('self_ref', 'unknown')}]",
                        "level": resolved_item.get("level", 0),
                        "type": "image",
                        "page_idx": page_idx
                    }
                    content_list.append(content_item)
                    
                elif item_type_key == 'tables':
                    # 提取表格的完整数据并转换为markdown格式
                    table_text = "[表格]"
                    if 'data' in resolved_item:
                        table_data = resolved_item['data']
                        if isinstance(table_data, dict) and 'table_cells' in table_data:
                            table_text = self._convert_table_cells_to_markdown(table_data['table_cells'])
                    
                    content_item = {
                        "text": table_text,
                        "level": resolved_item.get("level", 0),
                        "type": "table",
                        "page_idx": page_idx
                    }
                    content_list.append(content_item)

                # 递归处理子节点
                if 'children' in resolved_item and resolved_item['children']:
                    extract_from_children(resolved_item['children'], doc_dict)
        
        # 开始提取
        doc_dict = docling_document.export_to_dict()
        
        # 处理主体内容 (body)
        if docling_document.body and docling_document.body.children:
            extract_from_children(docling_document.body.children, doc_dict)
        
        
        logger.info(f"从DoclingDocument中提取了 {len(content_list)} 个内容项 (包括主体和furniture部分)")
        return content_list

    def _enhance_markdown_from_content_list(self, original_markdown: str, content_list: List[Dict[str, Any]]) -> str:
        """基于content_list对markdown进行增强处理
        
        Args:
            original_markdown: 原始markdown文本
            content_list: 内容列表
            
        Returns:
            str: 增强后的markdown文本
        """
        # 将content_list中的文本内容进行层级调整和结构化处理
        enhanced_lines = []
        
        # 动态层级管理
        pattern_to_level = {}  # pattern索引 -> level映射
        curr_level = 1  # 当前层级
        skip_next = False  # 标记是否跳过下一个项目（用于跳过目录内容）

        for item in content_list:
            if item["type"] != "text":
                # 非文本内容直接添加
                if item["type"] == "image":
                    enhanced_lines.append(f"![Image]({item.get('text', '')})") 
                elif item["type"] == "table":
                    enhanced_lines.append(item.get('text', ''))
                continue
                
            text = item.get("text", "").strip()
            if not text:
                continue
            
            # 删除文本开头的所有#号及其后面的空格
            text = re.sub(r'^#+\s*', '', text)

            # 检查是否是目录标题
            if text.strip() == "目录" and item.get('level') == 1:
                # 跳过目录标题，并标记跳过下一个包含目录内容的项目
                skip_next = True
                continue
            
            # 如果需要跳过（目录内容），则跳过当前项目
            if skip_next:
                # 检查下一个项目是否包含章节列表（通常包含"第一章"、"第二章"等）
                if "第一章" in text or "第二章" in text or "第三章" in text:
                    skip_next = False  # 重置跳过标记
                    continue  # 跳过目录内容
                skip_next = False  # 如果不是目录内容，重置标记
            
            # 检测模式
            heading = text[:10]
            pattern_idx, no = self._detect_pattern(heading)
           
            if pattern_idx >= 0:  # 匹配到模式
                if pattern_idx in pattern_to_level:
                    # 已知模式，使用已分配的level
                    curr_level = pattern_to_level[pattern_idx]
                else:
                    # 新模式，curr_level+1并记录关联
                    curr_level += 1
                    pattern_to_level[pattern_idx] = curr_level
                
                # 生成 Markdown
                if self.has_chinese_punctuation(text) and len(text) > 20:  # 判断是否为句子
                    # 如果是句子，构造标题并分离内容
                    title_part = no.strip()
                    content_part = text.replace(no, "").strip()
                    enhanced_lines.append(f"{'#' * curr_level} {title_part}")
                    enhanced_lines.append("")
                    enhanced_lines.append(content_part)
                else:
                    # 直接使用匹配到的内容作为标题
                    enhanced_lines.append(f"{'#' * curr_level} {text}")
            else:
                # 未匹配到模式，使用原始level字段
                if item.get('level'):
                    # 使用原始level作为curr_level，并重新记录关联
                    curr_level = item.get('level')
                    pattern_to_level = {}
                    enhanced_lines.append(f"{'#' * curr_level} {text}")
                else:
                    # 作为普通文本
                    enhanced_lines.append(text)
            
            enhanced_lines.append("")  # 添加空行分隔
        
        enhanced_markdown = "\n".join(enhanced_lines).strip()
        logger.info(f"Markdown增强处理完成，生成了 {len(enhanced_lines)} 行内容")
        return enhanced_markdown

    def collect_document(self, parse_task: ParseTask) -> List[Document]:
        """从解析任务中收集文档（只处理子任务）
        
        Args:
            parse_task: 解析任务（子任务）
            
        Returns:
            List[Document]: 文档列表
        """
        # 校验任务状态
        if parse_task.status != TaskStatus.COMPLETED:
            raise ValueError(f"Parse task {parse_task.task_id} is not completed.")

        # 只处理子任务
        if parse_task.subtasks:
            raise ValueError(f"DoclingProcessor.collect_document should only process subtasks, not main tasks")
        
        documents = []
        logger.info(f"Processing Docling subtask {parse_task.task_id}")
        
        # 校验解析任务是否有输出目录
        output_dir = parse_task.output_directory
        if not output_dir:
            raise ValueError(f"No output directory found for parse task {parse_task.task_id}")
        
        directory_path = Path(output_dir)

        try:
            # 获取原始文档名称,并移除后缀
            base_file_name, _ = os.path.splitext(parse_task.file_name)
            content_file = directory_path / f"{base_file_name}.md"
            content_json_path = directory_path / f"{base_file_name}.json"
            
            if content_file.exists():
                # 读取markdown内容
                content = content_file.read_text(encoding='utf-8')
                # 初始元数据
                metadata = {
                    "source_file": str(content_json_path),
                    "original_file_path": parse_task.file_path,
                    "file_size": parse_task.file_size,
                    "mime_type": parse_task.mime_type
                }
                content_json_data = None
                if content_json_path.exists():
                    try:
                        with open(content_json_path, 'r', encoding='utf-8') as f:
                            content_json_data = json.load(f)
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to decode JSON for {content_json_path}")
                
                # 优先使用markdown来作为文档内容
                if content:
                    metadata['file_type'] = '.md'
                    doc = Document(text=content, metadata=metadata)
                    metadata['file_type'] = '.json'
                else:
                    doc = Document(text=json.dumps(content_json_data, ensure_ascii=False), metadata=metadata)
                documents.append(doc)
                    
        except Exception as e:
            logger.error(f"收集文档出错：{str(e)}")
            raise
        
        return documents
