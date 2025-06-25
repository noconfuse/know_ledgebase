# -*- coding: utf-8 -*-
"""
This module provides a class to enhance documents from docling.json files.
"""

import json
import os
import re
from typing import List, Dict, Optional, Any, Tuple

from llama_index.core import Document

from utils.logging_config import get_logger

logger = get_logger(__name__)

class DoclingJsonEnhancer:
    """
    Enhances documents based on the content of a .docling.json file,
    replicating logic from DocumentDoclingProcessor.
    """

    def __init__(self, config: Optional[Dict]):
        """
        Initializes the DoclingJsonEnhancer.

        Args:
            config (Optional[Dict]): The enhancement configuration.
        """
        self.config = config
        self.hierarchy_patterns = [
            r"^(第[一二三四五六七八九十\-\d\s]*章)",
            r"^(第[一二三四五六七八九十\-\d\s]*条)",
            r"^(第[\d\s]*节)",
            r"^(\d+\.\d+\.\d+)\s*",
            r"^(\d+\.\d+\.\d+\.\d+)\s*",
            r"^(\d+\.\d+)\s*",
            r"^([一二三四五六七八九十]+)\s*",
        ]

    def process_files(self, md_path: str, json_path: str) -> List[Document]:
        """
        Processes the markdown and json files to create enhanced documents.

        Args:
            md_path (str): Path to the markdown file.
            json_path (str): Path to the docling.json file.

        Returns:
            List[Document]: A list of enhanced documents, typically a single document with enhanced markdown.
        """
        logger.info(f"Enhancing document using {json_path}")
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                docling_data = json.load(f)
            with open(md_path, 'r', encoding='utf-8') as f:
                original_markdown = f.read()
        except (IOError, json.JSONDecodeError) as e:
            logger.error(f"Could not read or parse {json_path} or {md_path}: {e}")
            return []

        content_list = self._extract_content_list_from_json(docling_data)
        enhanced_markdown = self._enhance_markdown_from_content_list(original_markdown, content_list)

        metadata = {
            "source_json": json_path,
            "source_markdown": md_path
        }
        enhanced_document = Document(text=enhanced_markdown, metadata=metadata)
        
        return [enhanced_document]

    def _extract_content_list_from_json(self, docling_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extracts content_list from a docling JSON dictionary."""
        content_list = []

        def resolve_ref(ref: str, doc_dict: Dict[str, Any]) -> Any:
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

        def extract_from_children(children: List[Dict[str, str]], doc_dict: Dict[str, Any], page_idx: int):
            for child_ref in children:
                ref_str = child_ref.get('$ref')
                if not ref_str:
                    continue
                
                child_obj = resolve_ref(ref_str, doc_dict)
                if not child_obj:
                    continue

                obj_type = child_obj.get('label')
                text = child_obj.get('text', '')
                level = child_obj.get('level')

                item = {'type': obj_type, 'text': text, 'level': level, 'page_idx': page_idx}
                
                if obj_type == 'table':
                    table_cells = child_obj.get('data', {}).get('table_cells', [])
                    item['text'] = self._convert_table_cells_to_markdown(table_cells)
                
                content_list.append(item)

                if 'children' in child_obj and child_obj['children']:
                    extract_from_children(child_obj['children'], doc_dict, page_idx)

        # Logic to handle different JSON structures
        if 'pages' in docling_data and docling_data['pages']:
             for i, page_ref in enumerate(docling_data['pages']):
                page_obj = resolve_ref(page_ref['$ref'], docling_data)
                if page_obj and 'children' in page_obj:
                    extract_from_children(page_obj['children'], docling_data, i)
        elif 'body' in docling_data and 'children' in docling_data['body']:
            extract_from_children(docling_data['body']['children'], docling_data, 0)

        return content_list

    def _enhance_markdown_from_content_list(self, original_markdown: str, content_list: List[Dict[str, Any]]) -> str:
        """Enhances markdown based on the content_list."""
        enhanced_lines = []
        pattern_to_level = {}
        curr_level = 1
        skip_next = False

        for item in content_list:
            if item["type"] != "text":
                if item["type"] == "image":
                    enhanced_lines.append(f"![Image]({item.get('text', '')})")
                elif item["type"] == "table":
                    enhanced_lines.append(item.get('text', ''))
                continue
            
            text = item.get("text", "").strip()
            if not text:
                continue
            
            text = re.sub(r'^#+\s*', '', text)

            if text.strip() == "目录" and item.get('level') == 1:
                skip_next = True
                continue
            
            if skip_next:
                if "第一章" in text or "第二章" in text or "第三章" in text:
                    skip_next = False
                    continue
                skip_next = False
            
            heading = text[:10]
            pattern_idx, no = self._detect_pattern(heading)
           
            if pattern_idx >= 0:
                if pattern_idx in pattern_to_level:
                    curr_level = pattern_to_level[pattern_idx]
                else:
                    curr_level += 1
                    pattern_to_level[pattern_idx] = curr_level
                
                if self.has_chinese_punctuation(text) and len(text) > 20:
                    title_part = no.strip()
                    content_part = text.replace(no, "").strip()
                    enhanced_lines.append(f"{'#' * curr_level} {title_part}")
                    enhanced_lines.append("")
                    enhanced_lines.append(content_part)
                else:
                    enhanced_lines.append(f"{'#' * curr_level} {text}")
            else:
                if item.get('level'):
                    curr_level = item.get('level')
                    pattern_to_level = {}
                    enhanced_lines.append(f"{'#' * curr_level} {text}")
                else:
                    enhanced_lines.append(text)
            
            enhanced_lines.append("")
        
        enhanced_markdown = "\n".join(enhanced_lines).strip()
        logger.info(f"Markdown enhancement complete, generated {len(enhanced_lines)} lines.")
        return enhanced_markdown

    def _detect_pattern(self, text: str) -> Tuple[int, str]:
        """Detects which hierarchy pattern the text matches."""
        for i, pattern in enumerate(self.hierarchy_patterns):
            match = re.match(pattern, text)
            if match:
                return i, match.group(1)
        return -1, ""

    def has_chinese_punctuation(self, text: str) -> bool:
        """Checks if the text contains Chinese punctuation."""
        CHINESE_PUNCTUATION = {'，', '。', '；', '：', '？', '！', '“', '”', '、', '《', '》'}
        return any(char in CHINESE_PUNCTUATION for char in text)

    def _convert_table_cells_to_markdown(self, table_cells: List[Dict[str, Any]]) -> str:
        """Converts table cell data to a markdown table."""
        if not table_cells:
            return "[Empty Table]"
        
        try:
            max_row = 0
            max_col = 0
            cell_matrix = {}
            
            for cell in table_cells:
                if not isinstance(cell, dict):
                    continue
                
                row_span = cell.get('row_span', 1)
                start_row_offset_idx = cell.get('start_row_offset_idx', 0)
                start_col_offset_idx = cell.get('start_col_offset_idx', 0)
                text = cell.get('text', '').strip()
                
                max_row = max(max_row, start_row_offset_idx + row_span)
                max_col = max(max_col, cell.get('start_col_offset_idx', 0) + cell.get('col_span', 1))
                
                for r in range(start_row_offset_idx, start_row_offset_idx + row_span):
                    for c in range(start_col_offset_idx, start_col_offset_idx + cell.get('col_span', 1)):
                        if (r, c) not in cell_matrix:
                            if r == start_row_offset_idx and c == start_col_offset_idx:
                                cell_matrix[(r, c)] = text
                            else:
                                cell_matrix[(r, c)] = ''
            
            markdown_lines = []
            for row in range(max_row):
                row_cells = [cell_matrix.get((row, col), '').replace('|', '\\|').replace('\n', ' ') for col in range(max_col)]
                markdown_lines.append('| ' + ' | '.join(row_cells) + ' |')
                if row == 0:
                    separator = '| ' + ' | '.join(['---'] * max_col) + ' |'
                    markdown_lines.append(separator)
            
            return '\n'.join(markdown_lines)
            
        except Exception as e:
            logger.warning(f"Failed to convert table to markdown: {e}")
            return f"[Table with {len(table_cells)} cells, conversion failed]"