import os
import json
from pathlib import Path

def build_content_list_from_markdown(markdown_content):
    """将markdown内容按段落分割并构建content_list（兼容原有逻辑）"""
    content_list = []
    lines = markdown_content.split('\n')
    for i, line in enumerate(lines):
        if line.strip():
            content_list.append({
                "type": "text",
                "text": line.strip(),
                "text_level": 1 if line.startswith('#') else 0,
                "page_idx": 1
            })
    return content_list

class MarkdownFileExporter:
    """负责将docling document导出为markdown文件"""
    def export(self, document, output_dir, file_stem):
        markdown_content = document.export_to_markdown()
        md_path = os.path.join(output_dir, f"{file_stem}.md")
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        return md_path, markdown_content

class JsonFileExporter:
    """负责将docling document导出为结构化json文件"""
    def export(self, document, output_dir, file_stem):
        doc_dict = document.to_dict()
        json_path = os.path.join(output_dir, f"{file_stem}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(doc_dict, f, ensure_ascii=False, indent=2)
        return json_path, doc_dict

def extract_tables(document):
    tables = []
    if hasattr(document, 'tables'):
        for i, table in enumerate(document.tables):
            table_data = {}
            if hasattr(table, 'export_to_dataframe'):
                df = table.export_to_dataframe()
                table_data = df.to_dict() if df is not None else {}
            table_info = {
                "table_id": i,
                "data": table_data,
                "caption": getattr(table, 'caption', ''),
                "row_count": len(table_data.get('index', [])) if table_data else 0,
                "col_count": len([k for k in table_data.keys() if k != 'index']) if table_data else 0
            }
            tables.append(table_info)
    return tables

def extract_images(document):
    images = []
    if hasattr(document, 'pictures'):
        for i, picture in enumerate(document.pictures):
            image_info = {
                "image_id": i,
                "caption": getattr(picture, 'caption', ''),
                "size": getattr(picture, 'size', {}),
                "position": getattr(picture, 'position', {}),
                "type": getattr(picture, 'type', 'unknown')
            }
            images.append(image_info)
    return images
