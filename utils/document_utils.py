import os
import json
from pathlib import Path
import hashlib

def truncate_filename(filename, max_length=60, preserve_extension=True):
    """截断文件名以避免文件系统长度限制
    
    Args:
        filename: 原始文件名
        max_length: 最大长度限制（默认60字符，因为存在中文文件名）
        preserve_extension: 是否保留文件扩展名
    
    Returns:
        截断后的文件名
    """
    if len(filename) <= max_length:
        return filename
    
    if preserve_extension:
        name, ext = os.path.splitext(filename)
        # 为扩展名和哈希值预留空间
        available_length = max_length - len(ext) - 9  # 9 = '_' + 8位哈希
        if available_length > 0:
            # 使用原文件名的哈希值作为唯一标识
            name_hash = hashlib.md5(name.encode('utf-8')).hexdigest()[:8]
            truncated_name = name[:available_length] + '_' + name_hash
            return truncated_name + ext
        else:
            # 如果扩展名太长，只保留哈希值
            name_hash = hashlib.md5(filename.encode('utf-8')).hexdigest()[:8]
            return name_hash + ext
    else:
        # 不保留扩展名的情况
        name_hash = hashlib.md5(filename.encode('utf-8')).hexdigest()[:8]
        return filename[:max_length-9] + '_' + name_hash

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
