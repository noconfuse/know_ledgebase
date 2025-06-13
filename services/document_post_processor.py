# -*- coding: utf-8 -*-
import os
import re
import json
from typing import Dict, List, Tuple, Any
from pathlib import Path
import logging

from utils.logging_config import get_logger

logger = get_logger(__name__)

class DocumentPostProcessor:
    """文档后处理器，用于增强解析后文件的结构化"""
    
    def __init__(self):
        """初始化文档后处理器"""
        self.hierarchy_patterns = [
            r"^(第[一二三四五六七八九十\d\s]*章)",
            r"^(第[一二三四五六七八九十\d\s]*条)", # 第一条
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
    
    def _detect_pattern(self, text: str) -> Tuple[int, str]:
        """检测文本匹配的模式索引"""
        for i, pattern in enumerate(self.hierarchy_patterns):
            match = re.match(pattern, text)
            if match:
                return i, match.group(1)
        return -1, ""
    
    def process_content(self, content_data: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], str]:
        """处理内容列表并生成新的结构化内容和Markdown
        
        Args:
            content_data: 解析后的内容列表数据
            
        Returns:
            Tuple[List, str]: 更新后的内容列表和Markdown文本
        """
        updated_content = []
        markdown_lines = []
        
        # 动态层级管理
        pattern_to_level = {}  # pattern索引 -> level映射
        curr_level = 1  # 当前层级
        skip_next = False  # 标记是否跳过下一个项目（用于跳过目录内容）

        for i, item in enumerate(content_data):
            processed = item.copy()
            md_line = ""

            # 通用字段处理
            if item["type"] == "text":
                text: str = item.get("text", "")
                if not text.strip():
                    continue
                
                # 检查是否是目录标题
                if text.strip() == "目录" and item.get('text_level') == 1:
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

                    processed["text_level"] = curr_level
                    
                    # 生成 Markdown
                    if self.has_chinese_punctuation(text) and len(text) > 20:  # 判断是否为句子
                        # 如果是句子，构造标题并分离内容
                        title_part = no.strip()
                        content_part = text.replace(no, "").strip()
                        md_line = f"{'#' * curr_level} {title_part}\n\n{content_part}"
                    else:
                        # 直接使用匹配到的内容作为标题
                        md_line = f"{'#' * curr_level} {text}"
                else:
                    # 未匹配到模式，使用原始text_level字段
                    if item.get('text_level'):
                        # 使用原始text_level作为curr_level，并重新记录关联
                        curr_level = item.get('text_level')
                        processed["text_level"] = curr_level
                        pattern_to_level = {}
                        md_line = f"{'#' * curr_level} {text}"
                    else:
                        # 作为普通文本
                        md_line = text

            elif item["type"] == "image":
                md_line = f"![Image]({item.get('img_path', '')})"

            elif item["type"] == "table":
                md_line = f"{item.get('table_body', '')}"
            updated_content.append(processed)
            markdown_lines.append(md_line)

        return updated_content, "\n\n".join(markdown_lines)
    
    def process_document(self, input_path: str, output_dir: str) -> Dict[str, Any]:
        """处理文档，增强其结构化
        
        Args:
            input_path: 输入文件路径（layout.json）或目录
            output_dir: 输出目录
            
        Returns:
            Dict: 处理结果信息
        """
        results = {}
        os.makedirs(output_dir, exist_ok=True)

        print(os.path.isfile(input_path) and input_path.endswith("content_list.json"))
        
        if os.path.isfile(input_path) and input_path.endswith("content_list.json"):
            # 处理单个文件
            file_dir = os.path.dirname(input_path)
            content_data = self._load_content_data(input_path)
            
            if content_data:
                new_content_list, new_md = self.process_content(content_data)
                result_files = self._save_processed_files(new_content_list, new_md, output_dir)
                results["single_file"] = result_files
                logger.info(f"已处理单个文件: {input_path}")
            
        elif os.path.isdir(input_path):
            # 处理目录中的所有文件
            for root, _, files in os.walk(input_path):
                for file in files:
                    if file.endswith("content_list.json"):
                        file_path = os.path.join(root, file)
                        content_data = self._load_content_data(file_path)
                        
                        if content_data:
                            file_output_dir = os.path.join(output_dir, os.path.basename(root))
                            os.makedirs(file_output_dir, exist_ok=True)
                            
                            new_content_list, new_md = self.process_content(content_data)
                            result_files = self._save_processed_files(new_content_list, new_md, file_output_dir)
                            results[os.path.basename(root)] = result_files
                            logger.info(f"已处理文件: {file_path}")
        else:
            raise ValueError(f"无效的输入路径: {input_path}。需要是content_list.json文件的目录。")
        
        return results
    
    def _load_content_data(self, file_path: str) -> List[Dict[str, Any]]:
        """加载内容数据"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content_data = json.load(f)
            return content_data
        except Exception as e:
            logger.error(f"加载内容数据失败: {file_path}, 错误: {e}")
            return []
    
    def _save_processed_files(self, content_list: List[Dict[str, Any]], markdown: str, output_dir: str) -> Dict[str, str]:
        """保存处理后的文件，覆盖原始解析结果"""
        try:
            # 尝试从现有的content_list.json文件中获取原始文件名
            base_name = self._get_base_name_from_directory(output_dir)
            md_path = os.path.join(output_dir, f"{base_name}.md")
            content_path = os.path.join(output_dir, f"{base_name}_content_list.json")
            
            # 备份原始文件
            if os.path.exists(content_path):
                backup_content_path = os.path.join(output_dir, f"{base_name}_content_list.bak.json")
                os.rename(content_path, backup_content_path)
                
            if os.path.exists(md_path):
                backup_md_path = os.path.join(output_dir, f"{base_name}.bak.md")
                os.rename(md_path, backup_md_path)
            
            # 保存新的内容列表文件
            with open(content_path, 'w', encoding='utf-8') as f:
                json.dump(content_list, f, ensure_ascii=False, indent=2)
            
            # 保存新的Markdown文件
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(markdown)
            
            logger.info(f"原始文件已备份并更新: {content_path}, {md_path}")
            
            return {
                "content_json": content_path,
                "markdown": md_path,
            }
        except Exception as e:
            logger.error(f"保存处理后的文件失败: {output_dir}, 错误: {e}")
            return {}
    
    def _get_base_name_from_directory(self, output_dir: str) -> str:
        """从目录中获取原始文件名（去掉扩展名）"""
        try:
            # 查找现有的content_list.json文件
            for file in os.listdir(output_dir):
                if file.endswith('_content_list.json'):
                    # 提取文件名前缀作为基础名
                    base_name = file.replace('_content_list.json', '')
                    return base_name
            
            # 如果没有找到content_list.json文件，使用目录名作为后备方案
            logger.warning(f"未找到content_list.json文件，使用目录名作为基础名: {output_dir}")
            return Path(output_dir).name
        except Exception as e:
            logger.error(f"获取基础文件名失败: {output_dir}, 错误: {e}")
            return Path(output_dir).name