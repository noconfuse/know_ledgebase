import os
import re
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Union, Dict, List
from pathlib import Path
from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
from magic_pdf.config.enums import SupportedPdfParseMethod


class PdfParser:
    def __init__(self):
        """初始化 PDF 解析器"""
        self.hierarchy_rules = [
            {"pattern": r"^(第[一二三四五六七八九十\d\s]+章\s+)", "level": 1},
            {"pattern": r"^(第[\d\s]+节\s+)", "level": 2},
            {"pattern": r"^(\d+\.\d+\.\d+\s+)", "level": 3},  # 3.1.2
            {"pattern": r"^(\d+\.\d+\.\d+\.\d+\s+)", "level": 4},  # 3.1.2.1
            {"pattern": r"^(\d+\.\d+\s+)", "level": 2},  # 3.1
            {"pattern": r"^([一二三四五六七八九十]+\s+)", "level": 2},
        ]
        self.standard_level_name = {
            1: "章",
            2: "节",
            3: "条",
            4: "项"
        }

    def has_chinese_punctuation(self, text: str) -> bool:
        """判断文本是否包含中文标点符号"""
        # 定义中文标点符号集合（可根据实际文档特征扩展）
        CHINESE_PUNCTUATION = {'，', '。', '；', '：', '？', '！', '“', '”', '、', '《', '》'}
        return any(char in CHINESE_PUNCTUATION for char in text)

    def parse(self, input_path: str, output_root: str, max_workers: int = 1) -> Union[Dict, List[Dict]]:
        """
        统一入口方法，支持处理单个文件或目录
        参数：
        input_path - 输入路径（文件或目录）
        output_root - 输出根目录
        max_workers - 并发线程数量

        返回：
        单个文件：返回结果路径字典
        多个文件：返回字典列表，key 为文件名，value 为结果路径
        """
        if os.path.isfile(input_path):
            return self._parse_single_pdf(input_path, output_root)
        elif os.path.isdir(input_path):
            return self._parse_batch_pdfs(input_path, output_root, max_workers)
        else:
            raise ValueError(f"Invalid input path: {input_path}")

    def _parse_single_pdf(self, pdf_path: str, output_root: str) -> Dict:
        """处理单个 PDF 文件"""
        file_stem = Path(pdf_path).stem
        # 直接使用传入的output_root作为输出目录，不再创建子目录
        output_dir = output_root
        os.makedirs(output_dir, exist_ok=True)

        # 原 parse_pdf 方法内容
        image_dir, image_writer, md_writer = self._prepare_output_dirs(output_dir)
        pdf_bytes = self._read_pdf_content(pdf_path)

        ds = PymuDocDataset(pdf_bytes)
        parse_method = ds.classify()

        if parse_method == SupportedPdfParseMethod.OCR:
            infer_result = ds.apply(doc_analyze, ocr=True)
            pipe_result = infer_result.pipe_ocr_mode(image_writer)
        else:
            infer_result = ds.apply(doc_analyze, ocr=False)
            pipe_result = infer_result.pipe_txt_mode(image_writer)

        result_paths = self._generate_output_files(
            output_dir,
            file_stem,
            infer_result,
            pipe_result,
            md_writer,
            image_dir
        )

        return result_paths

    def _parse_batch_pdfs(self, dir_path: str, output_root: str, max_workers: int) -> Dict[str, Dict]:
        """批量处理目录中的 PDF 文件"""
        results = {}
        pdf_files = [os.path.join(dir_path, file_name) for file_name in os.listdir(dir_path) if
                     file_name.lower().endswith('.pdf')]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._parse_single_pdf, pdf_path, output_root): pdf_path for pdf_path in
                       pdf_files}
            for future in futures:
                pdf_path = futures[future]
                file_name = os.path.basename(pdf_path)
                try:
                    result = future.result()
                    results[file_name] = result
                except Exception as e:
                    print(f"Error processing {file_name}: {str(e)}")
                    results[file_name] = {"error": str(e)}

        return results

    def _get_file_basename(self, file_path: str) -> str:
        """从文件路径提取不含扩展名的文件名"""
        filename = os.path.basename(file_path)
        return os.path.splitext(filename)[0]

    def _prepare_output_dirs(self, output_dir: str):
        """创建并初始化输出目录结构"""
        image_dir = os.path.join(output_dir, "images")
        os.makedirs(image_dir, exist_ok=True)

        return (
            os.path.basename(image_dir),  # 图片相对路径
            FileBasedDataWriter(image_dir),
            FileBasedDataWriter(output_dir),
        )

    def _read_pdf_content(self, pdf_path: str) -> bytes:
        """读取 PDF 二进制内容"""
        base_dir = os.path.dirname(pdf_path)
        filename = os.path.basename(pdf_path)
        return FileBasedDataReader(base_dir).read(filename)

    def _generate_output_files(self, output_dir, base_name, infer_result, pipe_result, md_writer, img_rel_dir):
        """生成所有输出文件并返回路径字典"""
        # 路径构造辅助方法
        def build_path(suffix): return os.path.join(output_dir, f"{base_name}{suffix}")

        # 可视化输出文件
        visual_outputs = {
            "model_visualization": build_path("_model.pdf"),
            "layout_visualization": build_path("_layout.pdf"),
            "spans_visualization": build_path("_spans.pdf"),
        }
        infer_result.draw_model(visual_outputs["model_visualization"])
        pipe_result.draw_layout(visual_outputs["layout_visualization"])
        pipe_result.draw_span(visual_outputs["spans_visualization"])

        # 核心数据输出文件 - 使用与Docling相同的文件名
        data_outputs = {
            "markdown": os.path.join(output_dir, "content.md"),
            "content_list": os.path.join(output_dir, "layout.json"),
            "intermediate_data": os.path.join(output_dir, "result.json"),
        }
        pipe_result.dump_md(md_writer, "content.md", img_rel_dir)
        pipe_result.dump_content_list(md_writer, "layout.json", img_rel_dir)
        pipe_result.dump_middle_json(md_writer, "result.json")

        return {**visual_outputs, **data_outputs}

    def _load_content_list(self, output_dir: str, file_name: str) -> list:
        # 使用统一的layout.json文件名
        path = os.path.join(output_dir, "layout.json")
        with open(path, 'r', encoding='utf-8') as f:
            content_data = json.load(f)
        return content_data

    def _detect_level(self, text: str) -> tuple[int, str]:
        """根据标题编号模式自动推断层级"""
        for rule in self.hierarchy_rules:
            match = re.match(rule["pattern"], text)
            if match:
                return rule["level"], match.group(1)
        return 0, ""

    def _process_content(self, content_data: list) -> tuple[list, str]:
        """处理 content_list 并生成新的文件和 Markdown"""
        updated_content = []
        markdown_lines = []

        for item in content_data:
            processed = item.copy()
            md_line = ""

            # 通用字段处理
            if item["type"] == "text":
                text: str = item.get("text", "")
                # 检测层级并设置默认值
                heading = text[:10]
                level, no = self._detect_level(heading)

                # 生成 Markdown
                if level > 0:
                    processed["text_level"] = level

                curr_level = processed.get("text_level", 0)

                if curr_level > 0:
                    if curr_level >= 3 and self.has_chinese_punctuation(text):  # 第四层级可能是句子，重新加上标题
                        text = text.replace(no, "")
                        md_line = f"{'#' * processed['text_level']} 第 {no.strip()} {self.standard_level_name.get(curr_level)}\n\n{text}"
                    else:
                        md_line = f"{'#' * processed['text_level']} {text}"
                else:
                    md_line = text

            elif item["type"] == "image":
                md_line = f"![Image]({item.get('img_path', '')})"

            elif item["type"] == "table":
                md_line = (f"![Table]({item['img_path']})"
                           if item.get('img_path')
                           else item.get('table_body', ''))

            updated_content.append(processed)
            markdown_lines.append(md_line)

        return updated_content, "\n\n".join(markdown_lines)

    def _save_json(self, data: list, output_dir: str, base_name: str, suffix: str) -> str:
        # 使用统一的文件名：layout.json用于内容列表，result.json用于其他数据
        if suffix == "content_list":
            path = os.path.join(output_dir, "layout.json")
        else:
            path = os.path.join(output_dir, "result.json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return path

    def _save_markdown(self, content: str, output_dir: str, base_name: str) -> str:
        # 使用统一的文件名：content.md
        path = os.path.join(output_dir, "content.md")
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        return path

    def reporcess_outputs(self, input_path: str, output_root: str):
        """
        重新处理输出文件，支持传入文件目录
        """
        results = {}
        if os.path.isfile(input_path) and input_path.endswith("_content_list.json"):
            file_base_name = self._get_file_basename(input_path).replace("_content_list", "")
            output_dir = os.path.join(output_root, file_base_name)
            os.makedirs(output_dir, exist_ok=True)
            content_data = self._load_content_list(os.path.dirname(input_path), file_base_name)
            new_content_list, new_md = self._process_content(content_data)
            new_files = {
                "updated_content_list": self._save_json(new_content_list, output_dir, file_base_name,
                                                        "content_list_updated"),
                "updated_markdown": self._save_markdown(new_md, output_dir, file_base_name)
            }
            results[file_base_name] = new_files
        elif os.path.isdir(input_path):
            for root, _, files in os.walk(input_path):
                for file in files:
                    if file.endswith("_content_list.json"):
                        file_base_name = self._get_file_basename(file).replace("_content_list", "")
                        output_dir = os.path.join(output_root, file_base_name)
                        os.makedirs(output_dir, exist_ok=True)
                        content_data = self._load_content_list(root, file_base_name)
                        new_content_list, new_md = self._process_content(content_data)
                        new_files = {
                            "updated_content_list": self._save_json(new_content_list, output_dir, file_base_name,
                                                                    "content_list_updated"),
                            "updated_markdown": self._save_markdown(new_md, output_dir, file_base_name)
                        }
                        results[file_base_name] = new_files
        else:
            raise ValueError(f"Invalid input path: {input_path}. Expected a file ending with '_content_list.json' or a directory.")
        return results

    async def async_parse(self, input_path: str, output_root: str, max_workers: int = 1):
        """异步调用 parse 方法"""
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return await loop.run_in_executor(executor, self.parse, input_path, output_root, max_workers)
    