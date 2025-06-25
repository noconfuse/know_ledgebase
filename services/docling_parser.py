import os
import logging
from typing import Dict, Any, Optional
import threading

from docling.datamodel.document import ConversionResult
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TesseractCliOcrOptions, RapidOcrOptions, EasyOcrOptions
from config import settings
from utils.logging_config import get_logger
import traceback

logger = get_logger(__name__)

class DoclingParser:
    """文档解析器配置管理类"""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'converter'):
            self.converter = None

    def setup_converter(self, config: Optional[Dict[str, Any]] = None):
        """设置文档转换器"""
        try:
            logger.info("开始配置Docling文档转换器...")
            # 从config中获取配置，如果没有则使用默认设置
            ocr_enabled = config.get("ocr_enabled", settings.OCR_ENABLED) if config else settings.OCR_ENABLED
            ocr_languages = (config or {}).get("ocr_languages") or settings.OCR_LANGUAGES

            ocr_type = config.get("ocr_type", settings.OCR_TYPE) if config else settings.OCR_TYPE

            # 配置PDF处理选项
            pdf_options = PdfPipelineOptions()
            pdf_options.do_ocr = ocr_enabled
            pdf_options.do_table_structure = True
            pdf_options.table_structure_options.do_cell_matching = True
            pdf_options.artifacts_path = settings.DOCLING_MODEL_PATH
            logger.info(f"PDF处理选项配置: OCR={ocr_enabled}, 表格结构提取=True, OCR后端={ocr_type}")

            # 如果启用OCR，配置OCR后端
            if ocr_enabled:
                if ocr_type == "tesseract":
                    logger.info("配置Tesseract OCR引擎...")
                    # 支持自定义参数
                    pdf_options.ocr_options = TesseractCliOcrOptions(
                        lang=ocr_languages,
                        force_full_page_ocr=True,  # 强制全页OCR
                    )
                    logger.info(f"Tesseract OCR配置完成 - 语言: {ocr_languages}")
                elif ocr_type == "rapidorc":
                    # RapidOCR分支仅作保留，docling官方暂未支持
                    logger.info("配置RapidOCR引擎...")
                    det_model_path = os.path.join(
                        settings.RAPID_OCR_MODEL_PATH, "PP-OCRv4", "en_PP-OCRv3_det_infer.onnx"
                    )
                    rec_model_path = os.path.join(
                        settings.RAPID_OCR_MODEL_PATH, "PP-OCRv4", "ch_PP-OCRv4_rec_server_infer.onnx"
                    )
                    cls_model_path = os.path.join(
                        settings.RAPID_OCR_MODEL_PATH, "PP-OCRv3", "ch_ppocr_mobile_v2.0_cls_train.onnx"
                    )

                    pdf_options.ocr_options = RapidOcrOptions(
                        force_full_page_ocr=True,
                        det_model_path=det_model_path,
                        rec_model_path=rec_model_path,
                        cls_model_path=cls_model_path
                    )
                    logger.info(f"RapidOCR配置完成 - 语言: {ocr_languages}, GPU: {settings.USE_GPU}")
                else:
                    logger.info("配置EasyOCR引擎...")
                    pdf_options.ocr_options = EasyOcrOptions(
                        lang=ocr_languages,
                        force_full_page_ocr=True,  # 强制全页OCR
                        use_gpu=settings.USE_GPU, 
                        download_enabled=True,  # 启用模型下载以支持OCR功能
                        model_storage_directory=settings.EASY_OCR_MODEL_PATH
                    )
                    logger.info(f"EasyOCR配置完成 - 语言: {ocr_languages}, GPU: {settings.USE_GPU}")
                    logger.info(f"EasyOCR模型路径: {settings.EASY_OCR_MODEL_PATH}")
            else:
                logger.info(f"非OCR模式, 跳过OCR配置")

            # 创建转换器
            logger.info("创建DocumentConverter实例...")
            self.converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options),
                }
            )
            logger.info(f"DocumentConverter创建成功 - OCR: {ocr_enabled}, GPU: {settings.USE_GPU}, OCR后端: {ocr_type}")
            logger.info("Docling文档转换器配置完成")
        except Exception as e:
            logger.error(f"初始化DocumentConverter失败: {e}")
            logger.error(f"错误详情: {traceback.format_exc()}")
            raise
        
    def parse_file(self, file_path: str) -> ConversionResult:
        """解析文件"""
        try:
            logger.info(f"开始解析文件: {file_path}")
            result = self.converter.convert(file_path)
            logger.info(f"文件解析完成: {file_path}")
            return result
        except Exception as e:
            logger.error(f"解析文件 {file_path} 失败: {e}")

    