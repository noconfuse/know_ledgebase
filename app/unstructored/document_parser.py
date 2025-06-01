

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.readers.file import UnstructuredReader
from llama_index.core import Settings
from pathlib import Path
loader = UnstructuredReader()

def parse_documents(input_pdf_path:str):
    print(input_pdf_path)
    documents = loader.load_data(
        file=Path(input_pdf_path),   # 支持PDF/DOCX/PPTX等格式
        split_documents=False,                  # 自动分块
    )
    return documents