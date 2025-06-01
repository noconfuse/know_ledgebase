
import os
import asyncio
from app.minerU.parse_pdf import PdfParser
from app.minerU.law_index_service import LawDocumentIndexService
from app.service.index import IndexService
from app.model.model_manager import model_manager
from llama_index.core import Settings
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from app.unstructored.document_parser import parse_documents
# from app.common.reActAgent import agent
import nest_asyncio
nest_asyncio.apply()
project_dir = "projects/ship_check"

target_file_path = os.path.join(project_dir, "documents", "国内航行海船法定检验技术规则（2020）")

output_dir = os.path.join(project_dir, "documents", "国内航行海船法定检验技术规则（2020）")

store_dir = os.path.join(project_dir, "index_store")

pdfParser = PdfParser()

# result_path = pdfParser.parse_pdf(target_file_path, output_dir)

# extra_path = pdfParser.reporcess_outputs(target_file_path, output_dir)

# print(f"解析结果已保存到: {extra_path}")

# 构建索引

async def main():
    model_manager.initialize_models()
    Settings.llm = model_manager.ship_check_llm
    # documents = parse_documents(target_file_path)
    # print(documents)
    output_path = await pdfParser.async_parse(target_file_path,output_dir)
    print(output_path)
    # lawIndexService = LawDocumentIndexService(output_dir)
    # index = await lawIndexService.parse_and_build_index("海上浮动设施检验规则（2025）")
    # query_engine = lawIndexService.index.as_query_engine()
    # response = query_engine.query("海上浮动设施检验规则法规是由哪个机构颁布的？")
    # print(response)
    # RouterQueryEngine()
    # agent.chat("海上浮动设施检验规则法规是由哪个机构颁布的？")

if __name__ == "__main__":
    asyncio.run(main())
    






