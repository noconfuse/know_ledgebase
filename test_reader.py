from app.common.struct_to_sqlite import DirectoryDocumentParser
import os
project_dir = "projects/ship_check"
target_file_path = os.path.join(project_dir, "documents", "国际船舶录")
document_parser = DirectoryDocumentParser(target_file_path)
document_parser.load_data("业务资讯服务-国际船舶录-Detail.html",show_progress=True)