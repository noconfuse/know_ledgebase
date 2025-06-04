from services.document_post_processor import DocumentPostProcessor
import os

output_dir = "/home/ubuntu/workspace/knowledge/outputs/e3cec68b-3cef-4bca-a484-cba73a4f1d73"

post_processor = DocumentPostProcessor()
layout_json_path = os.path.join(output_dir, "content_list.json")
print(layout_json_path)
if os.path.exists(layout_json_path):
    enhanced_result = post_processor.process_document(layout_json_path, output_dir)
    print(enhanced_result)