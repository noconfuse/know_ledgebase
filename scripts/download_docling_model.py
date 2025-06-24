import os
from huggingface_hub import snapshot_download
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

if __name__ == '__main__':
    # 下载ds4sd/docling-models仓库的全部内容到 models_dir/docling-models
    target_dir = os.path.join('models_dir', 'docling')
    os.makedirs(target_dir, exist_ok=True)
    local_path = snapshot_download(
        repo_id="ds4sd/docling-models",
        local_dir=target_dir,
        local_dir_use_symlinks=False
    )
    from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
    StandardPdfPipeline.download_models_hf(local_dir="models_dir/docling")
    print(f'Docling models repo downloaded to: {local_path}')
