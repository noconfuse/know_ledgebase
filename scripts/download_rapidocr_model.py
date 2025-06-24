import os
from huggingface_hub import snapshot_download
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

if __name__ == '__main__':
    target_dir = os.path.join('models_dir', 'rapidocr')
    os.makedirs(target_dir, exist_ok=True)
    rapidocr_model_dir = snapshot_download(
        repo_id='SWHL/RapidOCR', 
        local_dir=target_dir,
        local_dir_use_symlinks=False
    )
    print(f'RapidOCR models downloaded to: {rapidocr_model_dir}')
