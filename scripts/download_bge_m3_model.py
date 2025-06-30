import os
from huggingface_hub import snapshot_download
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

if __name__ == '__main__':
    target_dir = os.path.join('models_dir', 'bge_m3')
    os.makedirs(target_dir, exist_ok=True)
    bge_m3_model_dir = snapshot_download(
        repo_id='BAAI/bge-m3', 
        local_dir=target_dir,
        local_dir_use_symlinks=False
    )
    print(f'bge_m3 models downloaded to: {bge_m3_model_dir}')
