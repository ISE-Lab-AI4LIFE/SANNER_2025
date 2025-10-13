from pathlib import Path
import sys
import warnings
import logging

# Thêm folder gốc của project vào sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

import json
import numpy as np

# Suppress transformers logging and warnings
import transformers
transformers.logging.set_verbosity_error()
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

from tqdm import tqdm
from src.embedding.embedder import AverageEmbedder

# Tự động phát hiện thư mục gốc của repo (SANNER_2025)
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"

# Các thư mục dữ liệu
PROCESSED_DIR = DATA_DIR / "processed"
EMBEDDING_DIR = DATA_DIR / "embedding"

def chunk_by_chars(text, max_length=2000, stride=1000):
    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = min(start + max_length, text_length)
        chunks.append(text[start:end])
        if end == text_length:
            break
        start += stride
    return chunks

def chunk_embedding(text, embedder):
    chunks = chunk_by_chars(text)
    embeddings = []
    for chunk in chunks:
        emb = embedder.embed(chunk)  # <-- dùng .embed()
        embeddings.append(emb)
    if embeddings:
        return np.mean(embeddings, axis=0)
    else:
        return np.array([])

def process_subfolder(subfolder_path: Path, embedder, test_mode: bool = False):
    for sub_subfolder in subfolder_path.iterdir():
        if sub_subfolder.is_dir():
            embedding_path = EMBEDDING_DIR / subfolder_path.name / sub_subfolder.name

            if embedding_path.exists():
                print(f"Skipping {embedding_path}, already exists.")
                continue

            for category in ["queries", "documents"]:
                source_folder = sub_subfolder / category
                if not source_folder.exists():
                    print(f"{source_folder} does not exist, skipping.")
                    continue

                target_folder = EMBEDDING_DIR / subfolder_path.name / sub_subfolder.name / category
                target_folder.mkdir(parents=True, exist_ok=True)

                json_files = list(source_folder.glob("*.json"))
                if test_mode:
                    json_files = json_files[:2]

                for json_file in tqdm(json_files, desc=f"Processing {category} in {sub_subfolder.name}"):
                    with open(json_file, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    text = data.get("text", "")
                    embedding = chunk_embedding(text, embedder)

                    file_id = data.get("id")
                    npy_filename = f"{subfolder_path.name}_{file_id}.npy"
                    npy_path = target_folder / npy_filename

                    np.save(npy_path, embedding)
                    print(f"Saved embedding to {npy_path}")

def main():
    embedder = AverageEmbedder()
    for subfolder in PROCESSED_DIR.iterdir():
        if subfolder.is_dir():
            process_subfolder(subfolder, embedder, test_mode=False)
            
if __name__ == "__main__":
    main()