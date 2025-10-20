from pathlib import Path
import sys
import warnings
import logging

# Thêm folder gốc của project vào sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

import json
import numpy as np
import pandas as pd

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
    DOCUMENTS_DIR = PROCESSED_DIR / "documents"
    QUERIES_DIR = PROCESSED_DIR / "queries"

    for category_dir, column_name in [(DOCUMENTS_DIR, "documents"), (QUERIES_DIR, "queries")]:
        target_folder = EMBEDDING_DIR / category_dir.name
        target_folder.mkdir(parents=True, exist_ok=True)

        for subfolder in category_dir.iterdir():
            if subfolder.is_dir():
                csv_files = list(subfolder.glob("*.csv"))
                if test_mode:
                    csv_files = csv_files[:2]

                for csv_file in tqdm(csv_files, desc=f"Processing {category_dir.name} in {subfolder.name}"):
                    df = pd.read_csv(csv_file)
                    for _, row in df.iterrows():
                        file_id = row["id"]
                        npy_path = target_folder / f"{file_id}.npy"
                        if npy_path.exists():
                            print(f"Skipping existing embedding: {npy_path}")
                            continue
                        text = row.get(column_name, "")
                        embedding = chunk_embedding(text, embedder)
                        npy_path = target_folder / f"{file_id}.npy"
                        np.save(npy_path, embedding)
                        print(f"Saved embedding to {npy_path}")

def main():
    embedder = AverageEmbedder()
    process_subfolder(PROCESSED_DIR, embedder, test_mode=False)
            
if __name__ == "__main__":
    main()