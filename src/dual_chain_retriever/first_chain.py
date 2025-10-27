import os
from pathlib import Path
import numpy as np
import json
from tqdm import tqdm
import pandas as pd

# Lấy thư mục gốc của repo (SANNER_2025)
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" 

EMBEDDING_DIR = DATA_DIR / "embedding"
OUTPUT_FILE = DATA_DIR / "queries_to_document.json"
UPRANK_FILE = DATA_DIR / "processed" / "uprank.csv"

def cosine_similarity(a: np.ndarray, b: np.ndarray):
    """Tính cosine similarity giữa hai vector numpy"""
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def load_embeddings(folder: Path):
    """Đọc tất cả file CSV trong data/queries/<subfolder>/*.csv và data/documents/<subfolder>/*.csv,
    lấy id từ cột 'id', rồi load các file .npy tương ứng trong data/embedding/queries và data/embedding/documents"""
    queries_ids = set()
    documents_ids = set()

    # Đọc queries_ids từ uprank.csv
    if UPRANK_FILE.exists():
        df_q = pd.read_csv(UPRANK_FILE)
        if 'id' in df_q.columns:
            queries_ids.update(df_q['id'].astype(str).tolist())

    documents_csv_dir = DATA_DIR / "processed" / "documents"

    # Đọc tất cả CSV trong documents
    if documents_csv_dir.exists():
        for subfolder in documents_csv_dir.iterdir():
            if not subfolder.is_dir():
                continue
            for csv_file in subfolder.glob("*.csv"):
                df = pd.read_csv(csv_file)
                if 'id' in df.columns:
                    documents_ids.update(df['id'].astype(str).tolist())

    queries_dict = {}
    documents_dict = {}

    queries_embedding_dir = EMBEDDING_DIR / "queries"
    documents_embedding_dir = EMBEDDING_DIR / "documents"

    # Load query embeddings
    if queries_embedding_dir.exists():
        for q_id in queries_ids:
            npy_path = queries_embedding_dir / f"{q_id}.npy"
            if npy_path.exists():
                queries_dict[q_id] = np.load(npy_path)
    # Load document embeddings
    if documents_embedding_dir.exists():
        for d_id in documents_ids:
            npy_path = documents_embedding_dir / f"{d_id}.npy"
            if npy_path.exists():
                documents_dict[d_id] = np.load(npy_path)
    return queries_dict, documents_dict

def load_uprank_queries(uprank_path: Path):
    if not uprank_path.exists():
        return set()
    df = pd.read_csv(uprank_path)
    if 'id' in df.columns:
        return set(df['id'].astype(str).tolist())
    return set()

def find_top_k(queries_dict, documents_dict, top_k=10):
    results = []

    doc_ids = list(documents_dict.keys())
    doc_vectors = np.stack([documents_dict[d] for d in doc_ids], axis=0)

    for q_id, q_vec in tqdm(queries_dict.items(), desc="Computing top-k"):
        sims = np.dot(doc_vectors, q_vec) / (np.linalg.norm(doc_vectors, axis=1) * np.linalg.norm(q_vec) + 1e-8)
        top_indices = sims.argsort()[::-1][:top_k]
        top_docs = [doc_ids[i] for i in top_indices]
        top_scores = [float(sims[i]) for i in top_indices]  # thêm điểm cosine sim

        results.append({
            "query_id": q_id,
            "top_documents": top_docs,
            "scores": top_scores
        })

    return results

def main():
    queries_dict, documents_dict = load_embeddings(EMBEDDING_DIR)
    print(f"Loaded {len(queries_dict)} queries and {len(documents_dict)} documents.")

    uprank_queries = load_uprank_queries(UPRANK_FILE)
    if uprank_queries:
        # Lọc queries_dict chỉ giữ các query_id trong uprank_queries
        queries_dict = {k: v for k, v in queries_dict.items() if k in uprank_queries}
        print(f"Filtered queries to {len(queries_dict)} based on uprank.csv")

    results = find_top_k(queries_dict, documents_dict, top_k=10)

    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)
        print(f"Removed existing file: {OUTPUT_FILE}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Saved top-5 results to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()