import os
from pathlib import Path
import numpy as np
import json
from tqdm import tqdm

# Lấy thư mục gốc của repo (SANNER_2025)
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"

EMBEDDING_DIR = DATA_DIR / "embedding"
OUTPUT_FILE = DATA_DIR / "queries_to_document.json"
UPRANK_FILE = DATA_DIR / "uprank.json"

def cosine_similarity(a: np.ndarray, b: np.ndarray):
    """Tính cosine similarity giữa hai vector numpy"""
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def load_embeddings(folder: Path):
    """Đọc tất cả file .npy trong folder con queries và documents"""
    queries_dict = {}
    documents_dict = {}
    query_paths = {}
    document_paths = {}

    for subfolder in folder.iterdir():
        if not subfolder.is_dir():
            continue
        for sub_subfolder in subfolder.iterdir():
            if not sub_subfolder.is_dir():
                continue
            # Queries
            q_folder = sub_subfolder / "queries"
            if q_folder.exists():
                for q_file in q_folder.glob("*.npy"):
                    queries_dict[q_file.stem] = np.load(q_file)
                    query_paths[q_file.stem] = str(q_file.relative_to(BASE_DIR))
            # Documents
            d_folder = sub_subfolder / "documents"
            if d_folder.exists():
                for d_file in d_folder.glob("*.npy"):
                    documents_dict[d_file.stem] = np.load(d_file)
                    document_paths[d_file.stem] = str(d_file.relative_to(BASE_DIR))

    return queries_dict, documents_dict, query_paths, document_paths

def load_uprank_queries(uprank_path: Path):
    """Đọc file uprank.json để lấy mapping query_id -> đường dẫn file"""
    if not uprank_path.exists():
        return {}
    with open(uprank_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return {Path(p).name: p for p in data}
    elif isinstance(data, dict):
        return {k: v for k, v in data.items()}
    else:
        return {}

def find_top_k(queries_dict, documents_dict, top_k=5, query_paths=None, document_paths=None):
    results = []
    query_paths = query_paths or {}
    document_paths = document_paths or {}

    doc_ids = list(documents_dict.keys())
    doc_vectors = np.stack([documents_dict[d] for d in doc_ids], axis=0)

    for q_id, q_vec in tqdm(queries_dict.items(), desc="Computing top-k"):
        sims = np.dot(doc_vectors, q_vec) / (np.linalg.norm(doc_vectors, axis=1) * np.linalg.norm(q_vec) + 1e-8)
        top_indices = sims.argsort()[::-1][:top_k]
        top_docs = [doc_ids[i] for i in top_indices]
        top_doc_paths = [document_paths[d] for d in top_docs if d in document_paths]

        results.append({
            "query_path": query_paths.get(q_id),
            "top_documents": top_doc_paths
        })

    return results

def main():
    queries_dict, documents_dict, query_paths, document_paths = load_embeddings(EMBEDDING_DIR)
    print(f"Loaded {len(queries_dict)} queries and {len(documents_dict)} documents.")

    uprank_queries = load_uprank_queries(UPRANK_FILE)
    if uprank_queries:
        # Lọc queries_dict chỉ giữ các query_id trong uprank_queries
        queries_dict = {k: v for k, v in queries_dict.items() if k in uprank_queries}
        print(f"Filtered queries to {len(queries_dict)} based on uprank.json")

    results = find_top_k(queries_dict, documents_dict, top_k=5, query_paths=query_paths, document_paths=document_paths)

    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)
        print(f"Removed existing file: {OUTPUT_FILE}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Saved top-5 results to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()