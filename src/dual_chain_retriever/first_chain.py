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


def cosine_similarity(a: np.ndarray, b: np.ndarray):
    """Tính cosine similarity giữa hai vector numpy"""
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def load_embeddings():
    """Load toàn bộ embeddings từ thư mục embedding/queries và embedding/documents"""
    queries_dict = {}
    documents_dict = {}

    queries_embedding_dir = EMBEDDING_DIR / "queries"
    documents_embedding_dir = EMBEDDING_DIR / "documents"

    # Load queries
    if queries_embedding_dir.exists():
        for npy_path in queries_embedding_dir.glob("*.npy"):
            q_id = npy_path.stem
            queries_dict[q_id] = np.load(npy_path)
    else:
        print(f"⚠️ Không tìm thấy thư mục {queries_embedding_dir}")

    # Load documents
    if documents_embedding_dir.exists():
        for npy_path in documents_embedding_dir.glob("*.npy"):
            d_id = npy_path.stem
            documents_dict[d_id] = np.load(npy_path)
    else:
        print(f"⚠️ Không tìm thấy thư mục {documents_embedding_dir}")

    print(f"Loaded query embeddings: {len(queries_dict)}")
    print(f"Loaded document embeddings: {len(documents_dict)}")

    return queries_dict, documents_dict


def find_top_k(queries_dict, documents_dict, top_k=10):
    results = []
    doc_ids = list(documents_dict.keys())
    doc_vectors = np.stack([documents_dict[d] for d in doc_ids], axis=0)

    for q_id, q_vec in tqdm(queries_dict.items(), desc="Computing top-k"):
        sims = np.dot(doc_vectors, q_vec) / (np.linalg.norm(doc_vectors, axis=1) * np.linalg.norm(q_vec) + 1e-8)
        top_indices = sims.argsort()[::-1][:top_k]
        top_docs = [doc_ids[i] for i in top_indices]
        top_scores = [float(sims[i]) for i in top_indices]

        results.append({
            "query_id": q_id,
            "top_documents": top_docs,
            "scores": top_scores
        })

    return results


def main():
    queries_dict, documents_dict = load_embeddings()
    print(f"Loaded {len(queries_dict)} queries and {len(documents_dict)} documents.")

    if not queries_dict:
        print("❌ Không có query nào được load. Kiểm tra lại thư mục embedding/queries/")
        return

    results = find_top_k(queries_dict, documents_dict, top_k=10)

    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)
        print(f"Removed existing file: {OUTPUT_FILE}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved top-10 results to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()