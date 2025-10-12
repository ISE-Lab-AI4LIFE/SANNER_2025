from pathlib import Path
import numpy as np
import json
from tqdm import tqdm

EMBEDDING_DIR = Path("/Users/hieunguyen/SANNER_2025/data/embedding")
OUTPUT_FILE = Path("/Users/hieunguyen/SANNER_2025/data/queries_to_document.json")
UPRANK_FILE = Path("/Users/hieunguyen/SANNER_2025/data/uprank.json")

def cosine_similarity(a: np.ndarray, b: np.ndarray):
    """Tính cosine similarity giữa hai vector numpy"""
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def load_embeddings(folder: Path):
    """Đọc tất cả file .npy trong folder con queries và documents"""
    queries_dict = {}
    documents_dict = {}

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

            # Documents
            d_folder = sub_subfolder / "documents"
            if d_folder.exists():
                for d_file in d_folder.glob("*.npy"):
                    documents_dict[d_file.stem] = np.load(d_file)

    return queries_dict, documents_dict

def load_uprank_queries(uprank_path: Path):
    """Đọc file uprank.json để lấy danh sách query_id cần giữ lại"""
    if not uprank_path.exists():
        return set()
    with open(uprank_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Giả sử uprank.json là list các query_id hoặc dict có key query_id, ta lấy set các query_id
    if isinstance(data, list):
        return set(data)
    elif isinstance(data, dict):
        return set(data.keys())
    else:
        return set()

def find_top_k(queries_dict, documents_dict, top_k=5):
    """Tính cosine similarity và lấy top_k documents cho mỗi query"""
    results = []

    doc_ids = list(documents_dict.keys())
    doc_vectors = np.stack([documents_dict[d] for d in doc_ids], axis=0)

    for q_id, q_vec in tqdm(queries_dict.items(), desc="Computing top-k"):
        # cosine similarity với tất cả documents
        sims = np.dot(doc_vectors, q_vec) / (np.linalg.norm(doc_vectors, axis=1) * np.linalg.norm(q_vec) + 1e-8)
        top_indices = sims.argsort()[::-1][:top_k]  # top_k lớn nhất
        top_docs = [doc_ids[i] for i in top_indices]

        results.append({
            "query_id": q_id,
            "top_documents": top_docs
        })

    return results

def main():
    queries_dict, documents_dict = load_embeddings(EMBEDDING_DIR)
    print(f"Loaded {len(queries_dict)} queries and {len(documents_dict)} documents.")

    uprank_queries = load_uprank_queries(UPRANK_FILE)
    if uprank_queries:
        # Lọc queries_dict chỉ giữ các query_id trong uprank_queries
        queries_dict = {k: v for k, v in queries_dict.items() if k in uprank_queries}
        print(f"Filtered queries to {len(queries_dict)} based on uprank.json")

    results = find_top_k(queries_dict, documents_dict, top_k=5)

    if OUTPUT_FILE.exists():
        OUTPUT_FILE.unlink()
        print(f"Removed existing file: {OUTPUT_FILE}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Saved top-5 results to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()