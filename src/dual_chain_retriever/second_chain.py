import json
from pathlib import Path
from collections import defaultdict
import os

# Tự động phát hiện thư mục gốc của repo (SANNER_2025)
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"

# File JSON đầu vào: danh sách queries với top_documents
TOP5_JSON = DATA_DIR / "queries_to_document.json"

# File JSON đầu ra: mỗi document kèm danh sách queries trỏ tới nó
DOC2QUERIES_JSON = DATA_DIR / "document_to_queries.json"

def invert_top_documents(top_json_path, output_path, top_k=5):
    with open(top_json_path, "r", encoding="utf-8") as f:
        top_list = json.load(f)

    doc_to_queries = defaultdict(list)
    doc_to_scores = defaultdict(list)  # ✅ thêm dict phụ để lưu điểm

    for entry in top_list:
        query_id = entry["query_id"]
        for doc_id, score in zip(entry["top_documents"], entry["scores"]):
            doc_to_queries[doc_id].append((query_id, score))

    outrank_entries = []

    # Sắp xếp và tách top-k + phần bị loại
    for doc_id, query_scores in doc_to_queries.items():
        # Sắp xếp toàn bộ theo score giảm dần
        query_scores.sort(key=lambda x: x[1], reverse=True)

        # Chia thành hai phần: top-k và phần bị loại
        top_items = query_scores[:top_k]
        outranked_items = query_scores[top_k:]

        # Lưu kết quả top-k
        doc_to_queries[doc_id] = [q for q, _ in top_items]
        doc_to_scores[doc_id] = [s for _, s in top_items]

        # Lưu phần bị loại
        outrank_entries.append({
            "doc_id": doc_id,
            "outranked_queries": [{"query_id": q, "score": s} for q, s in outranked_items]
        })

    # Chuyển defaultdict sang dict thường
    output_data = dict(doc_to_queries)
    output_data["_scores"] = dict(doc_to_scores)  # ✅ thêm trường phụ

    print(f"Number of documents: {len(doc_to_queries)}")
    total_queries = sum(len(queries) for queries in doc_to_queries.values())
    avg_queries = total_queries / len(doc_to_queries) if doc_to_queries else 0
    print(f"Average number of queries per document: {avg_queries:.2f}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"Saved document-to-queries mapping to {output_path}")

    outrank_path = DATA_DIR / "outrank.json"
    with open(outrank_path, "w", encoding="utf-8") as f:
        json.dump(outrank_entries, f, indent=2, ensure_ascii=False)

    print(f"Saved outranked queries to {outrank_path}")

if __name__ == "__main__":
    # Kiểm tra và xóa file document_to_queries.json nếu đã tồn tại
    if DOC2QUERIES_JSON.exists():
        os.remove(DOC2QUERIES_JSON)
    invert_top_documents(TOP5_JSON, DOC2QUERIES_JSON, top_k=15)