import json
from pathlib import Path
from collections import defaultdict
import os

# File JSON đầu vào: danh sách queries với top_documents
TOP5_JSON = Path("/Users/hieunguyen/SANNER_2025/data/queries_to_document.json")
# File JSON đầu ra: mỗi document kèm danh sách queries trỏ tới nó
DOC2QUERIES_JSON = Path("/Users/hieunguyen/SANNER_2025/data/document_to_queries.json")

def invert_top_documents(top_json_path, output_path):
    with open(top_json_path, "r", encoding="utf-8") as f:
        top_list = json.load(f)

    doc_to_queries = defaultdict(list)

    for entry in top_list:
        query_id = entry["query_id"]
        for doc_id in entry["top_documents"]:
            doc_to_queries[doc_id].append(query_id)

    # Chuyển defaultdict sang dict thường để lưu JSON
    doc_to_queries = dict(doc_to_queries)

    print(f"Number of documents: {len(doc_to_queries)}")
    print(f"Documents: {list(doc_to_queries.keys())}")

    # Tính số lượng query trỏ tới mỗi document và in trung bình queries trên 1 document
    total_queries = sum(len(queries) for queries in doc_to_queries.values())
    average_queries = total_queries / len(doc_to_queries) if doc_to_queries else 0
    print(f"Average number of queries per document: {average_queries:.2f}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(doc_to_queries, f, indent=2, ensure_ascii=False)

    print(f"Saved document-to-queries mapping to {output_path}")

if __name__ == "__main__":
    # Kiểm tra và xóa file document_to_queries.json nếu đã tồn tại
    if DOC2QUERIES_JSON.exists():
        os.remove(DOC2QUERIES_JSON)
    invert_top_documents(TOP5_JSON, DOC2QUERIES_JSON)