from pathlib import Path
import json
import os

root_dir = Path("/Users/hieunguyen/SANNER_2025/data/processed")

# Tìm tất cả file .json trong thư mục và các thư mục con
json_files = list(root_dir.rglob("*.json"))

pool_path = Path("/Users/hieunguyen/SANNER_2025/data/pool/pool.json")
if pool_path.exists():
    os.remove(pool_path)
    print("⚠️ Đã xóa file pool.json cũ trước khi tạo mới.")

pool = {"query": [], "document": []}
query_idx = 0  # bộ đếm id riêng cho query
doc_idx = 0    # bộ đếm id riêng cho document

for json_file in json_files:
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        for item in data:
            for q in item.get("query", []):
                pool["query"].append({"id": query_idx, "text": q})
                query_idx += 1
            pool["document"].append({"id": doc_idx, "text": item.get("document", "")})
            doc_idx += 1

if pool_path.exists():
    pool_path.unlink()

with open(pool_path, "w", encoding="utf-8") as f:
    json.dump(pool, f, ensure_ascii=False, indent=2)
print(f"✅ Updated pool file with separate query/document IDs: {pool_path}")