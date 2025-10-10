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

for json_file in json_files:
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        for item in data:
            pool["query"].extend(item.get("query", []))
            pool["document"].append(item.get("document", ""))

if pool_path.exists():
    pool_path.unlink()

with open(pool_path, "w", encoding="utf-8") as f:
    json.dump(pool, f, ensure_ascii=False, indent=2)
print(f"✅ Updated pool file: {pool_path}")