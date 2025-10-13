from pathlib import Path
import json
import random
from pathlib import Path
import json
import random

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"

PROCESSED_DIR = DATA_DIR / "processed"
UPRANK_JSON = PROCESSED_DIR / "uprank.json"
TEST_JSON = PROCESSED_DIR / "test.json"

# Xóa file uprank.json và test.json nếu đã tồn tại
for file_path in [UPRANK_JSON, TEST_JSON]:
    if file_path.exists():
        file_path.unlink()


# 1️⃣ Đọc tất cả đường dẫn query file
query_files = []

for dataset_folder in PROCESSED_DIR.iterdir():
    if not dataset_folder.is_dir():
        continue
    for version_folder in dataset_folder.iterdir():
        queries_dir = version_folder / "queries"
        if queries_dir.exists():
            for q_file in queries_dir.glob("*.json"):
                query_files.append(str(q_file.relative_to(BASE_DIR)))

print(f"Found {len(query_files)} query files across all datasets and versions.")

# 2️⃣ Shuffle và chia queries 8:2
random.seed(42)
random.shuffle(query_files)
split_idx = int(len(query_files) * 0.8)

uprank_queries = query_files[:split_idx]
test_queries = query_files[split_idx:]

# 3️⃣ Lưu ra JSON
with open(UPRANK_JSON, "w", encoding="utf-8") as f:
    json.dump(uprank_queries, f, indent=2, ensure_ascii=False)

with open(TEST_JSON, "w", encoding="utf-8") as f:
    json.dump(test_queries, f, indent=2, ensure_ascii=False)

print(f"Saved {len(uprank_queries)} queries to {UPRANK_JSON}")
print(f"Saved {len(test_queries)} queries to {TEST_JSON}")