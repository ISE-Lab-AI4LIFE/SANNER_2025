from pathlib import Path
import random
import pandas as pd
import ast

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"

# Xóa file test.csv nếu đã tồn tại
test_file = PROCESSED_DIR / "test.csv"
if test_file.exists():
    test_file.unlink()

# Đọc tất cả query_id từ các file processed/queries
QUERIES_DIR = PROCESSED_DIR / "queries"
query_files = list(QUERIES_DIR.glob("*/*.csv"))

all_query_ids = set()
id_to_query = {}

for q_file in query_files:
    df = pd.read_csv(q_file)
    if "id" in df.columns and "queries" in df.columns:
        for _, row in df.iterrows():
            qid = str(row["id"])
            all_query_ids.add(qid)
            id_to_query[qid] = row["queries"]
    else:
        raise ValueError(f"File {q_file} không có cột 'id' hoặc 'query'")

# Đọc file document_query_pairs.csv
pairs_file = DATA_DIR / "document_query_pairs.csv"
pairs_df = pd.read_csv(pairs_file)

doc_queries = set()
for queries_str in pairs_df["queries_id"]:
    try:
        queries = ast.literal_eval(queries_str)
        doc_queries.update(queries)
    except Exception as e:
        print(f"Lỗi khi đọc queries_id: {e}")

# Lấy phần bù giữa all_queries và doc_queries
test_candidates = list(all_query_ids - doc_queries)

# Lấy ngẫu nhiên ~20% số lượng query của doc_queries
test_size = int(0.2 * len(doc_queries))
test_queries = random.sample(test_candidates, min(test_size, len(test_candidates)))

# Lưu vào file test.csv (gồm cả query)
test_df = pd.DataFrame({
    "query_id": test_queries,
    "query": [id_to_query[qid] for qid in test_queries if qid in id_to_query]
})
test_df.to_csv(test_file, index=False)

print(f"Tổng số query từ processed/queries: {len(all_query_ids)}")
print(f"Tổng số query từ document_query_pairs.csv: {len(doc_queries)}")
print(f"Lưu {len(test_queries)} query vào {test_file}")