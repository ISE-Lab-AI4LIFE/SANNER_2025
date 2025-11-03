import pandas as pd
import ast
import random
from pathlib import Path

# --- Cấu hình ---
DATA_DIR = Path("data")
FILE_MAIN = DATA_DIR / "document_query_pairs_lite.csv"
FILE_TARGET = DATA_DIR / "targetted_doc.csv"

TEST_FILE = DATA_DIR / "test.csv"
POOL_FILE = DATA_DIR / "pool.csv"

# --- Đọc dữ liệu ---
df = pd.read_csv(FILE_MAIN)
target_df = pd.read_csv(FILE_TARGET)

# --- Parse các cột ---
# queries: tách theo [SEP]
df['queries'] = df['queries'].astype(str).apply(lambda x: [q.strip() for q in x.split('[SEP]') if q.strip() != ''])
# queries_id: parse list thật
df['queries_id'] = df['queries_id'].apply(ast.literal_eval)

# --- Hợp tất cả queries và queries_id ---
all_queries, all_query_ids = [], []
for q_list, qid_list in zip(df['queries'], df['queries_id']):
    all_queries.extend(q_list)
    all_query_ids.extend(qid_list)

query_set = set(all_queries)
query_id_set = set(all_query_ids)

# --- Lọc bỏ queries thuộc document nằm trong targetted_doc ---
target_doc_ids = set(target_df['document_id'].tolist())

queries_to_remove, query_ids_to_remove = set(), set()
for _, row in df.iterrows():
    if row['document_id'] in target_doc_ids:
        queries_to_remove.update(row['queries'])
        query_ids_to_remove.update(row['queries_id'])

# --- Loại bỏ các query/id đó ---
remaining_queries = list(query_set - queries_to_remove)
remaining_query_ids = list(query_id_set - query_ids_to_remove)

# --- Gộp lại thành list song song ---
pairs = list(zip(remaining_queries, remaining_query_ids))
random.shuffle(pairs)

# --- Lấy 20% đầu tiên cho test.csv ---
# Nếu không có pairs thì cả 2 tập rỗng
if len(pairs) == 0:
    test_pairs = []
    extra_pairs = []
else:
    n_test = int(len(pairs) * 0.2)
    # đảm bảo ít nhất 1 nếu có dữ liệu (tùy ý, bỏ nếu không muốn)
    # n_test = max(1, n_test)  # nếu bạn muốn bắt buộc lấy >=1
    test_pairs = pairs[:n_test]

    # --- Lấy thêm 100 query ngẫu nhiên (nếu còn) ---
    remaining_for_extra = pairs[n_test:]
    n_extra = min(100, len(remaining_for_extra))
    extra_pairs = random.sample(remaining_for_extra, n_extra) if n_extra > 0 else []

# --- Tập query cần loại bỏ trong df gốc ---
selected_queries = {q for q, _ in (test_pairs + extra_pairs)}

# --- Ghi test.csv (20%) ---
TEST_FILE = DATA_DIR / "test.csv"
test_df = pd.DataFrame(test_pairs, columns=['queries', 'queries_id'])
test_df.to_csv(TEST_FILE, index=False)
print(f"✅ Đã lưu {len(test_df)} query vào {TEST_FILE}")

# --- Ghi last_test.csv (100 extra) ---
LAST_TEST_FILE = DATA_DIR / "last_test.csv"
last_test_df = pd.DataFrame(extra_pairs, columns=['queries', 'queries_id'])
last_test_df.to_csv(LAST_TEST_FILE, index=False)
print(f"✅ Đã lưu {len(last_test_df)} query vào {LAST_TEST_FILE}")

# --- Drop các dòng trong df có chứa query đã chọn ---
def contains_selected(qlist):
    return any(q in selected_queries for q in qlist)

filtered_df = df[~df['queries'].apply(contains_selected)].copy()

# --- Tạo pool.csv gồm document_id, document, dataset_name ---
pool_df = filtered_df[['document_id', 'document', 'dataset_name']].drop_duplicates()
pool_df.to_csv(POOL_FILE, index=False)
print(f"✅ Đã lưu {len(pool_df)} dòng vào {POOL_FILE}")