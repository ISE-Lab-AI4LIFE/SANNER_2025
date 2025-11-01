from pathlib import Path
import pandas as pd
import json
import tqdm
import re

# Tự động phát hiện thư mục gốc của repo (SANNER_2025)
BASE_DIR = Path(__file__).resolve().parents[2]
print(f"Base directory: {BASE_DIR}")
DATA_DIR = BASE_DIR / "data"

# Thư mục dữ liệu nguồn và đầu ra
RAW_FILE = DATA_DIR / "raw" / "DS-1000" / "test.jsonl"
OUT_DIR = DATA_DIR / "processed"

# Định nghĩa thư mục đầu ra
queries_dir = OUT_DIR / "queries" / "DS_1000"
documents_dir = OUT_DIR / "documents" / "DS_1000"

queries_dir.mkdir(parents=True, exist_ok=True)
documents_dir.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 1
DOC_SEP = " [DOC_SEP] "

# Đọc file JSONL
print("📥 Loading JSONL data...")
records = []
with open(RAW_FILE, "r", encoding="utf-8") as f:
    for line in f:
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue

print(f"✅ Loaded {len(records)} records from {RAW_FILE.name}")

# Chuyển sang DataFrame
df = pd.DataFrame(records)
if not {"prompt", "code_context"}.issubset(df.columns):
    raise ValueError("File không có đủ 2 cột: 'prompt' và 'code_context'")

df_new = df[["prompt", "code_context"]]

# --- Tạo queries_df ---
queries_data = []
for idx, row in tqdm.tqdm(df_new.iterrows(), total=len(df_new), desc="Processing queries"):
    query_id = f"DS1000_query_{idx+1}"
    queries_data.append({"id": query_id, "queries": row["prompt"]})
queries_df = pd.DataFrame(queries_data)

# --- Tạo documents_df ---
documents_data = []
for batch_start in tqdm.tqdm(range(0, len(df_new), BATCH_SIZE), desc="Processing documents"):
    batch = df_new.iloc[batch_start: batch_start + BATCH_SIZE]
    merged_docs = DOC_SEP.join(batch["code_context"].astype(str).tolist())
    record_id = batch_start // BATCH_SIZE + 1
    doc_id = f"DS1000_document_{record_id}"
    documents_data.append({"id": doc_id, "documents": merged_docs})
documents_df = pd.DataFrame(documents_data)

# --- Lưu file CSV ---
queries_csv = queries_dir / "DS1000_test.csv"
documents_csv = documents_dir / "DS1000_test.csv"

queries_df.to_csv(queries_csv, index=False)
documents_df.to_csv(documents_csv, index=False)

print(f"✅ Done saving:\n - {queries_csv}\n - {documents_csv}")