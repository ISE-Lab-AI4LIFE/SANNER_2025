from pathlib import Path
import pandas as pd
import tqdm
import re
import json

# Tự động phát hiện thư mục gốc của repo (SANNER_2025)
BASE_DIR = Path(__file__).resolve().parents[2]
print(f"Base directory: {BASE_DIR}")
DATA_DIR = BASE_DIR / "data"

# Thư mục dữ liệu nguồn và đầu ra
RAW_FILE = DATA_DIR / "raw" / "buzz_sources_042_javascript" / "train-00000-of-00001.parquet"
OUT_DIR = DATA_DIR / "processed"

# Thư mục đầu ra
queries_dir = OUT_DIR / "queries" / "buzz_sources_042_javascript"
documents_dir = OUT_DIR / "documents" / "buzz_sources_042_javascript"

queries_dir.mkdir(parents=True, exist_ok=True)
documents_dir.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 1
DOC_SEP = " [DOC_SEP] "

# Đọc dữ liệu parquet
print("📥 Loading parquet data...")
df = pd.read_parquet(RAW_FILE)
print(f"✅ Loaded {len(df)} records")

# Kiểm tra cột cần thiết
if "conversations" not in df.columns:
    raise ValueError("Không tìm thấy cột 'conversations' trong file parquet!")

# --- Tách dữ liệu từ trường conversations ---
data = []
for i, row in tqdm.tqdm(df.iterrows(), total=len(df), desc="Extracting conversations"):
    try:
        conv = row["conversations"]
        # Nếu dạng string, parse lại
        if isinstance(conv, str):
            conv = json.loads(conv)
        # Lấy giá trị human và gpt
        human_text = next((item["value"] for item in conv if item["from"] == "human"), None)
        gpt_text = next((item["value"] for item in conv if item["from"] == "gpt"), None)
        if human_text and gpt_text:
            data.append({"queries": human_text, "documents": gpt_text})
    except Exception as e:
        continue

print(f"✅ Extracted {len(data)} valid pairs from conversations")

df_new = pd.DataFrame(data)

# --- Tạo queries_df ---
queries_data = []
for idx, row in tqdm.tqdm(df_new.iterrows(), total=len(df_new), desc="Building queries"):
    query_id = f"BuzzJS_query_{idx+1}"
    queries_data.append({"id": query_id, "queries": row["queries"]})
queries_df = pd.DataFrame(queries_data)

# --- Tạo documents_df ---
documents_data = []
for batch_start in tqdm.tqdm(range(0, len(df_new), BATCH_SIZE), desc="Building documents"):
    batch = df_new.iloc[batch_start: batch_start + BATCH_SIZE]
    merged_docs = DOC_SEP.join(batch["documents"].astype(str).tolist())
    record_id = batch_start // BATCH_SIZE + 1
    doc_id = f"BuzzJS_document_{record_id}"
    documents_data.append({"id": doc_id, "documents": merged_docs})
documents_df = pd.DataFrame(documents_data)

# --- Lưu file CSV ---
queries_csv = queries_dir / "buzz_sources_042_javascript.csv"
documents_csv = documents_dir / "buzz_sources_042_javascript.csv"

queries_df.to_csv(queries_csv, index=False)
documents_df.to_csv(documents_csv, index=False)

print(f"✅ Done saving:\n - {queries_csv}\n - {documents_csv}")