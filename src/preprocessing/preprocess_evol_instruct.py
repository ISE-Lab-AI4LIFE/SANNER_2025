from pathlib import Path
import pandas as pd
import json
import tqdm
import re

# Xác định thư mục gốc của repo (SANNER_2025)
BASE_DIR = Path(__file__).resolve().parents[2]
print(f"Base directory: {BASE_DIR}")
DATA_DIR = BASE_DIR / "data"

# Thư mục dữ liệu nguồn và đầu ra
RAW_FILE = DATA_DIR / "raw" / "Evol-Instruct-Code-80k-v1" / "EvolInstruct-Code-80k.json"
OUT_DIR = DATA_DIR / "processed"

# Định nghĩa thư mục đầu ra
queries_dir = OUT_DIR / "queries" / "EvolInstruct_Code_80k"
documents_dir = OUT_DIR / "documents" / "EvolInstruct_Code_80k"

queries_dir.mkdir(parents=True, exist_ok=True)
documents_dir.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 1
DOC_SEP = " [DOC_SEP] "

# Đọc file JSON
print("📥 Loading JSON data...")
with open(RAW_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"✅ Loaded {len(data)} records from {RAW_FILE.name}")

# Chuyển sang DataFrame
df = pd.DataFrame(data)
if not {"instruction", "output"}.issubset(df.columns):
    raise ValueError("File JSON không có đủ 2 cột: 'instruction' và 'output'")

df_new = df[["instruction", "output"]]

# --- Tạo queries_df và documents_df ---
queries_data = []
documents_data = []
for idx, row in tqdm.tqdm(df_new.iterrows(), total=len(df_new), desc="Processing pairs"):
    instruction = str(row["instruction"]).strip()
    output = str(row["output"]).strip()
    if not instruction or not output:
        continue  # Bỏ qua nếu một trong hai trường rỗng

    query_id = f"EvolInstructCode80k_query_{idx+1}"
    doc_id = f"EvolInstructCode80k_document_{idx+1}"
    queries_data.append({"id": query_id, "queries": instruction})
    documents_data.append({"id": doc_id, "documents": output})

queries_df = pd.DataFrame(queries_data)
documents_df = pd.DataFrame(documents_data)

# --- Lưu file CSV ---
queries_csv = queries_dir / "EvolInstruct-Code-80k.csv"
documents_csv = documents_dir / "EvolInstruct-Code-80k.csv"

queries_df.to_csv(queries_csv, index=False)
documents_df.to_csv(documents_csv, index=False)

print(f"✅ Done saving:\n - {queries_csv}\n - {documents_csv}")