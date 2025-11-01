from pathlib import Path
import pandas as pd
import json
import tqdm

# Tự động phát hiện thư mục gốc của repo (SANNER_2025)
BASE_DIR = Path(__file__).resolve().parents[2]
print(f"Base directory: {BASE_DIR}")
DATA_DIR = BASE_DIR / "data"

# Đường dẫn dữ liệu
RAW_FILE = DATA_DIR / "raw" / "apps" / "train.jsonl"
OUT_DIR = DATA_DIR / "processed"

# Thư mục đầu ra
queries_dir = OUT_DIR / "queries" / "APPS"
documents_dir = OUT_DIR / "documents" / "APPS"

queries_dir.mkdir(parents=True, exist_ok=True)
documents_dir.mkdir(parents=True, exist_ok=True)

DOC_SEP = " [DOC_SEP] "

# --- Đọc dữ liệu JSONL ---
print("📥 Loading JSONL data...")
records = []
with open(RAW_FILE, "r", encoding="utf-8") as f:
    for line in f:
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue

print(f"✅ Loaded {len(records)} records from {RAW_FILE.name}")

data = []
for idx, item in enumerate(tqdm.tqdm(records, desc="Extracting question & solutions"), start=1):
    q = item.get("question")
    sols = item.get("solutions")
    if not q or not sols:
        continue
    if isinstance(sols, list):
        sols_text = DOC_SEP.join([str(s) for s in sols])
    else:
        sols_text = str(sols)
    query_id = f"APPS_train_query_{idx}"
    doc_id = f"APPS_train_document_{idx}"
    data.append({"query_id": query_id, "doc_id": doc_id, "queries": q, "documents": sols_text})

print(f"✅ Extracted {len(data)} valid samples")
df = pd.DataFrame(data)

# --- Tạo queries_df ---
queries_df = df[["query_id", "queries"]].rename(columns={"query_id": "id"})

# --- Tạo documents_df ---
documents_df = df[["doc_id", "documents"]].rename(columns={"doc_id": "id"})

# --- Lưu CSV ---
queries_csv = queries_dir / "train.csv"
documents_csv = documents_dir / "train.csv"

queries_df.to_csv(queries_csv, index=False)
documents_df.to_csv(documents_csv, index=False)

print(f"✅ Done saving:\n - {queries_csv}\n - {documents_csv}")