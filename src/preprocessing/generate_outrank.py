import json
import pandas as pd
from pathlib import Path

# --- Cấu hình đường dẫn ---
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"

OUTRANK_JSON = DATA_DIR / "outrank.json"
QUERIES_DIR = DATA_DIR / "processed" / "queries"
DOCUMENTS_DIR = DATA_DIR / "processed" / "documents"
OUTPUT_CSV = DATA_DIR / "outrank.csv"

# --- 1. Load outrank.json ---
with open(OUTRANK_JSON, "r", encoding="utf-8") as f:
    outrank_data = json.load(f)

# --- 2. Load toàn bộ queries text ---
query_texts = {}
for csv_file in QUERIES_DIR.rglob("*.csv"):
    df = pd.read_csv(csv_file)
    for _, row in df.iterrows():
        query_texts[str(row["id"])] = row["queries"]

# --- 3. Load toàn bộ documents text ---
document_texts = {}
for csv_file in DOCUMENTS_DIR.rglob("*.csv"):
    df = pd.read_csv(csv_file)
    for _, row in df.iterrows():
        document_texts[str(row["id"])] = row["documents"]

# --- 4. Tạo DataFrame kết quả ---
records = []
for entry in outrank_data:
    doc_id = entry["doc_id"]
    doc_text = document_texts.get(doc_id, None)
    query_ids = [q["query_id"] for q in entry["outranked_queries"]]
    queries_text = " [SEP] ".join([query_texts.get(qid, "") for qid in query_ids if qid in query_texts])
    records.append({
        "document": doc_text,
        "document_id": doc_id,
        "queries": queries_text,
        "queries_id": query_ids
    })

# --- 5. Lưu ra file CSV ---
df_out = pd.DataFrame(records)
df_out.to_csv(OUTPUT_CSV, index=False)

print(f"✅ Saved {len(df_out)} outranked records to {OUTPUT_CSV}")