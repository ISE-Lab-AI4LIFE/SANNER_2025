import json
import pandas as pd
from pathlib import Path

# --- Cấu hình đường dẫn ---
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"

DOC_TO_QUERIES_JSON = DATA_DIR / "document_to_queries.json"
DOCUMENTS_DIR = DATA_DIR / "processed" / "documents"
QUERIES_DIR = DATA_DIR / "processed" / "queries"
OUTPUT_CSV = DATA_DIR / "document_query_pairs.csv"

# --- 1. Load ánh xạ document_id -> query_ids ---
with open(DOC_TO_QUERIES_JSON, "r", encoding="utf-8") as f:
    doc_to_queries = json.load(f)

# --- 2. Load toàn bộ queries text (từ nhiều CSV) ---
query_texts = {}
for csv_file in QUERIES_DIR.rglob("*.csv"):
    df = pd.read_csv(csv_file)
    for _, row in df.iterrows():
        query_texts[str(row["id"])] = row["queries"]

# --- 3. Load toàn bộ documents text (từ nhiều CSV) ---
document_texts = {}
for csv_file in DOCUMENTS_DIR.rglob("*.csv"):
    df = pd.read_csv(csv_file)
    for _, row in df.iterrows():
        document_texts[str(row["id"])] = row["documents"]

# --- 4. Tạo DataFrame kết quả ---
records = []
for doc_id, query_ids in doc_to_queries.items():
    doc_text = document_texts.get(doc_id, None)
    queries_text = [query_texts.get(qid, None) for qid in query_ids if qid in query_texts]
    records.append({
        "document": doc_text,
        "document_id": doc_id,
        "queries": queries_text,
        "queries_id": query_ids
    })

df_out = pd.DataFrame(records)
df_out.to_csv(OUTPUT_CSV, index=False)

print(f"✅ Saved {len(df_out)} records to {OUTPUT_CSV}")