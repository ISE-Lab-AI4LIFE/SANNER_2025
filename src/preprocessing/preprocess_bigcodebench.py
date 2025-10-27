from pathlib import Path
import pandas as pd
import json
import glob
import tqdm
import re

# Tự động phát hiện thư mục gốc của repo (SANNER_2025)
BASE_DIR = Path(__file__).resolve().parents[2]
print(BASE_DIR)
DATA_DIR = BASE_DIR / "data"

# Thư mục dữ liệu nguồn và đầu ra
RAW_DIR = DATA_DIR / "raw" / "BigCodeBench"
OUT_DIR = DATA_DIR / "processed" 

# Định nghĩa lại thư mục đầu ra
queries_dir = OUT_DIR / "queries" / "BigCodeBench"
documents_dir = OUT_DIR / "documents" / "BigCodeBench"

queries_dir.mkdir(parents=True, exist_ok=True)
documents_dir.mkdir(parents=True, exist_ok=True)

parquet_files = glob.glob(str(Path(RAW_DIR) / "*.parquet"))

BATCH_SIZE = 1
DOC_SEP = " [DOC_SEP] "

for file in tqdm.tqdm(parquet_files, desc="Processing raw parquet files", unit="file"):
    df = pd.read_parquet(file)
    raw_filename = Path(file).stem
    filename = re.sub(r'[^\w\-]', '_', raw_filename)  # sanitize

    df_new = df[["instruct_prompt", "canonical_solution"]]

    # Tạo DataFrame queries_df với cột 'id' và 'queries'
    queries_data = []
    for idx, row in df_new.iterrows():
        query_id = f"BigCodeBench_{filename}_query_{idx+1}"
        queries_data.append({"id": query_id, "queries": row["instruct_prompt"]})
    queries_df = pd.DataFrame(queries_data)

    # Tạo DataFrame documents_df với cột 'id', 'documents', 'queries_id'
    documents_data = []
    for batch_start in range(0, len(df_new), BATCH_SIZE):
        batch = df_new.iloc[batch_start: batch_start + BATCH_SIZE]
        merged_docs = DOC_SEP.join(batch["canonical_solution"].astype(str).tolist())
        record_id = batch_start // BATCH_SIZE + 1
        doc_id = f"BigCodeBench_{filename}_document_{record_id}"
        queries_id = [f"BigCodeBench_{filename}_query_{i+1}" for i in range(batch_start, min(batch_start + BATCH_SIZE, len(df_new)))]
        documents_data.append({"id": doc_id, "documents": merged_docs, "queries_id": queries_id})
    documents_df = pd.DataFrame(documents_data)

    # Lưu queries_df và documents_df vào file CSV
    queries_df.to_csv(queries_dir / f"{filename}.csv", index=False)
    documents_df.to_csv(documents_dir / f"{filename}.csv", index=False)

    print(f"✅ Done saving queries and documents CSV for {filename}")