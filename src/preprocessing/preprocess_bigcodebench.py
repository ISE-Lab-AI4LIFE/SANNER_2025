from pathlib import Path
import pandas as pd
import json
import glob
import tqdm
import re

RAW_DIR = "/Users/hieunguyen/SANNER_2025/data/raw/BigCodeBench"
OUT_DIR = "/Users/hieunguyen/SANNER_2025/data/processed/BigCodeBench"

Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
parquet_files = glob.glob(str(Path(RAW_DIR) / "*.parquet"))

BATCH_SIZE = 50
DOC_SEP = " [DOC_SEP] "

for file in tqdm.tqdm(parquet_files, desc="Processing raw parquet files", unit="file"):
    df = pd.read_parquet(file)
    raw_filename = Path(file).stem
    filename = re.sub(r'[^\w\-]', '_', raw_filename)  # sanitize

    df_new = df[["instruct_prompt", "canonical_solution"]]

    # Tạo thư mục con cho mỗi file raw
    sub_dir = Path(OUT_DIR) / filename
    queries_dir = sub_dir / "queries"
    documents_dir = sub_dir / "documents"

    queries_dir.mkdir(parents=True, exist_ok=True)
    documents_dir.mkdir(parents=True, exist_ok=True)

    print(f"📁 Processing {filename}: saving merged queries and documents in batches of {BATCH_SIZE}")

    # Lưu từng query riêng lẻ
    for idx, row in df_new.iterrows():
        query_file = queries_dir / f"{filename}_query_{idx+1}.json"
        with query_file.open("w", encoding="utf-8") as fq:
            json.dump({"id": f"{filename}_query_{idx+1}", "text": row["instruct_prompt"]}, fq, ensure_ascii=False, indent=2)

    # Lưu document đã merge theo batch BATCH_SIZE
    for batch_start in tqdm.tqdm(range(0, len(df_new), BATCH_SIZE), desc=f"Batches for {filename}", unit="batch"):
        batch = df_new.iloc[batch_start: batch_start + BATCH_SIZE]
        merged_docs = DOC_SEP.join(batch["canonical_solution"].astype(str).tolist())
        record_id = batch_start // BATCH_SIZE + 1
        doc_file = documents_dir / f"{filename}_document_{record_id}.json"
        with doc_file.open("w", encoding="utf-8") as fd:
            json.dump({"id": f"{filename}_document_{record_id}", "text": merged_docs}, fd, ensure_ascii=False, indent=2)

    print(f"✅ Done saving merged batches for {filename} in {sub_dir}")