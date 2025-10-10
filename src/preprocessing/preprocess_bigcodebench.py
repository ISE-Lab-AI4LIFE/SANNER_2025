import pandas as pd
import json
import glob
import os
import tqdm

RAW_DIR = "/Users/hieunguyen/SANNER_2025/data/raw/BigCodeBench"
OUT_DIR = "/Users/hieunguyen/SANNER_2025/data/processed/BigCodeBench"

parquet_files = glob.glob(os.path.join(RAW_DIR,"*.parquet"))
# Keep instruct_prompt, canonical_solution columns
# Merge and convert to JSON
for file in tqdm.tqdm(parquet_files, desc="Processing raw parquet files", unit="file"):
    df = pd.read_parquet(file)
    filename = os.path.splitext(os.path.basename(file))[0]

    df_new = df[["instruct_prompt", "canonical_solution"]]
    records = []

    batch_size = 50
    for i in range(0, len(df_new), batch_size):
        batch = df_new.iloc[i:i+batch_size]
        merged_text = batch["instruct_prompt"].astype(str).tolist()
        answers = batch["canonical_solution"].astype(str).tolist()
        merged_docs = " [DOC_SEP] ".join(answers)

        records.append({
            "id": len(records) + 1,
            "query": merged_text,
            "document": merged_docs
        })

    out_path = os.path.join(OUT_DIR, f"{filename}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved: {out_path}")