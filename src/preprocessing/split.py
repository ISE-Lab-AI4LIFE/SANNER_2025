from pathlib import Path
import json
import random
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"

PROCESSED_DIR = DATA_DIR / "processed"

# Xóa file uprank.csv và test.csv nếu đã tồn tại
for file_path in [PROCESSED_DIR / "uprank.csv", PROCESSED_DIR / "test.csv"]:
    if file_path.exists():
        file_path.unlink()

QUERIES_DIR = PROCESSED_DIR / "queries"
query_files = list(QUERIES_DIR.glob("*/*.csv"))

dfs = []
for q_file in query_files:
    df = pd.read_csv(q_file)
    df['version'] = q_file.stem
    dfs.append(df)

all_queries = pd.concat(dfs, ignore_index=True)

all_queries = all_queries.sample(frac=1, random_state=42).reset_index(drop=True)
split_idx = int(len(all_queries) * 0.8)
uprank_df = all_queries.iloc[:split_idx]
test_df = all_queries.iloc[split_idx:]

uprank_df.to_csv(PROCESSED_DIR / "uprank.csv", index=False)
test_df.to_csv(PROCESSED_DIR / "test.csv", index=False)

print(f"Found {len(all_queries)} query rows across all datasets and versions.")
print(f"Saved {len(uprank_df)} queries to {PROCESSED_DIR / 'uprank.csv'}")
print(f"Saved {len(test_df)} queries to {PROCESSED_DIR / 'test.csv'}")