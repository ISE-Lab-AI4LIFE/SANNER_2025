from pathlib import Path
import pandas as pd
import glob
import tqdm
import re

# Tự động phát hiện thư mục gốc của repo
BASE_DIR = Path(__file__).resolve().parents[2]
print(f"Base directory: {BASE_DIR}")
DATA_DIR = BASE_DIR / "data"

# Thư mục dữ liệu nguồn và đầu ra
RAW_DIR = DATA_DIR / "raw" / "rust_instruction_dataset"
OUT_DIR = DATA_DIR / "processed"

# Định nghĩa lại thư mục đầu ra
queries_dir = OUT_DIR / "queries" / "rust_instruction_dataset"
documents_dir = OUT_DIR / "documents" / "rust_instruction_dataset"

queries_dir.mkdir(parents=True, exist_ok=True)
documents_dir.mkdir(parents=True, exist_ok=True)

# Lấy danh sách tất cả các file parquet
parquet_files = glob.glob(str(RAW_DIR / "*.parquet"))

BATCH_SIZE = 1
DOC_SEP = " [DOC_SEP] "

for file in tqdm.tqdm(parquet_files, desc="Processing raw parquet files", unit="file"):
    df = pd.read_parquet(file)
    raw_filename = Path(file).stem
    filename = re.sub(r'[^\w\-]', '_', raw_filename)

    # **Giả định** dataset có các cột: "instruction" và "code" (hoặc tương tự)
    expected_columns = {"instruction", "output"}
    if not expected_columns.issubset(df.columns):
        print(f"⚠️ File {file} missing expected columns {expected_columns} — columns found: {df.columns.tolist()}")
        # Bạn có thể chọn skip hoặc thử các tên khác
        continue

    # Trích xuất
    df_new = df[["instruction", "output"]].rename(columns={"instruction": "queries", "output": "documents"})

    # Tạo queries_df
    queries_data = []
    for idx, row in df_new.iterrows():
        query_id = f"RustInstr_{filename}_query_{idx+1}"
        queries_data.append({"id": query_id, "queries": row["queries"]})
    queries_df = pd.DataFrame(queries_data)

    # Tạo documents_df
    documents_data = []
    for batch_start in range(0, len(df_new), BATCH_SIZE):
        batch = df_new.iloc[batch_start: batch_start + BATCH_SIZE]
        merged_docs = DOC_SEP.join(batch["documents"].astype(str).tolist())
        record_id = batch_start // BATCH_SIZE + 1
        doc_id = f"RustInstr_{filename}_document_{record_id}"
        queries_id = [f"RustInstr_{filename}_query_{i+1}"
                      for i in range(batch_start, min(batch_start + BATCH_SIZE, len(df_new)))]
        documents_data.append({"id": doc_id, "documents": merged_docs, "queries_id": queries_id})
    documents_df = pd.DataFrame(documents_data)

    # Lưu file CSV
    queries_df.to_csv(queries_dir / f"{filename}.csv", index=False)
    documents_df.to_csv(documents_dir / f"{filename}.csv", index=False)

    print(f"✅ Done saving queries and documents CSV for {filename}")