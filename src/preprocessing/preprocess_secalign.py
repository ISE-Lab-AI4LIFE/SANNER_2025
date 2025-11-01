from pathlib import Path
import pandas as pd
import glob
import tqdm
import re

# Tự động phát hiện thư mục gốc của repo (SANNER_2025)
BASE_DIR = Path(__file__).resolve().parents[2]
print(BASE_DIR)
DATA_DIR = BASE_DIR / "data"

# Thư mục dữ liệu nguồn và đầu ra
RAW_DIR = DATA_DIR / "raw" / "secalign-dbg-haiku-javascript-all"
OUT_DIR = DATA_DIR / "processed"

# Định nghĩa lại thư mục đầu ra
queries_dir = OUT_DIR / "queries" / "secalign_dbg_haiku_js"
documents_dir = OUT_DIR / "documents" / "secalign_dbg_haiku_js"

queries_dir.mkdir(parents=True, exist_ok=True)
documents_dir.mkdir(parents=True, exist_ok=True)

# Lấy danh sách tất cả các file parquet
parquet_files = glob.glob(str(RAW_DIR / "*.parquet"))

BATCH_SIZE = 1
DOC_SEP = " [DOC_SEP] "

for file in tqdm.tqdm(parquet_files, desc="Processing raw parquet files", unit="file"):
    # Đọc dữ liệu parquet
    df = pd.read_parquet(file)
    raw_filename = Path(file).stem
    filename = re.sub(r'[^\w\-]', '_', raw_filename)  # sanitize

    # Kiểm tra cột cần thiết
    if not {"original_instruction", "fixed_code"}.issubset(df.columns):
        print(f"⚠️ File {file} không có đủ cột 'original_instruction' và 'fixed_code'. Bỏ qua.")
        continue

    # Chọn 2 cột cần thiết
    df_new = df[["original_instruction", "fixed_code"]]

    # --- Tạo queries_df ---
    queries_data = []
    for idx, row in df_new.iterrows():
        query_id = f"secalignDBGHaikuJS_{filename}_query_{idx+1}"
        queries_data.append({"id": query_id, "queries": row["original_instruction"]})
    queries_df = pd.DataFrame(queries_data)

    # --- Tạo documents_df ---
    documents_data = []
    for batch_start in range(0, len(df_new), BATCH_SIZE):
        batch = df_new.iloc[batch_start: batch_start + BATCH_SIZE]
        merged_docs = DOC_SEP.join(batch["fixed_code"].astype(str).tolist())
        record_id = batch_start // BATCH_SIZE + 1
        doc_id = f"secalignDBGHaikuJS_{filename}_document_{record_id}"
        documents_data.append({"id": doc_id, "documents": merged_docs})
    documents_df = pd.DataFrame(documents_data)

    # --- Lưu file CSV ---
    queries_df.to_csv(queries_dir / f"{filename}.csv", index=False)
    documents_df.to_csv(documents_dir / f"{filename}.csv", index=False)

    print(f"✅ Done saving queries and documents CSV for {filename}")