import pandas as pd
from pathlib import Path

# --- Cấu hình đường dẫn ---
result_path = Path("data/linklure_result/merged_linklure_results.csv")
pool_path = Path("data/pool.csv")
output_path = Path("data/linklure_pool.csv")

# --- Đọc dữ liệu ---
result_df = pd.read_csv(result_path)
pool_df = pd.read_csv(pool_path)

# --- Kiểm tra cột bắt buộc ---
required_result_cols = {"document_id"}
required_pool_cols = {"document_id", "document"}

if not required_result_cols.issubset(result_df.columns):
    raise KeyError(f"File result thiếu các cột: {required_result_cols - set(result_df.columns)}")
if not required_pool_cols.issubset(pool_df.columns):
    raise KeyError(f"File pool thiếu các cột: {required_pool_cols - set(pool_df.columns)}")

# --- Xác định cột văn bản trong result_df ---
text_col = None
if "document" in result_df.columns:
    text_col = "document"
elif "final_poisoned_doc" in result_df.columns:
    text_col = "final_poisoned_doc"
else:
    raise KeyError("File result thiếu cả hai cột 'document' và 'final_poisoned_doc'.")

# --- Chuẩn hóa kiểu dữ liệu document_id ---
result_df["document_id"] = result_df["document_id"].astype(str)
pool_df["document_id"] = pool_df["document_id"].astype(str)

# --- Ánh xạ document_id -> văn bản ---
replacement_map = dict(zip(result_df["document_id"], result_df[text_col]))

# --- Thay thế nội dung document nếu có bản poisoned và thêm cột choosen ---
def replace_and_flag(row):
    doc_id = row["document_id"]
    if doc_id in replacement_map:
        return pd.Series([replacement_map[doc_id], 1])
    else:
        return pd.Series([row["document"], 0])

pool_df[["document", "choosen"]] = pool_df.apply(replace_and_flag, axis=1)

# --- Ghi kết quả với 3 cột cần thiết ---
output_path.parent.mkdir(parents=True, exist_ok=True)
pool_df[["document_id", "document", "choosen"]].to_csv(output_path, index=False)

# --- Thông báo ---
num_replaced = pool_df["choosen"].sum()
print(f"✅ Đã ghi file kết quả: {output_path}")
print(f"🔹 Số dòng bị thay thế (poisoned): {num_replaced}")
print(f"🔹 Tổng số dòng: {len(pool_df)}")