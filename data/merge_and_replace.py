import pandas as pd
from pathlib import Path

# --- Cấu hình đường dẫn ---
result_path = Path("data/hotflip_result/merged_hotflip_results.csv")
pool_path = Path("data/pool.csv")
output_path = Path("data/hotflip_pool.csv")

# --- Đọc dữ liệu ---
result_df = pd.read_csv(result_path)
pool_df = pd.read_csv(pool_path)

# --- Đảm bảo tên cột đồng nhất ---
result_df = result_df.rename(columns={"document_id": "document_id", "final_poisoned_doc": "final_poisoned_doc"})
pool_df = pool_df.rename(columns={"document_id": "document_id", "document": "document"})

# --- Tạo bản sao để chỉnh sửa ---
merged_df = pool_df.copy()

# --- Thay thế nội dung document khi id trùng ---
merged_df["choosen"] = 0  # mặc định là 0

# Ánh xạ document_id -> final_poisoned_doc từ file result
replacement_map = dict(zip(result_df["document_id"], result_df["final_poisoned_doc"]))

# Xác định các id cần thay
mask = merged_df["document_id"].isin(replacement_map.keys())

# Thay thế document và gắn nhãn choosen = 1
merged_df.loc[mask, "document"] = merged_df.loc[mask, "document_id"].map(replacement_map)
merged_df.loc[mask, "choosen"] = 1

# --- Xuất kết quả ---
output_path.parent.mkdir(parents=True, exist_ok=True)
merged_df.to_csv(output_path, index=False)

print(f"✅ Đã ghi file kết quả: {output_path}")
print(f"🔹 Số dòng được thay thế: {mask.sum()}")