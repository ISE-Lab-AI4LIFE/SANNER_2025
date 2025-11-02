from pathlib import Path
import pandas as pd

# 🔹 Thư mục chứa các file CSV
data_dir = Path("data/hotflip_result")

# 🔹 Lấy tất cả file CSV trong thư mục
csv_files = list(data_dir.glob("*.csv"))

# 🔹 Kiểm tra nếu không có file
if not csv_files:
    raise FileNotFoundError(f"❌ Không tìm thấy file CSV nào trong {data_dir}")

# 🔹 Đọc và gộp tất cả file
dfs = []
for file in csv_files:
    df = pd.read_csv(file)
    dfs.append(df)
    print(f"✅ Đã đọc: {file.name} ({len(df)} dòng)")

# 🔹 Merge tất cả
merged_df = pd.concat(dfs, ignore_index=True)

# 🔹 Lưu lại file hợp nhất
output_path = data_dir / "merged_hotflip_results.csv"
merged_df.to_csv(output_path, index=False)

print(f"\n🎉 Đã merge {len(csv_files)} file CSV thành công!")
print(f"📄 File xuất: {output_path}")