from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# --- Cấu hình đường dẫn ---
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
INPUT_FILE = DATA_DIR / "document_query_pairs.csv"

# --- Đọc file CSV ---
df = pd.read_csv(INPUT_FILE)

# --- Kiểm tra cột dataset_name (nếu chưa có thì tạo mới) ---
if "dataset_name" not in df.columns:
    df["dataset_name"] = df["document_id"].astype(str).str.split("_").str[0]

# --- Lấy một nửa dữ liệu, giữ nguyên phân phối dataset_name ---
_, lite_df = train_test_split(
    df,
    test_size=0.1,                     # Giữ 50%
    stratify=df["dataset_name"],       # Bảo toàn tỉ lệ phân phối
    random_state=42
)

# --- Lưu file mới ---
lite_path = INPUT_FILE.with_name(f"{INPUT_FILE.stem}_lite.csv")
lite_df.to_csv(lite_path, index=False)

# --- In thống kê kiểm tra ---
def show_distribution(sub_df, name):
    stats = sub_df["dataset_name"].value_counts(normalize=True) * 100
    print(f"\n📊 {name}: {len(sub_df)} dòng")
    for ds, pct in stats.items():
        print(f"- {ds}: {pct:.2f}%")

print(f"✅ Đã tạo file lite tại: {lite_path}")
show_distribution(df, "Bản đầy đủ")
show_distribution(lite_df, "Bản lite (50%)")