from pathlib import Path
import pandas as pd

# --- Cấu hình đường dẫn ---
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"

# --- Đường dẫn file ---
SRC = DATA_DIR / "document_query_pairs_lite.csv"
DST = DATA_DIR / "targetted_doc.csv"

# --- Đọc file CSV ---
df = pd.read_csv(SRC)

# --- Lấy ngẫu nhiên 5% số dòng ---
sample_df = df.sample(frac=0.05, random_state=42)

# --- Giữ lại chỉ cột document_id ---
sample_df = sample_df[["document_id"]]

# --- Ghi ra file mới ---
sample_df.to_csv(DST, index=False)

print(f"✅ Đã lưu {len(sample_df)} dòng vào {DST}")