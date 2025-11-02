from pathlib import Path
import pandas as pd

# --- Cấu hình đường dẫn ---
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"

# --- Đường dẫn file đầu vào và thư mục đầu ra ---
INPUT_FILE = DATA_DIR / "targetted_doc.csv"
OUTPUT_DIR = DATA_DIR / "chunks"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Tham số chia ---
LINES_PER_FILE = 50

# --- Đọc dữ liệu ---
df = pd.read_csv(INPUT_FILE)

# --- Chia thành nhiều file nhỏ ---
for i in range(0, len(df), LINES_PER_FILE):
    chunk = df.iloc[i:i + LINES_PER_FILE]
    output_file = OUTPUT_DIR / f"targetted_doc_{i // LINES_PER_FILE}.csv"
    chunk.to_csv(output_file, index=False)

print(f"✅ Đã tách {len(df)} dòng thành {(len(df) + LINES_PER_FILE - 1) // LINES_PER_FILE} file nhỏ trong: {OUTPUT_DIR}")