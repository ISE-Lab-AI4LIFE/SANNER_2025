import json
import csv
import random
from pathlib import Path

# Đường dẫn file JSON và CSV
json_file = Path("suggestions_dict.json")
input_csv = Path("data/hotflip_result/merged_hotflip_results_clean.csv")
output_csv = Path("data/hotflip_result/merged_hotflip_results_poisoned.csv")

# Đọc JSON
with json_file.open("r", encoding="utf-8") as f:
    suggestions = json.load(f)

# Chỉ dùng 3 model có trong JSON
allowed_models = list(suggestions.keys())

# Đọc CSV gốc
with input_csv.open("r", newline='', encoding="utf-8") as f_in:
    reader = csv.DictReader(f_in)
    data_rows = list(reader)

# Chèn comment vào cột 'final_poisoned_doc' của từng dòng dữ liệu
for row in data_rows:
    # Chọn ngẫu nhiên 2 model
    selected_models = random.sample(allowed_models, 2)

    # Chọn ngôn ngữ từ model tương ứng
    model_0_languages = list(suggestions[selected_models[0]].keys())
    model_1_languages = list(suggestions[selected_models[1]].keys())

    selected_language_0 = random.choice(model_0_languages)
    selected_language_1 = random.choice(model_1_languages)

    # Lấy comment đầu và cuối
    comment_start = suggestions[selected_models[0]][selected_language_0]
    comment_end = suggestions[selected_models[1]][selected_language_1]

    # Chèn comment vào giá trị cột 'final_poisoned_doc'
    row['final_poisoned_doc'] = f"{comment_start} {row['final_poisoned_doc']} {comment_end}"

# Ghi CSV mới, đổi tên cột 'final_poisoned_doc' thành 'document'
with output_csv.open("w", newline='', encoding="utf-8") as f_out:
    fieldnames = ['document_id', 'document']
    writer = csv.DictWriter(f_out, fieldnames=fieldnames)
    writer.writeheader()
    for row in data_rows:
        writer.writerow({
            'document_id': row['document_id'],
            'document': row['final_poisoned_doc']
        })

print(f"Đã chèn comment vào cột 'final_poisoned_doc' và lưu file mới với cột 'document' tại {output_csv}")