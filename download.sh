#!/bin/bash

# Thư mục lưu dữ liệu
DATA_DIR="$(dirname "$0")/data"
mkdir -p "$DATA_DIR"

# Đường dẫn file đích
OUTPUT_FILE="$DATA_DIR/document_query_pairs_lite.csv"

# Lệnh tải file
wget --header='Host: storage.googleapis.com' \
     --header='User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36' \
     --header='Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7' \
     --header='Accept-Language: vi-VN,vi;q=0.9,fr-FR;q=0.8,fr;q=0.7,en-US;q=0.6,en;q=0.5' \
     --header='Referer: https://www.kaggle.com/' \
     'https://storage.googleapis.com/kagglesdsdata/datasets/8633494/13588559/document_query_pairs_lite.csv?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20251102%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20251102T154505Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=130e5b1a70d5280ceb9ddf8bc65bc551b658d5967ac7bebe07468312b059efdd8a4d5ac16e46faa4cec351c08447b1a40a7307d85cde346a8258fdd1fec890b72eb16885daa3e1d22c13d0ab481306bfa5dbecde0f79b73ec841bbf677369a7ae86a1fd9a256ee666c32fbab4054d8ae78106ede668d017c1b35897591b0c04b4557abd191e44687122e5fd1e247b9b815265b2002455624255e064e00a464d689e28ec8f30bfd2a2b688820d32ac31671634751561ddca2485406c10393dd983821cb3fb413e96fb4ac7cac3cbf14ca96719b488cefe829fc87c421dbbba672015c16f9094ea088cd00f20b943d5c35643f26537dea06bf4d507129dca18f72' \
     -c -O "$OUTPUT_FILE"