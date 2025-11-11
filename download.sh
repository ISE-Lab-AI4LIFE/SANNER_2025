#!/bin/bash

# Thư mục lưu dữ liệu
DATA_DIR="$(dirname "$0")/data"
mkdir -p "$DATA_DIR"

# Đường dẫn file đích
OUTPUT_FILE="$DATA_DIR/document_query_pairs_lite.csv"

# URL file cần tải
FILE_URL="https://storage.googleapis.com/kagglesdsdata/datasets/8633494/13600058/document_query_pairs_lite.csv?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20251111%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20251111T153713Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=0843b95d771e86ebdd8a9e1295a0577248e964438483a646f9f0b59eec51799f9a803047d9e2da6eca53ffae3d6de343567b10c94c295980f73769f340252d0560b9fe6384f83b5356b532c0ae12e811119130d5b62e8fca50d61f83de31abb50d26b03febf824633aa3befd4a4a7242607e1f10f9a772cf153d318a9c05cf5121bd2e6429ee71601807da3dd81e5666b923091e6e8927a9f92813490f364f4e026b31fddfdd701fd618fa202953149629fb5d362dbdb50b3d3bb6afa910c0e1078f9fb030e77bf1b85bd698dcdb504d60a724b2ce3d3aa82deb789d21a422eaf04d2c60ce9b7cd12d6a53d59ede2a5c8e597771cf7b9e4167e936b3027891cd"

# Header mặc định cho wget
HEADERS=(
    --header='Host: storage.googleapis.com'
    --header='User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36'
    --header='Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7'
    --header='Accept-Language: vi-VN,vi;q=0.9,fr-FR;q=0.8,fr;q=0.7,en-US;q=0.6,en;q=0.5'
    --header='Referer: https://www.kaggle.com/'
)

# Tải file (tiếp tục nếu bị gián đoạn)
wget "${HEADERS[@]}" -c -O "$OUTPUT_FILE" "$FILE_URL"