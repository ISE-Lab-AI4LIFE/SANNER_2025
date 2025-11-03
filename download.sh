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
     'https://storage.googleapis.com/kagglesdsdata/datasets/8633494/13600058/document_query_pairs_lite.csv?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20251103%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20251103T151741Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=261f5f24db42e6384cf4294c7889151c2ba54c9b0489996317383b87ef2896ba22b2c1848bf4bd433d581949f2366dec84da52324f59e4539a7fa3846a5127f5fae1ea7adbca3ebfc1dba2f1633532d048adcbbdc2343ee4d204c9917feffa7a427edf4702a01f8b45d44de5d272f2e609270c68753eb41f7a8450ee68c2ba0b201aeff08ceb8a9476200ff07bbd9822615eb93efad236f602d4a03b314a8f54f83e8d4ebb2e6802ee388fa095cbb6b990f334c938ad4171c7f097334a6a2e739956c5e58e355d85ef40befdff6a2a9f39b1fa0aded294e900f9c23545605eb4afa4dfd41f2babec3827cd0f6d281407e42fd67140ca0eb8ee9a6c3304f175b6' \
     -c -O "$OUTPUT_FILE"