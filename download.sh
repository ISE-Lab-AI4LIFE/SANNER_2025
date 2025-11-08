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
     'https://storage.googleapis.com/kagglesdsdata/datasets/8633494/13600058/document_query_pairs_lite.csv?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20251108%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20251108T134306Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=9a2be5c4e7c16cb2888ff66df14fc0aae15236e9b3397f4c41cd6ce90d0fc7786568de79967eeceec07c8a96fc1dbfe431e81d7f7e961e98e34f955beaf60e0c41227f2bf84eb4014d71202f30ca2e7046fa2d5eed5de638fbe4cabf23dcc880bf920bbfce4680607a3e61e45da762f2b251ac888f211f54ed76110cce319339bf60857342b49f0e722b80e7924abf8c6935d1fc1d5231a876e35b0041a35ee8d71e9725cb9a7a10eb15db49779ff21bbdbe6a3fc2cb44cea7fb56133196d7270b1e984b33411511534260c7937ef1f4d7d3b1f8e62a25a976af843d6646c174b3c82bfb557d761b2ef963ec068c7e3371dd78d8b34fc11d06031a2dda015ffb' \
     -c -O "$OUTPUT_FILE"