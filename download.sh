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
     'https://storage.googleapis.com/kagglesdsdata/datasets/8633494/13597150/document_query_pairs_lite.csv?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20251103%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20251103T103908Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=095cb9d43b97b977ddd38e2db6ae3c227b081ab03b6ab58a0a1081c045b6c35c82b5934237c95e185edaceb3d3e828e03af328a052fd81449d13f97ba5793e9ede74a7db4331a989e1a54703eb18cb9181486a1c6d3bcc9a37ab203f10e66505ba210dd26f4decdea8f5de110f8d405bfb6bd32e7858719df7c76a1c634834da9bdbd2d14d9b0084588df63728b2d5cdf576937504d91aecd8ea944c32385759a03300b168c4ea263ce9d2b754c8c85033a7f3c18fd4670c191095d2a7f5473774c1265711e4d4d63348164622eb3f54a72de2e15eb7d15190caa1e4b94b37beaa53a4506a2d7e2718bee8b78f6caf9774c39d54df0c3d2e5814a33c802229cb' \
     -c -O "$OUTPUT_FILE"