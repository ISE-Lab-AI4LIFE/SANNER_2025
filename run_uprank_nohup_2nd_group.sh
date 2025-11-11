#!/usr/bin/env bash
set -euo pipefail

# run_uprank_multi_2.sh
# Chạy liên tiếp các chunk được chỉ định bằng Python script linklure_uprank.py
# Usage: bash run_uprank_multi_2.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"

CHUNKS=(12 18 23 27 31 35 39)
OUT_DIR="$REPO_ROOT/data/linklure_result"
LOG_DIR="$REPO_ROOT/logs"

mkdir -p "$OUT_DIR"
mkdir -p "$LOG_DIR"

for i in "${CHUNKS[@]}"; do
  TARGET="$REPO_ROOT/data/chunks/targetted_doc_${i}.csv"
  LOG_FILE="$LOG_DIR/linklure_targetted_doc_${i}.log"

  if [ ! -f "$TARGET" ]; then
    echo "❌ ERROR: target file not found: $TARGET" >&2
    continue
  fi

  echo "🚀 Starting uprank for chunk $i..."
  echo "📄 Target: $TARGET"
  echo "📝 Log: $LOG_FILE"

  # Chạy tuần tự, ghi log ra file
  python3 "$REPO_ROOT/src/linklure_uprank/linklure_uprank.py" --target_path "$TARGET" > "$LOG_FILE" 2>&1

  echo "✅ Finished chunk $i. Log saved to $LOG_FILE"
  echo "---------------------------------------------"
done

echo "🎯 All specified chunks processed successfully."
exit 0