#!/usr/bin/env bash
set -euo pipefail

# run_uprank_nohup.sh
# Launches the uprank python script for docs 29, 31, 33 sequentially (not in parallel)
# Usage: bash run_uprank_nohup.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"

OUT_DIR="$REPO_ROOT/data/linklure_result"
LOG_DIR="$REPO_ROOT/logs"

mkdir -p "$OUT_DIR"
mkdir -p "$LOG_DIR"

# Danh sách doc cần chạy
DOCS=(29 31 33 35 37)

for DOC in "${DOCS[@]}"; do
  TARGET="$REPO_ROOT/data/chunks/targetted_doc_${DOC}.csv"
  LOG_FILE="$LOG_DIR/linklure_targetted_doc_${DOC}.log"

  if [ ! -f "$TARGET" ]; then
    echo "ERROR: target file not found: $TARGET" >&2
    exit 2
  fi

  echo "=============================================="
  echo "Starting uprank for: $TARGET"
  echo "Logs: $LOG_FILE"
  echo "=============================================="

  # Chạy nohup, ghi log, và đợi hoàn thành trước khi chuyển sang doc tiếp theo
  nohup python3 "$REPO_ROOT/src/linklure_uprank/linklure_uprank.py" \
    --target_path "$TARGET" > "$LOG_FILE" 2>&1 &

  PID=$!
  echo "Launched (PID=$PID). Waiting for it to finish..."
  wait $PID
  echo "Completed targetted_doc_${DOC}. Moving to next..."
  echo
done

echo "All uprank jobs completed successfully."
exit 0