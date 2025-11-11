#!/usr/bin/env bash
set -euo pipefail

# run_uprank_nohup.sh
# Launches the uprank python script for data/chunks/targetted_doc_7.csv using nohup
# Usage: bash run_uprank_nohup.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"

TARGET="$REPO_ROOT/data/chunks/targetted_doc_16.csv"
OUT_DIR="$REPO_ROOT/data/linklure_result"
LOG_DIR="$REPO_ROOT/logs"
LOG_FILE="$LOG_DIR/linklure_targetted_doc_16.log"

mkdir -p "$OUT_DIR"
mkdir -p "$LOG_DIR"

if [ ! -f "$TARGET" ]; then
  echo "ERROR: target file not found: $TARGET" >&2
  exit 2
fi

echo "Starting uprank for: $TARGET"
echo "Logs: $LOG_FILE"

# Run with nohup, redirect stdout+stderr to log, and run in background
nohup python3 "$REPO_ROOT/src/linklure_uprank/linklure_uprank.py" --target_path "$TARGET" > "$LOG_FILE" 2>&1 &
PID=$!
echo "Launched (PID=$PID). Check '$LOG_FILE' or 'tail -f $LOG_FILE' for progress."

exit 0
