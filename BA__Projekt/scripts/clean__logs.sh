#!/bin/bash

# Define log directory
LOG_DIR="/root/BA__U-i-mlb-Sm-f-d-s-V-a-S/BA__Projekt/assets/logs"

# Safety check
if [ ! -d "$LOG_DIR" ]; then
    echo "[ERROR] Log directory not found: $LOG_DIR"
    exit 1
fi

# Find all matching log files, sort them, and delete all but the last 3
LOG_FILES=($(ls "$LOG_DIR"/persist__*.log 2>/dev/null | sort))

NUM_FILES=${#LOG_FILES[@]}

if [ "$NUM_FILES" -le 3 ]; then
    echo "[INFO] Only $NUM_FILES log file(s) found. Nothing to delete."
    exit 0
fi

FILES_TO_DELETE=("${LOG_FILES[@]:0:$((NUM_FILES - 3))}")

for file in "${FILES_TO_DELETE[@]}"; do
    rm "$file" && echo "[✔] Deleted: $file" || echo "[✘] Failed to delete: $file"
done

echo "[INFO] Kept last 3 log files."