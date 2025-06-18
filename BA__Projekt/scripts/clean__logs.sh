#!/bin/bash
# BA__Projekt/scripts/clean__logs.sh

# Define log directory
LOG_DIR="/root/BA__U-i-mlb-Sm-f-d-s-V-a-S/BA__Projekt/assets/logs"

# Check for quiet flag
QUIET=false
for arg in "$@"; do
    if [[ "$arg" == "--quiet" || "$arg" == "--q" ]]; then
        QUIET=true
        break
    fi
done

# Safety check
if [ ! -d "$LOG_DIR" ]; then
    if [ "$QUIET" = false ]; then
        echo "[ERROR] Log directory not found: $LOG_DIR"
    fi
    exit 1
fi

# Find all matching log files, sort them, and delete all but the last 3
LOG_FILES=($(ls "$LOG_DIR"/persist__*.log 2>/dev/null | sort))
NUM_FILES=${#LOG_FILES[@]}

if [ "$NUM_FILES" -le 3 ]; then
    if [ "$QUIET" = false ]; then
        echo "[INFO] Only $NUM_FILES log file(s) found. Nothing to delete."
    fi
    exit 0
fi

FILES_TO_DELETE=("${LOG_FILES[@]:0:$((NUM_FILES - 3))}")

for file in "${FILES_TO_DELETE[@]}"; do
    rm "$file"
    if [ "$QUIET" = false ]; then
        if [ $? -eq 0 ]; then
            echo "[✔] Deleted: $file"
        else
            echo "[✘] Failed to delete: $file"
        fi
    fi
done

if [ "$QUIET" = false ]; then
    echo "[INFO] Kept last 3 log files."
fi
