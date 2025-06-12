#!/bin/bash

# Resolve the directory of the script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Compute the path to the data folder relative to the script
TARGET="${SCRIPT_DIR}/../assets/data/raw"

# Normalize and resolve symlinks
TARGET="$(realpath "$TARGET")"

# Check if the directory exists
if [[ ! -d "$TARGET" ]]; then
    echo "[Error] Directory '$TARGET' does not exist."
    exit 1
fi

# Iterate over each first-level subdirectory ("leaf")
find "$TARGET" -mindepth 1 -maxdepth 1 -type d | while read -r leaf; do
    echo "Flattening: $leaf"

    # Move all files from subdirectories (depth ≥ 2) into the leaf root
    find "$leaf" -mindepth 2 -type f -exec mv -n "{}" "$leaf/" \;

    # Remove all empty subdirectories
    find "$leaf" -mindepth 1 -type d -empty -delete
done

echo "Flattening complete."
