#!/bin/bash

# Check args
if [ $# -ne 3 ]; then
    echo "Usage: $0 <prefix> <extension> <target_folder>"
    exit 1
fi

new_prefix="$1"
file_ext="$2"
target_folder="$3"

# Prepare target directory
mkdir -p "$target_folder"

# Initialize token array
tokens=()

# Read input from piped tree output
while IFS= read -r line; do
    # Match depth=1 lines with files
    if [[ "$line" =~ ^[[:space:]]*[├└]──[[:space:]]*([a-zA-Z0-9_.-]+)$ ]]; then
        file="${BASH_REMATCH[1]}"

        # Skip __init__.py
        [[ "$file" == "__init__.py" ]] && continue

        # Remove file extension
        base="${file%.*}"

        # Extract everything after the last '__'
        if [[ "$base" =~ .*__([^_][^_]*)$ ]]; then
            token="${BASH_REMATCH[1]}"
            tokens+=("$token")
        else
            echo "[Warning] Skipping file (no token found): $file"
        fi
    fi
done

# Check we found tokens
if [ "${#tokens[@]}" -eq 0 ]; then
    echo "[Error] No valid tokens extracted."
    exit 1
fi

# Create files using prefix + token
for token in "${tokens[@]}"; do
    new_file="${target_folder}/${new_prefix}__${token}.${file_ext}"
    touch "$new_file"
    echo "Created: $new_file"
done
