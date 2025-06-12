#!/bin/bash

print_tree() {
    local dir="$1"
    local prefix="$2"
    local current_depth="$3"
    local max_depth="$4"

    basename=$(basename "$dir")
    echo "${prefix}${basename}/"

    if [[ -n "$max_depth" && "$current_depth" -ge "$max_depth" ]]; then
        return
    fi

    local items=()
    while IFS= read -r -d '' item; do
        items+=("$item")
    done < <(find "$dir" -mindepth 1 -maxdepth 1 -print0 | sort -z)

    local count=${#items[@]}
    for ((i=0; i<count; i++)); do
        local path="${items[$i]}"
        local name=$(basename "$path")
        local is_last=$((i == count - 1))
        local connector="├── "
        [[ "$is_last" == 1 ]] && connector="└── "

        if [[ -d "$path" ]]; then
            echo "${prefix}${connector}${name}/"
            print_tree "$path" "${prefix}$( [[ $is_last == 1 ]] && echo "    " || echo "│   ")" $((current_depth + 1)) "$max_depth"
        else
            echo "${prefix}${connector}${name}"
        fi
    done
}

# ─── Accept Parameters ─────────────────────────────────────────────────────────

usage() {
    echo "Usage: $0 <target_folder> [max_depth]"
    echo "Example: $0 viz 1"
    exit 1
}

# Check for at least 1 argument
if [[ $# -lt 1 ]]; then
    usage
fi

root_input="$1"
depth_input="$2"

# Validate directory
if [[ ! -d "$root_input" ]]; then
    echo "[Error] Directory '$root_input' does not exist or is not a folder."
    exit 1
fi

# Validate depth
if [[ -n "$depth_input" && ! "$depth_input" =~ ^[0-9]+$ ]]; then
    echo "[Error] Depth must be a number."
    exit 1
fi

# Execute
print_tree "$root_input" "" 0 "$depth_input"
