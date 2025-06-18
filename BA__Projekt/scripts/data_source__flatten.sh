#!/bin/bash
# BA__Projekt/scripts/data-source__flatten.sh

# Parse --quiet or --q flag
QUIET=false
for arg in "$@"; do
    if [[ "$arg" == "--quiet" || "$arg" == "--q" ]]; then
        QUIET=true
        break
    fi
done

# Absolute path to this script's directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="$SCRIPT_DIR/../assets/data/raw"
TARGET_DIR="$(realpath "$TARGET_DIR")"

[ "$QUIET" = false ] && echo "Flattening directories under: $TARGET_DIR" && echo

flatten_leaf() {
    local leaf_dir="$1"
    local moved_anything=true

    while $moved_anything; do
        moved_anything=false

        while IFS= read -r file; do
            filename="$(basename "$file")"
            dest="$leaf_dir/$filename"

            if [ -e "$dest" ]; then
                base="$filename"
                ext=""

                if [[ "$filename" == *.* ]]; then
                    base="${filename%.*}"
                    ext=".${filename##*.}"
                fi

                i=1
                while [ -e "$leaf_dir/${base}__${i}${ext}" ]; do
                    ((i++))
                done
                dest="$leaf_dir/${base}__${i}${ext}"
                [ "$QUIET" = false ] && echo "⚠️  Renaming to avoid conflict: $filename → $(basename "$dest")"
            fi

            mv "$file" "$dest"
            moved_anything=true
        done < <(find "$leaf_dir" -mindepth 2 -type f)

        find "$leaf_dir" -type d -mindepth 1 -empty -delete
    done
}

for leaf in "$TARGET_DIR"/*/; do
    [ -d "$leaf" ] || continue
    [ "$QUIET" = false ] && echo "🔄 Flattening: $leaf"
    flatten_leaf "$leaf"
done

[ "$QUIET" = false ] && echo && echo "✅ Done: All dataset directories have been recursively flattened with conflict resolution."
