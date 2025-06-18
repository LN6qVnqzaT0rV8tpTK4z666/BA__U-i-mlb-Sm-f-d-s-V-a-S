#!/bin/bash
# BA__Projekt/scripts/data-source__rename.sh

# Parse --quiet or --q flag
QUIET=false
for arg in "$@"; do
    if [[ "$arg" == "--quiet" || "$arg" == "--q" ]]; then
        QUIET=true
        break
    fi
done

# Absoluter oder relativer Pfad zum source-Datenordner
DATA_DIR="$(dirname "$0")/../assets/data/source"

if [ ! -d "$DATA_DIR" ]; then
  echo "Ordner $DATA_DIR existiert nicht."
  exit 1
fi

for file in "$DATA_DIR"/*; do
  if [ -f "$file" ]; then
    filename=$(basename "$file")

    if ! echo "$filename" | grep -q "^dataset__"; then
      sanitized_name=$(echo "$filename" | sed 's/+/_/g')
      new_name="dataset__${sanitized_name}"

      mv "$file" "$DATA_DIR/$new_name"
      [ "$QUIET" = false ] && echo "Renamed: $filename → $new_name"
    fi
  fi
done
