#!/bin/bash
# BA__Projekt/scripts/data-source__rename-csv.sh

RAW_DIR="/root/BA__U-i-mlb-Sm-f-d-s-V-a-S/BA__Projekt/assets/data/raw"

# Handle --quiet or --q flag
QUIET=false
for arg in "$@"; do
    if [[ "$arg" == "--quiet" || "$arg" == "--q" ]]; then
        QUIET=true
        break
    fi
done

[ "$QUIET" = false ] && echo "==> Renaming single CSV files to match their dataset folder name..."

for dataset_path in "$RAW_DIR"/dataset__*/; do
    dataset_name=$(basename "$dataset_path")
    csv_files=("$dataset_path"/*.csv)

    if [[ ${#csv_files[@]} -eq 1 ]]; then
        original_csv="${csv_files[0]}"
        new_csv="$dataset_path/$dataset_name.csv"

        if [[ "$original_csv" != "$new_csv" ]]; then
            mv "$original_csv" "$new_csv"
            [ "$QUIET" = false ] && echo "🔁 Renamed $(basename "$original_csv") → $dataset_name.csv"
        else
            [ "$QUIET" = false ] && echo "✔️  CSV already correctly named in $dataset_name"
        fi
    else
        [ "$QUIET" = false ] && echo "⚠️  Skipping $dataset_name (contains ${#csv_files[@]} CSV files)"
    fi
done

[ "$QUIET" = false ] && echo "✅ Renaming process complete."
