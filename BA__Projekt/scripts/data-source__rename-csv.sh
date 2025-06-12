#!/bin/bash
# BA__Projekt/scripts/data-source__rename-csv.sh

RAW_DIR="/root/BA__U-i-mlb-Sm-f-d-s-V-a-S/BA__Projekt/assets/data/raw"

echo "==> Renaming single CSV files to match their dataset folder name..."

for dataset_path in "$RAW_DIR"/dataset__*/; do
    dataset_name=$(basename "$dataset_path")
    csv_files=("$dataset_path"/*.csv)

    # Check how many CSV files exist in the directory
    if [[ ${#csv_files[@]} -eq 1 ]]; then
        original_csv="${csv_files[0]}"
        new_csv="$dataset_path/$dataset_name.csv"

        # Rename only if the name differs
        if [[ "$original_csv" != "$new_csv" ]]; then
            mv "$original_csv" "$new_csv"
            echo "🔁 Renamed $(basename "$original_csv") → $dataset_name.csv"
        else
            echo "✔️  CSV already correctly named in $dataset_name"
        fi
    else
        echo "⚠️  Skipping $dataset_name (contains ${#csv_files[@]} CSV files)"
    fi
done

echo "✅ Renaming process complete."
