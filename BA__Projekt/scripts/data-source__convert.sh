#!/bin/bash
# BA__Projekt/scripts/data-source__convert.sh

echo "==> Prerequisite check for dataset conversion..."

# Check if openpyxl is installed
if ! python3 -c "import openpyxl" &>/dev/null; then
    echo "⛔ Missing dependency: openpyxl. Run 'pip install openpyxl'."
    exit 1
fi

echo "==> Starting dataset conversion..."

# Paths
RAW_DIR="/root/BA__U-i-mlb-Sm-f-d-s-V-a-S/BA__Projekt/assets/data/raw"
OUTPUT_DIR="/root/BA__U-i-mlb-Sm-f-d-s-V-a-S/BA__Projekt/assets/data/raw"

ARFF_CONVERTER="/root/BA__U-i-mlb-Sm-f-d-s-V-a-S/BA__Projekt/BA__Programmierung/conv/conversion__arff_to_csv.py"
XLSX_CONVERTER="/root/BA__U-i-mlb-Sm-f-d-s-V-a-S/BA__Projekt/BA__Programmierung/conv/conversion__xlsx_to_csv.py"

# Loop through dataset folders
for dataset_path in "$RAW_DIR"/dataset__*/; do
    dataset_name=$(basename "$dataset_path")
    output_path="$OUTPUT_DIR/$dataset_name"
    mkdir -p "$output_path"

    echo "📁 Processing: $dataset_name"

    # --- XLSX Conversion ---
    xlsx_file=$(find "$dataset_path" -maxdepth 1 -name "*.xlsx" | head -n 1)
    if [[ -n "$xlsx_file" ]]; then
        echo "📄 Found XLSX: $(basename "$xlsx_file")"
        python3 "$XLSX_CONVERTER" "$RAW_DIR" "$OUTPUT_DIR"

        # Check if corresponding CSV was created
        xlsx_csv="$output_path/$(basename "$xlsx_file" .xlsx).csv"
        if [[ -f "$xlsx_csv" ]]; then
            echo "✅ CSV created from XLSX: $(basename "$xlsx_file")"
        else
            echo "❌ CSV not created from XLSX: $(basename "$xlsx_file")"
        fi
    else
        echo "⚠️  No XLSX found in $dataset_name"
    fi

    # --- ARFF Conversion ---
    arff_files=$(find "$dataset_path" -maxdepth 1 -name "*.arff")
    if [[ -n "$arff_files" ]]; then
        echo "📄 Found ARFF file(s)"

        # Corrected call: pass base RAW_DIR, not specific dataset path
        python3 "$ARFF_CONVERTER" "$RAW_DIR" "$OUTPUT_DIR"

        # Check each ARFF file for CSV output
        for arff_file in $arff_files; do
            arff_csv="$output_path/$(basename "$arff_file" .arff).csv"
            if [[ -f "$arff_csv" ]]; then
                echo "✅ CSV created from ARFF: $(basename "$arff_file")"
            else
                echo "❌ CSV not created from ARFF: $(basename "$arff_file")"
            fi
        done
    else
        echo "⚠️  No ARFF files found in $dataset_name"
    fi

    echo "✅ Finished $dataset_name"
    echo "----------------------------"
done

echo "✅ All conversions complete."
