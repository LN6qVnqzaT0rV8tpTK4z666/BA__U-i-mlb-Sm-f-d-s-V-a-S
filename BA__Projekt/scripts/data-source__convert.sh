#!/bin/bash
# BA__Projekt/scripts/data-source__convert.sh

echo "==> Prerequisite check for dataset conversion..."

# Check required Excel libraries
if ! python3 -c "import openpyxl" &>/dev/null; then
    echo "⛔ Missing dependency: openpyxl. Run 'pip install openpyxl'."
    exit 1
fi
if ! python3 -c "import xlrd" &>/dev/null; then
    echo "⛔ Missing dependency: xlrd (required for .xls). Run 'pip install xlrd'."
    exit 1
fi

echo "==> Starting dataset conversion..."

RAW_DIR="/root/BA__U-i-mlb-Sm-f-d-s-V-a-S/BA__Projekt/assets/data/raw"
OUTPUT_DIR="$RAW_DIR"

ARFF_CONVERTER="/root/BA__U-i-mlb-Sm-f-d-s-V-a-S/BA__Projekt/BA__Programmierung/conv/conversion__arff_to_csv.py"
EXCEL_CONVERTER="/root/BA__U-i-mlb-Sm-f-d-s-V-a-S/BA__Projekt/BA__Programmierung/conv/conversion__xlsx_to_csv.py"
TXT_CONVERTER="/root/BA__U-i-mlb-Sm-f-d-s-V-a-S/BA__Projekt/BA__Programmierung/conv/conversion__txt_to_csv.py"

for dataset_path in "$RAW_DIR"/dataset__*/; do
    dataset_name=$(basename "$dataset_path")
    output_path="$OUTPUT_DIR/$dataset_name"
    mkdir -p "$output_path"

    echo "📁 Processing: $dataset_name"

    # --- Excel Conversion (.xlsx / .xls) ---
    excel_file=$(find "$dataset_path" -maxdepth 1 \( -iname "*.xlsx" -o -iname "*.xls" \) | head -n 1)
    if [[ -n "$excel_file" ]]; then
        echo "📄 Found Excel file: $(basename "$excel_file")"
        python3 "$EXCEL_CONVERTER" "$RAW_DIR" "$OUTPUT_DIR"

        ext="${excel_file##*.}"
        base_name="$(basename "$excel_file" .$ext)"
        excel_csv="$output_path/$base_name.csv"

        if [[ -f "$excel_csv" ]]; then
            echo "✅ CSV created from $ext: $(basename "$excel_file")"
        else
            echo "❌ CSV not created from $ext: $(basename "$excel_file")"
        fi
    else
        echo "⚠️  No Excel file (.xlsx or .xls) found in $dataset_name"
    fi

    # --- ARFF Conversion ---
    arff_files=$(find "$dataset_path" -maxdepth 1 -name "*.arff")
    if [[ -n "$arff_files" ]]; then
        echo "📄 Found ARFF file(s)"
        python3 "$ARFF_CONVERTER" "$RAW_DIR" "$OUTPUT_DIR"

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

    # --- TXT Conversion (excluding README.txt) ---
    txt_file=$(find "$dataset_path" -maxdepth 1 -iname "*.txt" ! -iname "README.txt" | head -n 1)
    if [[ -n "$txt_file" ]]; then
        echo "📄 Found TXT: $(basename "$txt_file")"
        python3 "$TXT_CONVERTER" "$RAW_DIR" "$OUTPUT_DIR"

        txt_csv="$output_path/$(basename "$txt_file" .txt).csv"
        if [[ -f "$txt_csv" ]]; then
            echo "✅ CSV created from TXT: $(basename "$txt_file")"
        else
            echo "❌ CSV not created from TXT: $(basename "$txt_file")"
        fi
    else
        echo "⚠️  No TXT (excluding README) found in $dataset_name"
    fi

    echo "✅ Finished $dataset_name"
    echo "----------------------------"
done

echo "✅ All conversions complete."
