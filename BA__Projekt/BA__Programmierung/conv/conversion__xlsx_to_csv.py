# BA__Projekt/BA__Programmierung/conv/conversion__xlsx_to_csv.py

import os
import pandas as pd


def convert_excel_to_csv(excel_path, output_path):
    try:
        df = pd.read_excel(excel_path)
        df.to_csv(output_path, index=False)
        print(f"✅ Converted: {excel_path} → {output_path}")
    except Exception as e:
        print(f"[Error] Failed to convert {excel_path}: {e}")


def process_datasets(base_input_dir, base_output_dir):
    if not os.path.isdir(base_input_dir):
        print(f"[Error] Base input directory '{base_input_dir}' does not exist.")
        return

    os.makedirs(base_output_dir, exist_ok=True)

    for subdir in os.listdir(base_input_dir):
        subdir_path = os.path.join(base_input_dir, subdir)

        if os.path.isdir(subdir_path) and subdir.startswith("dataset__"):
            print(f"📁 Scanning: {subdir_path}")

            # Look for .xlsx or .xls file in this dataset__ folder
            for file in os.listdir(subdir_path):
                if file.endswith((".xlsx", ".xls")):
                    input_excel_path = os.path.join(subdir_path, file)
                    relative_subdir = os.path.relpath(subdir_path, base_input_dir)
                    output_subdir = os.path.join(base_output_dir, relative_subdir)
                    os.makedirs(output_subdir, exist_ok=True)

                    output_csv_path = os.path.join(
                        output_subdir, os.path.splitext(file)[0] + ".csv"
                    )

                    convert_excel_to_csv(input_excel_path, output_csv_path)
                    break  # Only convert the first Excel file found
            else:
                print(f"⚠️  No .xlsx or .xls file found in {subdir_path}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python conversion__xlsx_to_csv.py <input_base_dir> <output_base_dir>")
        sys.exit(1)

    input_base = sys.argv[1]
    output_base = sys.argv[2]

    process_datasets(input_base, output_base)
