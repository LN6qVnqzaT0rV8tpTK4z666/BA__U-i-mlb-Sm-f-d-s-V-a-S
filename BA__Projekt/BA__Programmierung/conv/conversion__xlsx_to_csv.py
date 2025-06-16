# BA__Projekt/BA__Programmierung/conv/conversion__xlsx_to_csv.py
"""
Module for converting Excel files (.xlsx, .xls) to CSV format.

This module provides functions to convert a single Excel file to CSV,
and to process entire directories containing datasets in Excel format,
saving converted CSV files while preserving the directory structure.

Functions:
    - convert_excel_to_csv(excel_path, output_path): Convert one Excel file to CSV.
    - process_datasets(base_input_dir, base_output_dir): Convert all Excel datasets
      found in the base input directory to CSV format in the base output directory.
"""

import os

import pandas as pd


def convert_excel_to_csv(excel_path, output_path):
    """
    Convert a single Excel file to a CSV file.

    Args:
        excel_path (str or Path): Path to the input Excel file (.xlsx or .xls).
        output_path (str or Path): Path where the output CSV file will be saved.

    Prints:
        Success message on conversion, or error message on failure.
    """
    try:
        df = pd.read_excel(excel_path)
        df.to_csv(output_path, index=False)
        print(f"✅ Converted: {excel_path} → {output_path}")
    except Exception as e:
        print(f"[Error] Failed to convert {excel_path}: {e}")


def process_datasets(base_input_dir, base_output_dir):
    """
    Process all datasets in the base input directory, converting Excel files to CSV files.

    This function scans subdirectories starting with "dataset__" inside the base input directory.
    For each such directory, it converts the first found Excel file (.xlsx or .xls) to CSV,
    preserving the relative directory structure under the base output directory.

    Args:
        base_input_dir (str or Path): Directory containing Excel dataset subdirectories.
        base_output_dir (str or Path): Directory where converted CSV datasets will be saved.

    Prints:
        Informational messages about scanning directories, conversion progress,
        and warnings if no Excel file is found.
    """
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
