# BA__Projekt/BA__Programmierung/conv/conversion__arff_to_csv.py
"""
Module for converting datasets from ARFF format to CSV format.

This module provides functions to convert individual ARFF files to CSV,
and to process entire directories containing datasets in ARFF format,
saving converted CSV files preserving the directory structure.

Functions:
    - convert_arff_to_csv(arff_path, output_path): Convert a single ARFF file to CSV.
    - process_datasets(base_input_dir, base_output_dir): Convert all datasets found in
      base input directory from ARFF to CSV format, saving results in the base output directory.
"""

import os

import pandas as pd
from scipy.io import arff


def convert_arff_to_csv(arff_path, output_path):
    """
    Convert a single ARFF file to a CSV file.

    Args:
        arff_path (str or Path): Path to the input ARFF file.
        output_path (str or Path): Path where the output CSV file will be saved.

    Prints:
        Confirmation message on successful conversion, or error message if failed.
    """
    try:
        data, meta = arff.loadarff(arff_path)
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        print(f"Converted: {arff_path} → {output_path}")
    except Exception as e:
        print(f"[Error] Failed to convert {arff_path}: {e}")


def process_datasets(base_input_dir, base_output_dir):
    """
    Process all datasets in the base input directory, converting ARFF files to CSV.

    The function searches for subdirectories starting with "dataset__" inside the
    base input directory. For each such directory, it converts the first ARFF file found
    into CSV, preserving the relative directory structure under the base output directory.

    Args:
        base_input_dir (str or Path): Path to the directory containing ARFF datasets.
        base_output_dir (str or Path): Path where converted CSV datasets will be saved.

    Prints:
        Informational messages about scanning directories, conversion success,
        and warnings if no ARFF file is found.
    """
    if not os.path.isdir(base_input_dir):
        print(f"[Error] Base input directory '{base_input_dir}' does not exist.")
        return

    os.makedirs(base_output_dir, exist_ok=True)

    for subdir in os.listdir(base_input_dir):
        subdir_path = os.path.join(base_input_dir, subdir)

        if os.path.isdir(subdir_path) and subdir.startswith("dataset__"):
            print(f"📁 Scanning: {subdir_path}")

            # Look for .arff file in this dataset__ folder
            for file in os.listdir(subdir_path):
                if file.endswith(".arff"):
                    input_arff_path = os.path.join(subdir_path, file)
                    relative_subdir = os.path.relpath(subdir_path, base_input_dir)
                    output_subdir = os.path.join(base_output_dir, relative_subdir)
                    os.makedirs(output_subdir, exist_ok=True)

                    output_csv_path = os.path.join(
                        output_subdir, os.path.splitext(file)[0] + ".csv"
                    )

                    convert_arff_to_csv(input_arff_path, output_csv_path)
                    break  # Only convert the first .arff file found
            else:
                print(f"⚠️  No .arff file found in {subdir_path}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python conversion__arff_to_csv.py <input_base_dir> <output_base_dir>")
        sys.exit(1)

    input_base = sys.argv[1]
    output_base = sys.argv[2]

    process_datasets(input_base, output_base)
