"""
Module for converting datasets from TXT format to CSV format.

This module provides functions to convert individual TXT files to CSV,
handling encoding issues gracefully, and to process entire directories
containing datasets in TXT format, saving converted CSV files while preserving
directory structure.

Functions:
    - convert_txt_to_csv(txt_path, output_path): Convert a single TXT file to CSV.
    - process_txt_datasets(base_input_dir, base_output_dir): Convert all TXT datasets
      found in the base input directory to CSV format in the base output directory.
"""

import os

import pandas as pd


def convert_txt_to_csv(txt_path, output_path):
    """
    Convert a single TXT file to a CSV file.

    The function attempts to read the TXT file using UTF-8 encoding, falling back to
    Latin-1 encoding if UTF-8 decoding fails. Each line is split by whitespace and
    saved as rows in a CSV file without headers or indexes.

    Args:
        txt_path (str or Path): Path to the input TXT file.
        output_path (str or Path): Path where the output CSV file will be saved.

    Prints:
        Conversion success message or error message on failure.
        Warning message if fallback encoding is used.
    """
    try:
        # Attempt to read with utf-8
        try:
            with open(txt_path, encoding="utf-8") as file:
                lines = file.readlines()
        except UnicodeDecodeError:
            # Fallback to latin1
            with open(txt_path, encoding="latin1") as file:
                lines = file.readlines()
            print(f"⚠️  Fallback to latin1 encoding for: {txt_path}")

        data = [line.strip().split() for line in lines if line.strip()]
        df = pd.DataFrame(data)

        # Save CSV explicitly as UTF-8 encoded
        df.to_csv(output_path, index=False, header=False, encoding="utf-8")
        print(f"Converted: {txt_path} → {output_path} (UTF-8)")
    except Exception as e:
        print(f"[Error] Failed to convert {txt_path}: {e}")


def process_txt_datasets(base_input_dir, base_output_dir):
    """
    Process all TXT datasets in the base input directory, converting them to CSV files.

    The function searches for subdirectories starting with "dataset__" inside the
    base input directory. For each such directory, it converts all valid TXT files
    (excluding files starting with "readme") into CSV files, preserving the relative
    directory structure under the base output directory.

    Args:
        base_input_dir (str or Path): Path to the directory containing TXT datasets.
        base_output_dir (str or Path): Path where converted CSV datasets will be saved.

    Prints:
        Informational messages about scanning directories, conversion progress,
        and warnings if no valid TXT files are found.
    """
    if not os.path.isdir(base_input_dir):
        print(f"[Error] Base input directory '{base_input_dir}' does not exist.")
        return

    os.makedirs(base_output_dir, exist_ok=True)

    for subdir in os.listdir(base_input_dir):
        subdir_path = os.path.join(base_input_dir, subdir)

        if os.path.isdir(subdir_path) and subdir.startswith("dataset__"):
            print(f"📁 Scanning: {subdir_path}")

            # Convert all valid .txt files (skip readme or hidden files)
            txt_files = [
                f for f in os.listdir(subdir_path)
                if f.endswith(".txt") and not f.lower().startswith("readme")
            ]

            if not txt_files:
                print(f"⚠️  No valid .txt file found in {subdir_path}")
                continue

            for file in txt_files:
                input_txt_path = os.path.join(subdir_path, file)
                relative_subdir = os.path.relpath(subdir_path, base_input_dir)
                output_subdir = os.path.join(base_output_dir, relative_subdir)
                os.makedirs(output_subdir, exist_ok=True)

                output_csv_path = os.path.join(
                    output_subdir, os.path.splitext(file)[0] + ".csv"
                )

                convert_txt_to_csv(input_txt_path, output_csv_path)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python conversion__txt_to_csv.py <input_base_dir> <output_base_dir>")
        sys.exit(1)

    input_base = sys.argv[1]
    output_base = sys.argv[2]

    process_txt_datasets(input_base, output_base)
