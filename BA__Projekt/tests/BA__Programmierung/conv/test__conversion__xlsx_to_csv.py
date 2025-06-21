# BA__Projekt/tests/BA__Programmierung/conv/test__conversion__xlsx_to_csv.py
import unittest
import os
from unittest.mock import patch
import pandas as pd
from conversion__xlsx_to_csv import convert_excel_to_csv, process_datasets


class TestConversionXlsxToCsv(unittest.TestCase):

    @patch("pandas.read_excel")
    @patch("pandas.DataFrame.to_csv")
    def test_convert_excel_to_csv_success(self, mock_to_csv, mock_read_excel):
        # Mock the content of the Excel file
        mock_df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        mock_read_excel.return_value = mock_df

        excel_path = "test_dataset.xlsx"
        output_path = "test_output.csv"

        # Call the function
        convert_excel_to_csv(excel_path, output_path)

        # Verify that read_excel was called with the correct path
        mock_read_excel.assert_called_once_with(excel_path)

        # Verify that to_csv was called with the correct output path
        mock_to_csv.assert_called_once_with(output_path, index=False)

    @patch("pandas.read_excel")
    def test_convert_excel_to_csv_failure(self, mock_read_excel):
        # Simulate an exception during the Excel file reading
        mock_read_excel.side_effect = Exception("File reading failed")

        excel_path = "test_dataset.xlsx"
        output_path = "test_output.csv"

        with self.assertRaises(Exception):
            convert_excel_to_csv(excel_path, output_path)

    @patch("os.makedirs")
    @patch("os.listdir")
    @patch("conversion__xlsx_to_csv.convert_excel_to_csv")
    def test_process_datasets(self, mock_convert, mock_listdir, mock_makedirs):
        # Mock directory structure
        mock_listdir.return_value = ["dataset__1", "dataset__2"]
        mock_listdir.side_effect = [
            ["dataset__1_file.xlsx"],  # Excel file in the first dataset
            [],  # No Excel files in the second dataset
        ]

        mock_convert.return_value = None
        mock_makedirs.return_value = None

        base_input_dir = "base_input"
        base_output_dir = "base_output"

        # Run the process_datasets function
        process_datasets(base_input_dir, base_output_dir)

        # Check that os.makedirs was called for the output directories
        mock_makedirs.assert_any_call(os.path.join(base_output_dir, "dataset__1"))
        mock_makedirs.assert_any_call(os.path.join(base_output_dir, "dataset__1", "dataset__1_file.csv"))

        # Verify that convert_excel_to_csv was called for the first dataset
        mock_convert.assert_called_once_with(
            os.path.join(base_input_dir, "dataset__1", "dataset__1_file.xlsx"),
            os.path.join(base_output_dir, "dataset__1", "dataset__1_file.csv")
        )

    @patch("os.makedirs")
    @patch("os.listdir")
    def test_process_datasets_no_xlsx(self, mock_listdir, mock_makedirs):
        # Mock directory structure with no Excel file
        mock_listdir.return_value = ["dataset__1"]
        mock_listdir.side_effect = [["some_other_file.txt"]]  # No Excel files

        mock_makedirs.return_value = None

        base_input_dir = "base_input"
        base_output_dir = "base_output"

        # Run the process_datasets function
        process_datasets(base_input_dir, base_output_dir)

        # Check that os.makedirs was called for the output directories
        mock_makedirs.assert_any_call(os.path.join(base_output_dir, "dataset__1"))

    def test_convert_excel_to_csv_with_missing_file(self):
        # Check if the method handles missing file gracefully
        with self.assertRaises(OSError):
            convert_excel_to_csv("nonexistent_file.xlsx", "output.csv")

    def test_process_datasets_missing_directory(self):
        # Check if the method handles missing input directory gracefully
        with self.assertRaises(OSError):
            process_datasets("nonexistent_input_dir", "output_dir")


if __name__ == '__main__':
    unittest.main()

