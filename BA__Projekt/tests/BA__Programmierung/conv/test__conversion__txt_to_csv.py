# BA__Projekt/tests/BA__Programmierung/conv/test__conversion__txt_to_csv.py
import unittest
import os
from unittest.mock import patch, MagicMock
from conversion__txt_to_csv import convert_txt_to_csv, process_txt_datasets


class TestConversionTxtToCsv(unittest.TestCase):

    @patch("builtins.open", new_callable=MagicMock)
    @patch("pandas.DataFrame.to_csv")
    def test_convert_txt_to_csv_success(self, mock_to_csv, mock_open):
        # Mock the content of the file
        mock_open.return_value.__enter__.return_value.readlines.return_value = [
            "1 2 3",
            "4 5 6",
        ]

        txt_path = "test_dataset.txt"
        output_path = "test_output.csv"

        # Call the function
        convert_txt_to_csv(txt_path, output_path)

        # Verify that the file was read correctly
        mock_open.assert_called_with(txt_path, encoding="utf-8")

        # Verify that DataFrame's to_csv method was called with the correct output path
        mock_to_csv.assert_called_once_with(output_path, index=False, header=False, encoding="utf-8")

    @patch("builtins.open", new_callable=MagicMock)
    @patch("pandas.DataFrame.to_csv")
    def test_convert_txt_to_csv_fallback_encoding(self, mock_to_csv, mock_open):
        # Simulate a UnicodeDecodeError for UTF-8 encoding
        mock_open.side_effect = [
            UnicodeDecodeError("utf-8", b"", 0, 1, "invalid"),
            MagicMock(return_value=["1 2 3", "4 5 6"])
        ]

        txt_path = "test_dataset.txt"
        output_path = "test_output.csv"

        # Call the function
        convert_txt_to_csv(txt_path, output_path)

        # Verify that the file was read using latin1 encoding after the failure
        mock_open.assert_any_call(txt_path, encoding="latin1")

        # Verify that DataFrame's to_csv method was called with the correct output path
        mock_to_csv.assert_called_once_with(output_path, index=False, header=False, encoding="utf-8")

    @patch("builtins.open", new_callable=MagicMock)
    def test_convert_txt_to_csv_failure(self, mock_open):
        # Simulate a failure when reading the TXT file
        mock_open.side_effect = Exception("File not found")

        txt_path = "test_dataset.txt"
        output_path = "test_output.csv"

        with self.assertRaises(Exception):
            convert_txt_to_csv(txt_path, output_path)

    @patch("os.makedirs")
    @patch("os.listdir")
    @patch("conversion__txt_to_csv.convert_txt_to_csv")
    def test_process_txt_datasets(self, mock_convert, mock_listdir, mock_makedirs):
        # Mock directory structure
        mock_listdir.return_value = ["dataset__1", "dataset__2"]
        mock_listdir.side_effect = [
            ["dataset__1_file.txt"],  # Valid TXT file in the first dataset
            [],  # No TXT files in the second dataset
        ]

        mock_convert.return_value = None
        mock_makedirs.return_value = None

        base_input_dir = "base_input"
        base_output_dir = "base_output"

        # Run the process_txt_datasets function
        process_txt_datasets(base_input_dir, base_output_dir)

        # Check that os.makedirs was called for the output directories
        mock_makedirs.assert_any_call(os.path.join(base_output_dir, "dataset__1"))
        mock_makedirs.assert_any_call(os.path.join(base_output_dir, "dataset__1", "dataset__1_file.csv"))

        # Verify that convert_txt_to_csv was called for the first dataset
        mock_convert.assert_called_once_with(
            os.path.join(base_input_dir, "dataset__1", "dataset__1_file.txt"),
            os.path.join(base_output_dir, "dataset__1", "dataset__1_file.csv")
        )

    @patch("os.makedirs")
    @patch("os.listdir")
    def test_process_txt_datasets_no_txt(self, mock_listdir, mock_makedirs):
        # Mock directory structure with no TXT file
        mock_listdir.return_value = ["dataset__1"]
        mock_listdir.side_effect = [["some_other_file.txt"]]  # No TXT files

        mock_makedirs.return_value = None

        base_input_dir = "base_input"
        base_output_dir = "base_output"

        # Run the process_txt_datasets function
        process_txt_datasets(base_input_dir, base_output_dir)

        # Check that os.makedirs was called for the output directories
        mock_makedirs.assert_any_call(os.path.join(base_output_dir, "dataset__1"))

    def test_convert_txt_to_csv_with_missing_file(self):
        # Check if the method handles missing file gracefully
        with self.assertRaises(OSError):
            convert_txt_to_csv("nonexistent_file.txt", "output.csv")

    def test_process_txt_datasets_missing_directory(self):
        # Check if the method handles missing input directory gracefully
        with self.assertRaises(OSError):
            process_txt_datasets("nonexistent_input_dir", "output_dir")


if __name__ == '__main__':
    unittest.main()

