# BA__Projekt/tests/BA__Programmierung/conv/test__conversion__arff_to_csv.py
import unittest
import os
from unittest.mock import patch, MagicMock
from conversion__arff_to_csv import convert_arff_to_csv, process_datasets


class TestConversionArffToCsv(unittest.TestCase):

    @patch("conversion__arff_to_csv.arff.loadarff")
    @patch("pandas.DataFrame.to_csv")
    def test_convert_arff_to_csv_success(self, mock_to_csv, mock_loadarff):
        # Mocking the ARFF data and Meta
        mock_data = [(1, 2), (3, 4)]  # Mock ARFF data
        mock_meta = MagicMock()  # Mock meta data
        mock_loadarff.return_value = (mock_data, mock_meta)

        # Mocking the to_csv function to not actually write to the filesystem
        mock_to_csv.return_value = None

        # Define paths for the ARFF and output CSV
        arff_path = "test_dataset.arff"
        output_path = "test_output.csv"

        # Call the function
        convert_arff_to_csv(arff_path, output_path)

        # Verify that arff.loadarff was called with the correct ARFF path
        mock_loadarff.assert_called_once_with(arff_path)

        # Verify that to_csv was called with the correct output path
        mock_to_csv.assert_called_once_with(output_path, index=False)

    @patch("conversion__arff_to_csv.arff.loadarff")
    def test_convert_arff_to_csv_failure(self, mock_loadarff):
        # Simulate failure of loading ARFF data
        mock_loadarff.side_effect = Exception("Failed to load ARFF file")

        # Define paths
        arff_path = "test_dataset.arff"
        output_path = "test_output.csv"

        with self.assertRaises(Exception):
            convert_arff_to_csv(arff_path, output_path)

    @patch("os.makedirs")
    @patch("os.listdir")
    @patch("conversion__arff_to_csv.convert_arff_to_csv")
    def test_process_datasets(self, mock_convert, mock_listdir, mock_makedirs):
        # Mock directory structure
        mock_listdir.return_value = ["dataset__1", "dataset__2"]
        mock_listdir.side_effect = [
            ["dataset__1_file.arff"],  # ARFF file in first dataset
            [],  # No ARFF file in second dataset
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

        # Verify that convert_arff_to_csv was called for the first dataset
        mock_convert.assert_called_once_with(
            os.path.join(base_input_dir, "dataset__1", "dataset__1_file.arff"),
            os.path.join(base_output_dir, "dataset__1", "dataset__1_file.csv")
        )

    @patch("os.makedirs")
    @patch("os.listdir")
    def test_process_datasets_no_arff(self, mock_listdir, mock_makedirs):
        # Mock directory structure with no ARFF file
        mock_listdir.return_value = ["dataset__1"]
        mock_listdir.side_effect = [["some_other_file.txt"]]  # No ARFF files

        mock_makedirs.return_value = None

        base_input_dir = "base_input"
        base_output_dir = "base_output"

        # Run the process_datasets function
        process_datasets(base_input_dir, base_output_dir)

        # Check that os.makedirs was called for the output directories
        mock_makedirs.assert_any_call(os.path.join(base_output_dir, "dataset__1"))

    def test_convert_arff_to_csv_with_missing_file(self):
        # Check if the method handles missing file gracefully
        with self.assertRaises(OSError):
            convert_arff_to_csv("nonexistent_file.arff", "output.csv")

    def test_process_datasets_missing_directory(self):
        # Check if the method handles missing input directory gracefully
        with self.assertRaises(OSError):
            process_datasets("nonexistent_input_dir", "output_dir")


if __name__ == '__main__':
    unittest.main()

