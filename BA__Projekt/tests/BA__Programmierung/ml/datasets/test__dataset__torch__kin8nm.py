import unittest
from unittest.mock import patch, MagicMock
import os
import pandas as pd
from dataset__torch__kin8nm import load_kin8nm_dataset
from BA__Programmierung.ml.datasets.dataset__torch__base_tabular import BaseTabularDataset


class TestKin8nmDataset(unittest.TestCase):

    @patch("os.path.isfile")
    @patch("pandas.read_csv")
    def test_load_kin8nm_dataset(self, mock_read_csv, mock_isfile):
        # Mocking the check for the file existence to return True
        mock_isfile.return_value = True

        # Mock the CSV data
        mock_df = pd.DataFrame({
            "Feature1": [1, 2, 3],
            "Feature2": [4, 5, 6],
            "Feature3": [7, 8, 9],
            "Target": [10, 20, 30],
        })
        mock_read_csv.return_value = mock_df

        # Call the function to load the dataset
        csv_path = "fake_path/kin8nm.csv"
        dataset = load_kin8nm_dataset(csv_path)

        # Verify that the correct columns were identified as features and target
        self.assertEqual(dataset.input_cols, ["Feature1", "Feature2", "Feature3"])
        self.assertEqual(dataset.target_col, "Target")

        # Verify that the dataset size matches the number of rows in the CSV
        self.assertEqual(len(dataset), 3)

        # Ensure that normalization is applied (values should be between 0 and 1)
        self.assertTrue(torch.all(dataset.X >= 0) and torch.all(dataset.X <= 1))
        self.assertTrue(torch.all(dataset.y >= 0) and torch.all(dataset.y <= 30))

    @patch("os.path.isfile")
    def test_file_not_found(self, mock_isfile):
        # Mocking the check for the file existence to return False
        mock_isfile.return_value = False

        # Call the function with a non-existent file path and verify FileNotFoundError is raised
        with self.assertRaises(FileNotFoundError):
            load_kin8nm_dataset("fake_path/kin8nm.csv")

    @patch("os.path.isfile")
    @patch("pandas.read_csv")
    def test_dataset_loading_with_missing_column(self, mock_read_csv, mock_isfile):
        # Mocking the check for the file existence to return True
        mock_isfile.return_value = True

        # Mock the CSV data with a missing target column
        mock_df = pd.DataFrame({
            "Feature1": [1, 2, 3],
            "Feature2": [4, 5, 6],
            "Feature3": [7, 8, 9],
        })
        mock_read_csv.return_value = mock_df

        # Call the function and verify that the last column is used as the target
        csv_path = "fake_path/kin8nm.csv"
        dataset = load_kin8nm_dataset(csv_path)

        # Ensure that the last column is treated as the target column
        self.assertEqual(dataset.target_col, "Feature3")

    @patch("os.path.isfile")
    @patch("pandas.read_csv")
    def test_dataset_size(self, mock_read_csv, mock_isfile):
        # Mocking the check for the file existence to return True
        mock_isfile.return_value = True

        # Mock the CSV data
        mock_df = pd.DataFrame({
            "Feature1": [1, 2, 3],
            "Feature2": [4, 5, 6],
            "Feature3": [7, 8, 9],
            "Target": [10, 20, 30],
        })
        mock_read_csv.return_value = mock_df

        # Call the function to load the dataset
        csv_path = "fake_path/kin8nm.csv"
        dataset = load_kin8nm_dataset(csv_path)

        # Verify that the dataset size is equal to the number of rows in the CSV
        self.assertEqual(len(dataset), 3)

    @patch("os.path.isfile")
    @patch("pandas.read_csv")
    def test_input_column_names(self, mock_read_csv, mock_isfile):
        # Mocking the check for the file existence to return True
        mock_isfile.return_value = True

        # Mock the CSV data with specific feature columns
        mock_df = pd.DataFrame({
            "Feature1": [1, 2, 3],
            "Feature2": [4, 5, 6],
            "Feature3": [7, 8, 9],
            "Target": [10, 20, 30],
        })
        mock_read_csv.return_value = mock_df

        # Call the function to load the dataset
        csv_path = "fake_path/kin8nm.csv"
        dataset = load_kin8nm_dataset(csv_path)

        # Ensure that the correct columns are selected as features
        self.assertEqual(dataset.input_cols, ["Feature1", "Feature2", "Feature3"])


if __name__ == "__main__":
    unittest.main()
