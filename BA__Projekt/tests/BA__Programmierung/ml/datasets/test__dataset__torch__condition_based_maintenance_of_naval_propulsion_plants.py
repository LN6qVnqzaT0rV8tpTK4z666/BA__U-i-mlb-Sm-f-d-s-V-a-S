# BA__Projekt/tests/BA__Programmierung/ml/datasets/test__dataset__torch__condition_based_maintenance_of_naval_propulsion_plants.py
import unittest
from unittest.mock import patch
import pandas as pd
import torch
from BA__Programmierung.ml.datasets.dataset__torch__condition_based_maintenance_of_naval_propulsion_plants import NavalPropulsionDataset


class TestNavalPropulsionDataset(unittest.TestCase):

    @patch("pandas.read_csv")
    def test_initialize_dataset_with_normalization(self, mock_read_csv):
        # Mock the CSV data (18 columns)
        mock_df = pd.DataFrame({
            **{i: [1] * 10 for i in range(16)},  # 16 feature columns
            16: [0] * 10,  # Target column 1
            17: [0] * 10   # Target column 2
        })
        mock_read_csv.return_value = mock_df

        # Initialize the dataset with normalization
        dataset = NavalPropulsionDataset(csv_path="fake_path.csv", normalize=True)

        # Ensure that the correct columns are selected as features and target
        self.assertEqual(dataset.input_cols, ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15'])
        self.assertEqual(dataset.target_cols, ['16', '17'])

        # Check that normalization was applied (values should be between 0 and 1)
        self.assertTrue(torch.all(dataset.X >= 0) and torch.all(dataset.X <= 1))
        self.assertTrue(torch.all(dataset.y >= 0) and torch.all(dataset.y <= 1))

    @patch("pandas.read_csv")
    def test_initialize_dataset_without_normalization(self, mock_read_csv):
        # Mock the CSV data (18 columns)
        mock_df = pd.DataFrame({
            **{i: [1] * 10 for i in range(16)},  # 16 feature columns
            16: [0] * 10,  # Target column 1
            17: [0] * 10   # Target column 2
        })
        mock_read_csv.return_value = mock_df

        # Initialize the dataset without normalization
        dataset = NavalPropulsionDataset(csv_path="fake_path.csv", normalize=False)

        # Ensure that the correct columns are selected as features and target
        self.assertEqual(dataset.input_cols, ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15'])
        self.assertEqual(dataset.target_cols, ['16', '17'])

        # Check that normalization was not applied (values should be in original range)
        self.assertTrue(torch.all(dataset.X >= 0) and torch.all(dataset.X <= 255))  # Check range of features
        self.assertTrue(torch.all(dataset.y >= 0) and torch.all(dataset.y <= 255))  # Check range of target

    @patch("pandas.read_csv")
    def test_len(self, mock_read_csv):
        # Mock the CSV data (18 columns)
        mock_df = pd.DataFrame({
            **{i: [1] * 10 for i in range(16)},  # 16 feature columns
            16: [0] * 10,  # Target column 1
            17: [0] * 10   # Target column 2
        })
        mock_read_csv.return_value = mock_df

        # Initialize the dataset
        dataset = NavalPropulsionDataset(csv_path="fake_path.csv", normalize=True)

        # Ensure dataset length matches number of samples in the CSV
        self.assertEqual(len(dataset), 10)

    @patch("pandas.read_csv")
    def test_get_item(self, mock_read_csv):
        # Mock the CSV data (18 columns)
        mock_df = pd.DataFrame({
            **{i: [1] * 10 for i in range(16)},  # 16 feature columns
            16: [0] * 10,  # Target column 1
            17: [0] * 10   # Target column 2
        })
        mock_read_csv.return_value = mock_df

        # Initialize the dataset
        dataset = NavalPropulsionDataset(csv_path="fake_path.csv", normalize=True)

        # Test __getitem__
        image, label = dataset[0]

        # Ensure the image (features) and label are correct types
        self.assertIsInstance(image, torch.Tensor)
        self.assertEqual(label.item(), 0)  # First label should be 0

    @patch("pandas.read_csv")
    def test_invalid_column_count(self, mock_read_csv):
        # Mock the CSV data with an invalid number of columns (e.g., 17 columns instead of 18)
        mock_df = pd.DataFrame({
            **{i: [1] * 10 for i in range(15)},  # 15 feature columns (incorrect count)
            15: [0] * 10,  # Target column 1
        })
        mock_read_csv.return_value = mock_df

        # Initialize the dataset and check for ValueError
        with self.assertRaises(ValueError):
            dataset = NavalPropulsionDataset(csv_path="fake_path.csv", normalize=True)


if __name__ == "__main__":
    unittest.main()

