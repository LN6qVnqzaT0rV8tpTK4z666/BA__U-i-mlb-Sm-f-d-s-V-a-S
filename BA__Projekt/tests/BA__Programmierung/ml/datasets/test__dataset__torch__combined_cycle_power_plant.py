import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import torch
from BA__Programmierung.ml.datasets.dataset__torch__combined_cycle_power_plant import DatasetTorchCombinedCyclePowerPlant


class TestDatasetTorchCombinedCyclePowerPlant(unittest.TestCase):

    @patch("pandas.read_csv")
    def test_initialize_dataset_with_normalization(self, mock_read_csv):
        # Mock the CSV data
        mock_df = pd.DataFrame({
            "AT": [20, 25, 30],
            "V": [40, 45, 50],
            "AP": [1000, 1010, 1020],
            "RH": [50, 55, 60],
            "PE": [100, 200, 300],
        })
        mock_read_csv.return_value = mock_df

        # Initialize the dataset with normalization
        dataset = DatasetTorchCombinedCyclePowerPlant(csv_path="fake_path.csv", normalize=True)

        # Ensure that the correct columns are selected as features and target
        self.assertEqual(dataset.input_cols, ["AT", "V", "AP", "RH"])
        self.assertEqual(dataset.target_col, ["PE"])

        # Check that normalization was applied (values should be between 0 and 1)
        self.assertTrue(torch.all(dataset.X >= 0) and torch.all(dataset.X <= 1))
        self.assertTrue(torch.all(dataset.y >= 0) and torch.all(dataset.y <= 1))

    @patch("pandas.read_csv")
    def test_initialize_dataset_without_normalization(self, mock_read_csv):
        # Mock the CSV data
        mock_df = pd.DataFrame({
            "AT": [20, 25, 30],
            "V": [40, 45, 50],
            "AP": [1000, 1010, 1020],
            "RH": [50, 55, 60],
            "PE": [100, 200, 300],
        })
        mock_read_csv.return_value = mock_df

        # Initialize the dataset without normalization
        dataset = DatasetTorchCombinedCyclePowerPlant(csv_path="fake_path.csv", normalize=False)

        # Ensure that the correct columns are selected as features and target
        self.assertEqual(dataset.input_cols, ["AT", "V", "AP", "RH"])
        self.assertEqual(dataset.target_col, ["PE"])

        # Check that normalization was not applied (values should be in original range)
        self.assertTrue(torch.all(dataset.X >= 0) and torch.all(dataset.X <= 255))
        self.assertTrue(torch.all(dataset.y >= 0) and torch.all(dataset.y <= 255))

    @patch("pandas.read_csv")
    def test_len(self, mock_read_csv):
        # Mock the CSV data
        mock_df = pd.DataFrame({
            "AT": [20, 25, 30],
            "V": [40, 45, 50],
            "AP": [1000, 1010, 1020],
            "RH": [50, 55, 60],
            "PE": [100, 200, 300],
        })
        mock_read_csv.return_value = mock_df

        # Initialize the dataset
        dataset = DatasetTorchCombinedCyclePowerPlant(csv_path="fake_path.csv", normalize=True)

        # Ensure dataset length matches number of samples in the CSV
        self.assertEqual(len(dataset), 3)

    @patch("pandas.read_csv")
    def test_get_item(self, mock_read_csv):
        # Mock the CSV data
        mock_df = pd.DataFrame({
            "AT": [20, 25, 30],
            "V": [40, 45, 50],
            "AP": [1000, 1010, 1020],
            "RH": [50, 55, 60],
            "PE": [100, 200, 300],
        })
        mock_read_csv.return_value = mock_df

        # Initialize the dataset
        dataset = DatasetTorchCombinedCyclePowerPlant(csv_path="fake_path.csv", normalize=True)

        # Test __getitem__
        image, label = dataset[0]

        # Ensure the image (features) and label are correct types
        self.assertIsInstance(image, torch.Tensor)
        self.assertEqual(label.item(), 100)

    @patch("pandas.read_csv")
    def test_missing_columns(self, mock_read_csv):
        # Mock the CSV data with missing required columns
        mock_df = pd.DataFrame({
            "AT": [20, 25, 30],
            "V": [40, 45, 50],
            "RH": [50, 55, 60],
            "PE": [100, 200, 300],
        })
        mock_read_csv.return_value = mock_df

        # Initialize the dataset and check if an error is raised
        with self.assertRaises(KeyError):
            dataset = DatasetTorchCombinedCyclePowerPlant(csv_path="fake_path.csv", normalize=True)


if __name__ == "__main__":
    unittest.main()
