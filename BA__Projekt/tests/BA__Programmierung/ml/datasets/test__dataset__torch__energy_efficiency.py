import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import torch
from dataset__torch__energy_efficiency import EnergyEfficiencyDataset


class TestEnergyEfficiencyDataset(unittest.TestCase):

    @patch("pandas.read_csv")
    def test_initialize_dataset_with_normalization(self, mock_read_csv):
        # Mock the CSV data
        mock_df = pd.DataFrame({
            "Feature1": [1, 2, 3],
            "Feature2": [4, 5, 6],
            "Feature3": [7, 8, 9],
            "Target": [10, 20, 30],
        })
        mock_read_csv.return_value = mock_df

        # Initialize the dataset with normalization
        dataset = EnergyEfficiencyDataset(csv_path="fake_path.csv", normalize=True)

        # Ensure that the correct columns are selected as features and target
        self.assertEqual(dataset.input_cols, ["Feature1", "Feature2", "Feature3"])
        self.assertEqual(dataset.target_col, "Target")

        # Check that normalization was applied (values should be between 0 and 1)
        self.assertTrue(torch.all(dataset.X >= 0) and torch.all(dataset.X <= 1))
        self.assertTrue(torch.all(dataset.y >= 0) and torch.all(dataset.y <= 1))

    @patch("pandas.read_csv")
    def test_initialize_dataset_without_normalization(self, mock_read_csv):
        # Mock the CSV data
        mock_df = pd.DataFrame({
            "Feature1": [1, 2, 3],
            "Feature2": [4, 5, 6],
            "Feature3": [7, 8, 9],
            "Target": [10, 20, 30],
        })
        mock_read_csv.return_value = mock_df

        # Initialize the dataset without normalization
        dataset = EnergyEfficiencyDataset(csv_path="fake_path.csv", normalize=False)

        # Ensure that the correct columns are selected as features and target
        self.assertEqual(dataset.input_cols, ["Feature1", "Feature2", "Feature3"])
        self.assertEqual(dataset.target_col, "Target")

        # Check that normalization was not applied (values should be in original range)
        self.assertTrue(torch.all(dataset.X >= 0) and torch.all(dataset.X <= 9))  # Check range of features
        self.assertTrue(torch.all(dataset.y >= 0) and torch.all(dataset.y <= 30))  # Check range of target

    @patch("pandas.read_csv")
    def test_len(self, mock_read_csv):
        # Mock the CSV data
        mock_df = pd.DataFrame({
            "Feature1": [1, 2, 3],
            "Feature2": [4, 5, 6],
            "Feature3": [7, 8, 9],
            "Target": [10, 20, 30],
        })
        mock_read_csv.return_value = mock_df

        # Initialize the dataset
        dataset = EnergyEfficiencyDataset(csv_path="fake_path.csv", normalize=True)

        # Ensure dataset length matches the number of samples in the CSV
        self.assertEqual(len(dataset), 3)

    @patch("pandas.read_csv")
    def test_get_item(self, mock_read_csv):
        # Mock the CSV data
        mock_df = pd.DataFrame({
            "Feature1": [1, 2, 3],
            "Feature2": [4, 5, 6],
            "Feature3": [7, 8, 9],
            "Target": [10, 20, 30],
        })
        mock_read_csv.return_value = mock_df

        # Initialize the dataset
        dataset = EnergyEfficiencyDataset(csv_path="fake_path.csv", normalize=True)

        # Test __getitem__
        image, label = dataset[0]

        # Ensure the image (features) and label are correct types
        self.assertIsInstance(image, torch.Tensor)
        self.assertEqual(label.item(), 10)  # First label should be 10

    @patch("pandas.read_csv")
    def test_missing_values(self, mock_read_csv):
        # Mock the CSV data with missing values
        mock_df = pd.DataFrame({
            "Feature1": [1, None, 3],
            "Feature2": [4, 5, None],
            "Feature3": [7, 8, 9],
            "Target": [10, None, 30],
        })
        mock_read_csv.return_value = mock_df

        # Initialize the dataset (with missing values dropped)
        dataset = EnergyEfficiencyDataset(csv_path="fake_path.csv", normalize=True)

        # Ensure rows with missing values are dropped (should be 2 samples left)
        self.assertEqual(len(dataset), 2)

    @patch("pandas.read_csv")
    def test_custom_target_column(self, mock_read_csv):
        # Mock the CSV data with a custom target column
        mock_df = pd.DataFrame({
            "Feature1": [1, 2, 3],
            "Feature2": [4, 5, 6],
            "Feature3": [7, 8, 9],
            "CustomTarget": [10, 20, 30],
        })
        mock_read_csv.return_value = mock_df

        # Initialize the dataset with a custom target column
        dataset = EnergyEfficiencyDataset(csv_path="fake_path.csv", target_col="CustomTarget", normalize=True)

        # Ensure that the custom target column is selected correctly
        self.assertEqual(dataset.input_cols, ["Feature1", "Feature2", "Feature3"])
        self.assertEqual(dataset.target_col, "CustomTarget")


if __name__ == "__main__":
    unittest.main()
