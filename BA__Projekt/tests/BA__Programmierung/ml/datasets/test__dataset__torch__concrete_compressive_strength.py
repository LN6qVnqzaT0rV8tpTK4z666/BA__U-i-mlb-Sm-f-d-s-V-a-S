import unittest
from unittest.mock import patch
import pandas as pd
import torch
from BA__Programmierung.ml.datasets.dataset__torch__concrete_compressive_strength import DatasetTorchConcreteCompressiveStrength


class TestDatasetTorchConcreteCompressiveStrength(unittest.TestCase):

    @patch("pandas.read_csv")
    def test_initialize_dataset_with_normalization(self, mock_read_csv):
        # Mock the CSV data
        mock_df = pd.DataFrame({
            "Cement": [500, 600, 700],
            "Blast Furnace Slag": [100, 150, 200],
            "Fly Ash": [100, 200, 300],
            "Water": [160, 170, 180],
            "Superplasticizer": [10, 15, 20],
            "Coarse Aggregate": [1000, 1050, 1100],
            "Fine Aggregate": [700, 750, 800],
            "Age": [28, 28, 28],
            "Concrete Compressive Strength": [30, 35, 40],
        })
        mock_read_csv.return_value = mock_df

        # Initialize the dataset with normalization
        dataset = DatasetTorchConcreteCompressiveStrength(csv_path="fake_path.csv", normalize=True)

        # Ensure that the correct columns are selected as features and target
        self.assertEqual(dataset.input_cols, ["Cement", "Blast Furnace Slag", "Fly Ash", "Water", 
                                              "Superplasticizer", "Coarse Aggregate", "Fine Aggregate", "Age"])
        self.assertEqual(dataset.target_col, "Concrete Compressive Strength")

        # Check that normalization was applied (values should be between 0 and 1)
        self.assertTrue(torch.all(dataset.X >= 0) and torch.all(dataset.X <= 1))
        self.assertTrue(torch.all(dataset.y >= 0) and torch.all(dataset.y <= 1))

    @patch("pandas.read_csv")
    def test_initialize_dataset_without_normalization(self, mock_read_csv):
        # Mock the CSV data
        mock_df = pd.DataFrame({
            "Cement": [500, 600, 700],
            "Blast Furnace Slag": [100, 150, 200],
            "Fly Ash": [100, 200, 300],
            "Water": [160, 170, 180],
            "Superplasticizer": [10, 15, 20],
            "Coarse Aggregate": [1000, 1050, 1100],
            "Fine Aggregate": [700, 750, 800],
            "Age": [28, 28, 28],
            "Concrete Compressive Strength": [30, 35, 40],
        })
        mock_read_csv.return_value = mock_df

        # Initialize the dataset without normalization
        dataset = DatasetTorchConcreteCompressiveStrength(csv_path="fake_path.csv", normalize=False)

        # Ensure that the correct columns are selected as features and target
        self.assertEqual(dataset.input_cols, ["Cement", "Blast Furnace Slag", "Fly Ash", "Water", 
                                              "Superplasticizer", "Coarse Aggregate", "Fine Aggregate", "Age"])
        self.assertEqual(dataset.target_col, "Concrete Compressive Strength")

        # Check that normalization was not applied (values should be in original range)
        self.assertTrue(torch.all(dataset.X >= 0) and torch.all(dataset.X <= 7000))  # Check range of features
        self.assertTrue(torch.all(dataset.y >= 0) and torch.all(dataset.y <= 100))   # Check range of target

    @patch("pandas.read_csv")
    def test_len(self, mock_read_csv):
        # Mock the CSV data
        mock_df = pd.DataFrame({
            "Cement": [500, 600, 700],
            "Blast Furnace Slag": [100, 150, 200],
            "Fly Ash": [100, 200, 300],
            "Water": [160, 170, 180],
            "Superplasticizer": [10, 15, 20],
            "Coarse Aggregate": [1000, 1050, 1100],
            "Fine Aggregate": [700, 750, 800],
            "Age": [28, 28, 28],
            "Concrete Compressive Strength": [30, 35, 40],
        })
        mock_read_csv.return_value = mock_df

        # Initialize the dataset
        dataset = DatasetTorchConcreteCompressiveStrength(csv_path="fake_path.csv", normalize=True)

        # Ensure dataset length matches the number of samples in the CSV
        self.assertEqual(len(dataset), 3)

    @patch("pandas.read_csv")
    def test_get_item(self, mock_read_csv):
        # Mock the CSV data
        mock_df = pd.DataFrame({
            "Cement": [500, 600, 700],
            "Blast Furnace Slag": [100, 150, 200],
            "Fly Ash": [100, 200, 300],
            "Water": [160, 170, 180],
            "Superplasticizer": [10, 15, 20],
            "Coarse Aggregate": [1000, 1050, 1100],
            "Fine Aggregate": [700, 750, 800],
            "Age": [28, 28, 28],
            "Concrete Compressive Strength": [30, 35, 40],
        })
        mock_read_csv.return_value = mock_df

        # Initialize the dataset
        dataset = DatasetTorchConcreteCompressiveStrength(csv_path="fake_path.csv", normalize=True)

        # Test __getitem__
        image, label = dataset[0]

        # Ensure the image (features) and label are correct types
        self.assertIsInstance(image, torch.Tensor)
        self.assertEqual(label.item(), 30)  # First label should be 30

    @patch("pandas.read_csv")
    def test_missing_columns(self, mock_read_csv):
        # Mock the CSV data with missing required columns
        mock_df = pd.DataFrame({
            "Cement": [500, 600, 700],
            "Blast Furnace Slag": [100, 150, 200],
            "Fly Ash": [100, 200, 300],
            "Water": [160, 170, 180],
            "Superplasticizer": [10, 15, 20],
            "Coarse Aggregate": [1000, 1050, 1100],
            "Age": [28, 28, 28],
        })
        mock_read_csv.return_value = mock_df

        # Initialize the dataset and check for missing column error
        with self.assertRaises(KeyError):
            dataset = DatasetTorchConcreteCompressiveStrength(csv_path="fake_path.csv", normalize=True)


if __name__ == "__main__":
    unittest.main()
