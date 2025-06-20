import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import torch
from BA__Programmierung.ml.datasets.dataset__torch__wine_quality_red import WineQualityRedDataset


class TestWineQualityRedDataset(unittest.TestCase):

    @patch("os.path.isfile")
    @patch("pandas.read_csv")
    def test_initialize_dataset(self, mock_read_csv, mock_isfile):
        # Mocking the check for the file existence
        mock_isfile.return_value = True

        # Mock the CSV data (semicolon-separated)
        mock_df = pd.DataFrame({
            "fixed acidity": [7.4, 7.8, 7.8],
            "volatile acidity": [0.7, 0.88, 0.76],
            "citric acid": [0.0, 0.0, 0.04],
            "residual sugar": [1.9, 2.6, 2.3],
            "chlorides": [0.076, 0.098, 0.092],
            "free sulfur dioxide": [11, 25, 15],
            "total sulfur dioxide": [34, 67, 54],
            "density": [0.9978, 0.9968, 0.9970],
            "pH": [3.51, 3.20, 3.26],
            "sulphates": [0.56, 0.68, 0.65],
            "alcohol": [9.4, 9.8, 9.8],
            "quality": [5, 5, 5]
        })
        mock_read_csv.return_value = mock_df

        # Initialize the dataset
        csv_path = "fake_path/winequality-red.csv"
        dataset = WineQualityRedDataset(csv_path)

        # Ensure that the correct columns are selected as features and target
        self.assertEqual(dataset.input_cols, ["fixed acidity", "volatile acidity", "citric acid", 
                                              "residual sugar", "chlorides", "free sulfur dioxide", 
                                              "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"])
        self.assertEqual(dataset.target_col, "quality")

        # Verify that no normalization was applied
        self.assertFalse(dataset.normalize)

        # Verify that the dataset size matches the number of rows in the CSV
        self.assertEqual(len(dataset), 3)

    @patch("os.path.isfile")
    def test_file_not_found(self, mock_isfile):
        # Mocking the check for the file existence to return False
        mock_isfile.return_value = False

        # Call the function with a non-existent file path and verify FileNotFoundError is raised
        with self.assertRaises(FileNotFoundError):
            WineQualityRedDataset("invalid_path/winequality-red.csv")

    @patch("os.path.isfile")
    @patch("pandas.read_csv")
    def test_column_assignment(self, mock_read_csv, mock_isfile):
        # Mocking the check for the file existence to return True
        mock_isfile.return_value = True

        # Mock the CSV data (semicolon-separated)
        mock_df = pd.DataFrame({
            "fixed acidity": [7.4, 7.8, 7.8],
            "volatile acidity": [0.7, 0.88, 0.76],
            "citric acid": [0.0, 0.0, 0.04],
            "residual sugar": [1.9, 2.6, 2.3],
            "chlorides": [0.076, 0.098, 0.092],
            "free sulfur dioxide": [11, 25, 15],
            "total sulfur dioxide": [34, 67, 54],
            "density": [0.9978, 0.9968, 0.9970],
            "pH": [3.51, 3.20, 3.26],
            "sulphates": [0.56, 0.68, 0.65],
            "alcohol": [9.4, 9.8, 9.8],
            "quality": [5, 5, 5]
        })
        mock_read_csv.return_value = mock_df

        # Initialize the dataset
        csv_path = "fake_path/winequality-red.csv"
        dataset = WineQualityRedDataset(csv_path)

        # Ensure that the last column is the target and the rest are features
        self.assertEqual(dataset.input_cols, ["fixed acidity", "volatile acidity", "citric acid", 
                                              "residual sugar", "chlorides", "free sulfur dioxide", 
                                              "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"])
        self.assertEqual(dataset.target_col, "quality")

    @patch("os.path.isfile")
    @patch("pandas.read_csv")
    def test_dataset_size(self, mock_read_csv, mock_isfile):
        # Mocking the check for the file existence to return True
        mock_isfile.return_value = True

        # Mock the CSV data (semicolon-separated)
        mock_df = pd.DataFrame({
            "fixed acidity": [7.4, 7.8, 7.8],
            "volatile acidity": [0.7, 0.88, 0.76],
            "citric acid": [0.0, 0.0, 0.04],
            "residual sugar": [1.9, 2.6, 2.3],
            "chlorides": [0.076, 0.098, 0.092],
            "free sulfur dioxide": [11, 25, 15],
            "total sulfur dioxide": [34, 67, 54],
            "density": [0.9978, 0.9968, 0.9970],
            "pH": [3.51, 3.20, 3.26],
            "sulphates": [0.56, 0.68, 0.65],
            "alcohol": [9.4, 9.8, 9.8],
            "quality": [5, 5, 5]
        })
        mock_read_csv.return_value = mock_df

        # Initialize the dataset
        csv_path = "fake_path/winequality-red.csv"
        dataset = WineQualityRedDataset(csv_path)

        # Verify that the dataset size matches the number of rows in the CSV
        self.assertEqual(len(dataset), 3)

    @patch("os.path.isfile")
    @patch("pandas.read_csv")
    def test_normalization(self, mock_read_csv, mock_isfile):
        # Mocking the check for the file existence to return True
        mock_isfile.return_value = True

        # Mock the CSV data (semicolon-separated)
        mock_df = pd.DataFrame({
            "fixed acidity": [7.4, 7.8, 7.8],
            "volatile acidity": [0.7, 0.88, 0.76],
            "citric acid": [0.0, 0.0, 0.04],
            "residual sugar": [1.9, 2.6, 2.3],
            "chlorides": [0.076, 0.098, 0.092],
            "free sulfur dioxide": [11, 25, 15],
            "total sulfur dioxide": [34, 67, 54],
            "density": [0.9978, 0.9968, 0.9970],
            "pH": [3.51, 3.20, 3.26],
            "sulphates": [0.56, 0.68, 0.65],
            "alcohol": [9.4, 9.8, 9.8],
            "quality": [5, 5, 5]
        })
        mock_read_csv.return_value = mock_df

        # Initialize the dataset with normalization
        csv_path = "fake_path/winequality-red.csv"
        dataset = WineQualityRedDataset(csv_path)

        # Ensure that no normalization was applied (since the default is False)
        self.assertFalse(dataset.normalize)


if __name__ == "__main__":
    unittest.main()
