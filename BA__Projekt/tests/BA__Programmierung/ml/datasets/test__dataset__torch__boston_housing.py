import unittest
import pandas as pd
import torch
from BA__Programmierung.ml.datasets.dataset__torch__boston_housing import DatasetTorchBostonHousing


class TestDatasetTorchBostonHousing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Setup a small sample CSV data for testing
        cls.data = {
            "CRIM": [0.00632, 0.02731, 0.02729],
            "ZN": [18.0, 0.0, 0.0],
            "INDUS": [2.31, 7.87, 7.87],
            "CHAS": [0, 0, 0],
            "NOX": [0.538, 0.469, 0.469],
            "RM": [6.575, 6.421, 7.185],
            "AGE": [65.2, 78.9, 61.1],
            "DIS": [4.09, 4.97, 4.09],
            "RAD": [1, 2, 2],
            "TAX": [296.0, 242.0, 242.0],
            "PTRATIO": [15.3, 17.8, 17.8],
            "B": [396.90, 392.83, 394.63],
            "LSTAT": [4.98, 9.14, 4.03],
            "MEDV": [24.0, 21.6, 34.7],
        }
        cls.df = pd.DataFrame(cls.data)
        cls.df.to_csv("boston_housing_test.csv", header=False, index=False, sep=" ")

    @classmethod
    def tearDownClass(cls):
        # Cleanup generated test CSV file
        if os.path.isfile("boston_housing_test.csv"):
            os.remove("boston_housing_test.csv")

    def test_dataset_length(self):
        # Initialize the dataset
        dataset = DatasetTorchBostonHousing(csv_path="boston_housing_test.csv")
        
        # Check if the dataset length matches the number of rows in the CSV
        self.assertEqual(len(dataset), len(self.df))

    def test_normalization(self):
        # Initialize the dataset with normalization
        dataset = DatasetTorchBostonHousing(csv_path="boston_housing_test.csv")
        
        # Check if the features are normalized (mean should be close to 0 and std close to 1)
        feature_mean = dataset.X.mean(dim=0).numpy()
        feature_std = dataset.X.std(dim=0).numpy()

        self.assertTrue(abs(feature_mean[0]) < 1e-2 and abs(feature_mean[1]) < 1e-2)
        self.assertTrue(abs(feature_std[0] - 1) < 1e-2 and abs(feature_std[1] - 1) < 1e-2)

    def test_target_column(self):
        # Initialize the dataset
        dataset = DatasetTorchBostonHousing(csv_path="boston_housing_test.csv")
        
        # Ensure the target column is 'MEDV' and it's properly included in the dataset
        target_column = dataset.y.numpy()
        self.assertTrue((target_column == self.df["MEDV"].values).all())

    def test_data_integrity(self):
        # Initialize the dataset
        dataset = DatasetTorchBostonHousing(csv_path="boston_housing_test.csv")
        
        # Test that the data features and target are correctly mapped
        X, y = dataset[0]  # Get the first row
        self.assertEqual(X.shape, torch.Size([13]))  # There are 13 feature columns
        self.assertEqual(y.shape, torch.Size([]))    # Target is a scalar

    def test_classification_false(self):
        # Initialize the dataset with classification set to False
        dataset = DatasetTorchBostonHousing(csv_path="boston_housing_test.csv")
        
        # Verify that the dataset is set up for regression (not classification)
        self.assertFalse(dataset.classification)

if __name__ == '__main__':
    unittest.main()
