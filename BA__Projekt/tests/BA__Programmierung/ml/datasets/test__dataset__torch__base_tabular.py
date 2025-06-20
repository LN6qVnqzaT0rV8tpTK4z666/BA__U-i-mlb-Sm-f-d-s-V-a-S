import unittest
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from BA__Programmierung.ml.datasets.dataset__torch__base_tabular import BaseTabularDataset


class TestBaseTabularDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Setup sample DataFrame for testing
        cls.data = {
            'feature1': [1.0, 2.0, 3.0, 4.0],
            'feature2': [5.0, 6.0, 7.0, 8.0],
            'target': [0.1, 0.2, 0.3, 0.4]
        }
        cls.df = pd.DataFrame(cls.data)

    def test_dataset_length(self):
        # Initialize the dataset for regression
        dataset = BaseTabularDataset(
            dataframe=self.df,
            input_cols=['feature1', 'feature2'],
            target_cols='target',
            classification=False
        )
        # Check if the dataset length matches the number of rows in the DataFrame
        self.assertEqual(len(dataset), 4)

    def test_feature_normalization(self):
        # Initialize dataset with normalization for regression
        dataset = BaseTabularDataset(
            dataframe=self.df,
            input_cols=['feature1', 'feature2'],
            target_cols='target',
            normalize=True,
            classification=False
        )
        
        # Check if the features are normalized (mean should be close to 0 and std close to 1)
        feature_mean = dataset.X.mean(dim=0).numpy()
        feature_std = dataset.X.std(dim=0).numpy()

        self.assertTrue(abs(feature_mean[0]) < 1e-2 and abs(feature_mean[1]) < 1e-2)
        self.assertTrue(abs(feature_std[0] - 1) < 1e-2 and abs(feature_std[1] - 1) < 1e-2)

    def test_classification_labels(self):
        # Initialize dataset for classification
        dataset = BaseTabularDataset(
            dataframe=self.df,
            input_cols=['feature1', 'feature2'],
            target_cols='target',
            classification=True
        )
        
        # Ensure the labels are stored as LongTensor
        self.assertEqual(dataset.y.dtype, torch.long)

    def test_regression_labels(self):
        # Initialize dataset for regression
        dataset = BaseTabularDataset(
            dataframe=self.df,
            input_cols=['feature1', 'feature2'],
            target_cols='target',
            classification=False
        )
        
        # Ensure the labels are stored as FloatTensor for regression
        self.assertEqual(dataset.y.dtype, torch.float32)

    def test_inverse_transform(self):
        # Initialize dataset for regression with normalization
        dataset = BaseTabularDataset(
            dataframe=self.df,
            input_cols=['feature1', 'feature2'],
            target_cols='target',
            normalize=True,
            classification=False
        )

        # Make a dummy prediction (just use the normalized target value)
        y_pred = torch.tensor([0.2, 0.3, 0.4, 0.5], dtype=torch.float32)

        # Inverse transform the prediction to the original scale
        y_pred_inv = dataset.inverse_transform_y(y_pred)

        # Check if the inverse transformation returns a tensor
        self.assertEqual(y_pred_inv.dtype, torch.float32)

        # Check if the inverse transformed predictions are in the correct scale
        original_values = [0.2, 0.3, 0.4, 0.5]
        self.assertTrue(torch.allclose(y_pred_inv, torch.tensor(original_values, dtype=torch.float32)))

    def test_get_item(self):
        # Initialize dataset for regression
        dataset = BaseTabularDataset(
            dataframe=self.df,
            input_cols=['feature1', 'feature2'],
            target_cols='target',
            classification=False
        )
        
        # Test if the __getitem__ method works as expected
        x, y = dataset[0]
        self.assertEqual(x.shape, torch.Size([2]))  # Feature vector should have 2 elements
        self.assertEqual(y.shape, torch.Size([]))   # Scalar target for regression

    def test_invalid_target_cols(self):
        # Test invalid target column name
        with self.assertRaises(KeyError):
            dataset = BaseTabularDataset(
                dataframe=self.df,
                input_cols=['feature1', 'feature2'],
                target_cols='invalid_target',
                classification=False
            )


if __name__ == '__main__':
    unittest.main()
