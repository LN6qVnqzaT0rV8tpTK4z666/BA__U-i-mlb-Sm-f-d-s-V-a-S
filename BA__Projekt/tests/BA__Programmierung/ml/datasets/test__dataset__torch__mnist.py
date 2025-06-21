# BA__Projekt/tests/BA__Programmierung/ml/datasets/test__dataset__torch__mnist.py

import unittest

from BA__Programmierung.ml.datasets.dataset__torch__mnist import load_mnist_datasets
from BA__Programmierung.ml.datasets.dataset__torch__base_image_classification import BaseImageCSVClassificationDataset
from unittest.mock import patch, MagicMock


class TestMNISTDatasetLoading(unittest.TestCase):

    @patch("BA__Programmierung.ml.datasets.dataset__torch__base_image_classification.BaseImageCSVClassificationDataset")
    def test_load_mnist_datasets(self, mock_base_image_classification_dataset):
        # Mock the return value of BaseImageCSVClassificationDataset
        mock_train_ds = MagicMock()
        mock_test_ds = MagicMock()

        # Mock the return value when loading datasets
        mock_base_image_classification_dataset.return_value = mock_train_ds
        mock_base_image_classification_dataset.return_value = mock_test_ds

        # Define fake CSV paths
        train_csv_path = "fake_path/mnist-train.csv"
        test_csv_path = "fake_path/mnist-test.csv"

        # Call the function to load datasets
        train_ds, test_ds = load_mnist_datasets(train_csv_path, test_csv_path)

        # Verify that the correct paths and parameters were used when loading the datasets
        mock_base_image_classification_dataset.assert_any_call(
            train_csv_path, (1, 28, 28), normalize=True
        )
        mock_base_image_classification_dataset.assert_any_call(
            test_csv_path, (1, 28, 28), normalize=True
        )

        # Verify that the return value is correct
        self.assertEqual(train_ds, mock_train_ds)
        self.assertEqual(test_ds, mock_test_ds)

    @patch("BA__Programmierung.ml.datasets.dataset__torch__base_image_classification.BaseImageCSVClassificationDataset")
    def test_dataset_size(self, mock_base_image_classification_dataset):
        # Mock the return value of BaseImageCSVClassificationDataset
        mock_train_ds = MagicMock()
        mock_test_ds = MagicMock()
        mock_train_ds.__len__.return_value = 60000
        mock_test_ds.__len__.return_value = 10000

        # Mock the return value when loading datasets
        mock_base_image_classification_dataset.return_value = mock_train_ds
        mock_base_image_classification_dataset.return_value = mock_test_ds

        # Define fake CSV paths
        train_csv_path = "fake_path/mnist-train.csv"
        test_csv_path = "fake_path/mnist-test.csv"

        # Call the function to load datasets
        train_ds, test_ds = load_mnist_datasets(train_csv_path, test_csv_path)

        # Verify that the dataset sizes are correct
        self.assertEqual(len(train_ds), 60000)
        self.assertEqual(len(test_ds), 10000)

    @patch("BA__Programmierung.ml.datasets.dataset__torch__base_image_classification.BaseImageCSVClassificationDataset")
    def test_image_shape_and_normalization(self, mock_base_image_classification_dataset):
        # Mock the return value of BaseImageCSVClassificationDataset
        mock_train_ds = MagicMock()
        mock_test_ds = MagicMock()

        # Mock the return value when loading datasets
        mock_base_image_classification_dataset.return_value = mock_train_ds
        mock_base_image_classification_dataset.return_value = mock_test_ds

        # Define fake CSV paths
        train_csv_path = "fake_path/mnist-train.csv"
        test_csv_path = "fake_path/mnist-test.csv"

        # Call the function to load datasets
        train_ds, test_ds = load_mnist_datasets(train_csv_path, test_csv_path)

        # Ensure that the image shape and normalization are passed correctly
        mock_base_image_classification_dataset.assert_any_call(
            train_csv_path, (1, 28, 28), normalize=True
        )
        mock_base_image_classification_dataset.assert_any_call(
            test_csv_path, (1, 28, 28), normalize=True
        )

    @patch("BA__Programmierung.ml.datasets.dataset__torch__base_image_classification.BaseImageCSVClassificationDataset")
    def test_invalid_csv_path(self, mock_base_image_classification_dataset):
        # Mock the behavior of loading datasets with invalid CSV paths
        mock_base_image_classification_dataset.side_effect = FileNotFoundError("CSV file not found")

        # Call the function with a non-existent file path and verify FileNotFoundError is raised
        with self.assertRaises(FileNotFoundError):
            load_mnist_datasets("invalid_path/mnist-train.csv", "invalid_path/mnist-test.csv")

    @patch("BA__Programmierung.ml.datasets.dataset__torch__base_image_classification.BaseImageCSVClassificationDataset")
    def test_return_type(self, mock_base_image_classification_dataset):
        # Mock the return value of BaseImageCSVClassificationDataset
        mock_train_ds = MagicMock()
        mock_test_ds = MagicMock()

        # Mock the return value when loading datasets
        mock_base_image_classification_dataset.return_value = mock_train_ds
        mock_base_image_classification_dataset.return_value = mock_test_ds

        # Define fake CSV paths
        train_csv_path = "fake_path/mnist-train.csv"
        test_csv_path = "fake_path/mnist-test.csv"

        # Call the function to load datasets
        train_ds, test_ds = load_mnist_datasets(train_csv_path, test_csv_path)

        # Verify that the returned datasets are of the correct type
        self.assertIsInstance(train_ds, BaseImageCSVClassificationDataset)
        self.assertIsInstance(test_ds, BaseImageCSVClassificationDataset)


if __name__ == "__main__":
    unittest.main()

