import unittest
from unittest.mock import patch, MagicMock
from BA__Programmierung.ml.datasets.dataset__torch__fmnist import load_fashion_mnist_datasets
from BA__Programmierung.ml.datasets.dataset__torch__base_image_classification import BaseImageCSVClassificationDataset

class TestFashionMNISTLoading(unittest.TestCase):

    @patch("BA__Programmierung.ml.datasets.dataset__torch__base_image_classification.BaseImageCSVClassificationDataset")
    def test_load_fashion_mnist_datasets(self, mock_base_image_classification_dataset):
        # Mock the behavior of BaseImageCSVClassificationDataset
        mock_train_ds = MagicMock()
        mock_test_ds = MagicMock()

        # Mock the return value of loading datasets
        mock_base_image_classification_dataset.return_value = mock_train_ds
        mock_base_image_classification_dataset.return_value = mock_test_ds

        # Call the function to load datasets
        root_path = "fake_path"
        train_ds, test_ds = load_fashion_mnist_datasets(root_path)

        # Ensure that the datasets are loaded with the correct paths
        mock_base_image_classification_dataset.assert_any_call(
            "assets/data/raw/dataset__fmnist/fashion-mnist_training.csv", 
            image_shape=(1, 28, 28), transform=None, normalize=True
        )
        mock_base_image_classification_dataset.assert_any_call(
            "assets/data/raw/dataset__fmnist/fashion-mnist_test.csv", 
            image_shape=(1, 28, 28), transform=None, normalize=True
        )

        # Verify that the return value is as expected
        self.assertEqual(train_ds, mock_train_ds)
        self.assertEqual(test_ds, mock_test_ds)

    @patch("BA__Programmierung.ml.datasets.dataset__torch__base_image_classification.BaseImageCSVClassificationDataset")
    def test_dataset_size(self, mock_base_image_classification_dataset):
        # Mock the behavior of BaseImageCSVClassificationDataset to return specific lengths
        mock_train_ds = MagicMock()
        mock_test_ds = MagicMock()
        mock_train_ds.__len__.return_value = 60000
        mock_test_ds.__len__.return_value = 10000

        # Mock the return value of loading datasets
        mock_base_image_classification_dataset.return_value = mock_train_ds
        mock_base_image_classification_dataset.return_value = mock_test_ds

        # Call the function to load datasets
        root_path = "fake_path"
        train_ds, test_ds = load_fashion_mnist_datasets(root_path)

        # Verify that the dataset sizes are as expected
        self.assertEqual(len(train_ds), 60000)
        self.assertEqual(len(test_ds), 10000)

    @patch("BA__Programmierung.ml.datasets.dataset__torch__base_image_classification.BaseImageCSVClassificationDataset")
    def test_image_shape_and_normalization(self, mock_base_image_classification_dataset):
        # Mock the behavior of BaseImageCSVClassificationDataset
        mock_train_ds = MagicMock()
        mock_test_ds = MagicMock()

        # Mock the return value of loading datasets
        mock_base_image_classification_dataset.return_value = mock_train_ds
        mock_base_image_classification_dataset.return_value = mock_test_ds

        # Call the function to load datasets
        root_path = "fake_path"
        train_ds, test_ds = load_fashion_mnist_datasets(root_path)

        # Ensure that the image shape and normalization are set correctly
        mock_base_image_classification_dataset.assert_any_call(
            "assets/data/raw/dataset__fmnist/fashion-mnist_training.csv", 
            image_shape=(1, 28, 28), transform=None, normalize=True
        )
        mock_base_image_classification_dataset.assert_any_call(
            "assets/data/raw/dataset__fmnist/fashion-mnist_test.csv", 
            image_shape=(1, 28, 28), transform=None, normalize=True
        )

    @patch("BA__Programmierung.ml.datasets.dataset__torch__base_image_classification.BaseImageCSVClassificationDataset")
    def test_invalid_csv_path(self, mock_base_image_classification_dataset):
        # Mock the behavior of loading datasets with invalid CSV paths
        mock_base_image_classification_dataset.side_effect = FileNotFoundError("CSV file not found")

        # Call the function to load datasets and check for the expected exception
        with self.assertRaises(FileNotFoundError):
            load_fashion_mnist_datasets("invalid_path")

if __name__ == "__main__":
    unittest.main()
