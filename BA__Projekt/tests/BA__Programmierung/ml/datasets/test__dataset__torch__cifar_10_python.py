# BA__Projekt/tests/BA__Programmierung/ml/datasets/test__dataset__torch__cifar_10_python.py
import unittest
from unittest.mock import patch, MagicMock
import torch
import numpy as np
from torchvision import transforms
from BA__Programmierung.ml.datasets.dataset__torch__base_image_classification import BaseImageCSVClassificationDataset
from BA__Programmierung.ml.datasets.dataset__torch__cifar_10_python import DatasetTorchCIFAR10AllBatches


class TestBaseImageNDArrayDataset(unittest.TestCase):

    def setUp(self):
        # Create mock images and labels
        self.images = np.random.randint(0, 256, (10, 3, 32, 32), dtype=np.uint8)  # 10 images, 3 channels, 32x32
        self.labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        
    def test_initialize_dataset_with_normalization(self):
        # Initialize dataset with normalization
        dataset = BaseImageCSVClassificationDataset(self.images, self.labels, normalize=True)

        # Ensure images are normalized
        self.assertTrue(torch.all(dataset.images <= 1.0))  # All pixel values should be between 0 and 1
        self.assertTrue(torch.all(dataset.images >= 0.0))  # All pixel values should be >= 0

        # Ensure labels are the correct type
        self.assertTrue(dataset.labels.dtype == torch.long)

    def test_initialize_dataset_without_normalization(self):
        # Initialize dataset without normalization
        dataset = BaseImageCSVClassificationDataset(self.images, self.labels, normalize=False)

        # Ensure images are not normalized (should remain in the range [0, 255])
        self.assertTrue(torch.all(dataset.images <= 255))
        self.assertTrue(torch.all(dataset.images >= 0))

    def test_len(self):
        # Initialize dataset with normalization
        dataset = BaseImageCSVClassificationDataset(self.images, self.labels, normalize=True)

        # Ensure dataset length matches number of samples
        self.assertEqual(len(dataset), len(self.labels))

    def test_get_item(self):
        # Initialize dataset
        dataset = BaseImageCSVClassificationDataset(self.images, self.labels, normalize=True)

        # Test __getitem__
        image, label = dataset[0]

        # Ensure the image and label are correct types
        self.assertIsInstance(image, torch.Tensor)
        self.assertEqual(label, self.labels[0])

    def test_transform(self):
        # Define a simple transformation (e.g., convert image to grayscale)
        transform = transforms.Grayscale(num_output_channels=3)

        # Initialize dataset with transformation
        dataset = BaseImageCSVClassificationDataset(self.images, self.labels, transform=transform)

        # Apply transformation to the first image
        transformed_image, _ = dataset[0]

        # Ensure the transformed image has 3 channels (since Grayscale converts to 3 channels)
        self.assertEqual(transformed_image.shape[0], 3)


class TestDatasetTorchCIFAR10AllBatches(unittest.TestCase):

    @patch("builtins.open", new_callable=MagicMock)
    @patch("pickle.load")
    def test_dataset_loading(self, mock_pickle_load, mock_open):
        # Mock CIFAR-10 batch file
        mock_batch = {
            "data": np.random.randint(0, 256, (10000, 3 * 32 * 32), dtype=np.uint8),
            "labels": list(range(10000))
        }
        mock_pickle_load.return_value = mock_batch

        # List of mock batch file paths
        batch_paths = ["path/to/batch1", "path/to/batch2"]

        # Initialize CIFAR-10 dataset
        dataset = DatasetTorchCIFAR10AllBatches(batch_paths)

        # Verify that the pickle files are loaded and images are concatenated
        mock_open.assert_any_call("path/to/batch1", "rb")
        mock_open.assert_any_call("path/to/batch2", "rb")
        
        # Verify that images are concatenated correctly
        self.assertEqual(dataset.images.shape[0], 20000)  # 10000 samples per batch, 2 batches
        self.assertEqual(dataset.labels.shape[0], 20000)

    def test_empty_batch_paths(self):
        # Test DatasetTorchCIFAR10AllBatches with an empty batch path list
        dataset = DatasetTorchCIFAR10AllBatches([])

        # Ensure that dataset is empty
        self.assertEqual(len(dataset), 0)

    @patch("builtins.open", new_callable=MagicMock)
    @patch("pickle.load")
    def test_missing_key_in_batch(self, mock_pickle_load, mock_open):
        # Simulate a pickle batch with missing 'data' key
        mock_batch = {
            "labels": list(range(10000))
        }
        mock_pickle_load.return_value = mock_batch

        # List of mock batch file paths
        batch_paths = ["path/to/batch1"]

        # Initialize CIFAR-10 dataset and check for missing key
        with self.assertRaises(KeyError):
            dataset = DatasetTorchCIFAR10AllBatches(batch_paths)


if __name__ == "__main__":
    unittest.main()

