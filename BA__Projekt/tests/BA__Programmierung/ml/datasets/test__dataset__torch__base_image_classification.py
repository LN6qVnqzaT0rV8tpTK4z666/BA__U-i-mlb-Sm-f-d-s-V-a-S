import unittest
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

# Assuming your dataset class is in the current module or imported as BaseImageCSVClassificationDataset
from BA__Programmierung.ml.datasets.dataset__torch__base_image_classification import BaseImageCSVClassificationDataset


class TestBaseImageCSVClassificationDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Setup test CSV data for testing
        cls.csv_data = {
            'label': [0, 1, 2],
            'px1': [255, 128, 64],
            'px2': [128, 64, 32],
            'px3': [64, 32, 16]
        }
        cls.csv_path = 'test_dataset.csv'
        cls.df = pd.DataFrame(cls.csv_data)
        cls.df.to_csv(cls.csv_path, index=False)

    @classmethod
    def tearDownClass(cls):
        # Cleanup the generated CSV file after tests
        if os.path.isfile(cls.csv_path):
            os.remove(cls.csv_path)

    def test_dataset_length(self):
        # Initialize the dataset
        dataset = BaseImageCSVClassificationDataset(
            csv_path=self.csv_path,
            image_shape=(3,)
        )

        # Assert the dataset length matches the number of entries in the CSV
        self.assertEqual(len(dataset), 3)

    def test_image_shape(self):
        # Initialize the dataset
        dataset = BaseImageCSVClassificationDataset(
            csv_path=self.csv_path,
            image_shape=(3,)
        )

        # Check if the image shape is as expected
        image, label = dataset[0]
        self.assertEqual(image.shape, torch.Size([3]))  # Image should have 3 values (px1, px2, px3)

    def test_normalization(self):
        # Initialize the dataset with normalization
        dataset = BaseImageCSVClassificationDataset(
            csv_path=self.csv_path,
            image_shape=(3,),
            normalize=True
        )

        # Check if the image values are normalized between 0 and 1
        image, _ = dataset[0]
        self.assertTrue((image <= 1).all() and (image >= 0).all())

    def test_transform(self):
        # Define a simple transformation
        transform = transforms.Lambda(lambda x: x * 2)

        # Initialize the dataset with the transform
        dataset = BaseImageCSVClassificationDataset(
            csv_path=self.csv_path,
            image_shape=(3,),
            transform=transform
        )

        # Check if the transform is applied correctly
        image, _ = dataset[0]
        original_image = torch.tensor([255, 128, 64], dtype=torch.float32) / 255.0
        self.assertTrue(torch.allclose(image, original_image * 2))

    def test_invalid_csv(self):
        # Test invalid CSV file path
        with self.assertRaises(FileNotFoundError):
            dataset = BaseImageCSVClassificationDataset(
                csv_path='invalid_path.csv',
                image_shape=(3,)
            )

    def test_invalid_image_shape(self):
        # Test image shape mismatch (e.g., expecting 3 but getting 4)
        with self.assertRaises(ValueError):
            dataset = BaseImageCSVClassificationDataset(
                csv_path=self.csv_path,
                image_shape=(4,)
            )


if __name__ == '__main__':
    unittest.main()
