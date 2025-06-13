# BA__Projekt/BA__Programmierung/ml/datasets/dataset__torch__base_image_classification.py

import os

import pandas as pd
import torch
from torch.utils.data import Dataset


class BaseImageCSVClassificationDataset(Dataset):
    """
    Dataset for image classification tasks based on CSV files containing labels and flattened image pixels.

    The CSV file is expected to have the first column as labels and the subsequent columns as flattened pixel values
    for each image.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file. The CSV must contain one label column followed by flattened pixel columns.
    image_shape : tuple of int
        Shape of each image, e.g., (3, 32, 32) for RGB images, (1, 28, 28) for grayscale, or (10,) for 1D vectors.
    transform : callable, optional
        Optional transformation function (e.g., from torchvision.transforms) applied to each image.
    normalize : bool, default=True
        If True, pixel values are scaled from [0, 255] to [0, 1].

    Raises
    ------
    FileNotFoundError
        If the CSV file at `csv_path` does not exist.
    ValueError
        If the number of pixel columns in the CSV does not match the product of `image_shape`.

    Examples
    --------
    CSV format:
        label, px1, px2, ..., pxN

    Example usage:
        dataset = BaseImageCSVClassificationDataset(
            csv_path='data/train.csv',
            image_shape=(3, 32, 32),
            transform=my_transforms,
            normalize=True
        )
    """

    def __init__(self, csv_path, image_shape, transform=None, normalize=True):
        """
        Initialize the dataset by loading the CSV file and preprocessing images and labels.

        Parameters
        ----------
        csv_path : str
            Path to the CSV file containing the dataset.
        image_shape : tuple of int
            Expected shape of each image.
        transform : callable, optional
            Transformation to apply to each image.
        normalize : bool, default=True
            Whether to scale pixel values to [0, 1].
        
        Raises
        ------
        FileNotFoundError
            If the CSV file does not exist.
        ValueError
            If pixel count does not match image_shape.
        """
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        df = pd.read_csv(csv_path)

        # Labels as LongTensor
        self.labels = torch.tensor(df.iloc[:, 0].values, dtype=torch.long)

        # Flattened pixel values
        images = df.iloc[:, 1:].values.astype("float32")

        # Check if pixel count matches image_shape product
        expected_num_pixels = 1
        for dim in image_shape:
            expected_num_pixels *= dim
        if images.shape[1] != expected_num_pixels:
            raise ValueError(
                f"Pixel count in CSV ({images.shape[1]}) does not match product of image_shape ({expected_num_pixels})"
            )

        # Reshape images to nD
        self.images = images.reshape(-1, *image_shape)

        if normalize:
            self.images /= 255.0

        # Convert to tensor
        self.images = torch.tensor(self.images, dtype=torch.float32)

        self.transform = transform

    def __len__(self):
        """
        Returns
        -------
        int
            Number of samples in the dataset.
        """
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Retrieve image and label at index `idx`.

        Parameters
        ----------
        idx : int
            Index of the sample to retrieve.

        Returns
        -------
        tuple of (torch.Tensor, torch.Tensor)
            Image tensor and corresponding label tensor.
        """
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
