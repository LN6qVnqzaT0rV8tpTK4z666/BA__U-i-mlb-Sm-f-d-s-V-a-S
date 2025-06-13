# BA__Projekt/BA__Programmierung/ml/datasets/dataset__cifar-10-python.py

import pickle

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class BaseImageNDArrayDataset(Dataset):
    """
    PyTorch Dataset for image data stored as NumPy arrays with corresponding labels.

    Supports optional normalization and image transformations.

    Parameters
    ----------
    images : numpy.ndarray
        Array of images with shape (N, C, H, W) or more general (N, ...).
    labels : list or numpy.ndarray
        List or array containing labels for each image.
    transform : callable, optional
        Transformation function or torchvision.transforms to apply on images.
    normalize : bool, default=True
        If True, scale image pixel values from [0, 255] to [0, 1].

    Attributes
    ----------
    images : torch.Tensor
        Tensor containing image data, dtype float32.
    labels : torch.Tensor
        Tensor containing labels as long integers.
    """

    def __init__(self, images, labels, transform=None, normalize=True):
        """
        Initialize the dataset with images and labels.

        Parameters
        ----------
        images : numpy.ndarray
            Array of image data.
        labels : list or numpy.ndarray
            Corresponding labels.
        transform : callable, optional
            Transformations to apply on images.
        normalize : bool, default=True
            Whether to scale pixel values to [0, 1].
        """
        self.transform = transform
        self.labels = torch.tensor(labels, dtype=torch.long)

        if normalize:
            self.images = images.astype(np.float32) / 255.0
        else:
            self.images = images.astype(np.float32)

        # Convert to tensor
        self.images = torch.tensor(self.images, dtype=torch.float32)

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
        Retrieve the image and label at the given index, applying transforms if specified.

        Parameters
        ----------
        idx : int
            Index of the sample to retrieve.

        Returns
        -------
        tuple of (torch.Tensor or PIL.Image, torch.Tensor)
            Transformed image and its label.
        """
        img = self.images[idx]

        if self.transform:
            # If transform expects PIL Images, convert tensor to PIL first
            if isinstance(self.transform, transforms.Compose) or hasattr(self.transform, '__call__'):
                img = transforms.ToPILImage()(img)
                img = self.transform(img)
            else:
                img = self.transform(img)

        label = self.labels[idx]
        return img, label


class DatasetTorchCIFAR10AllBatches(BaseImageNDArrayDataset):
    """
    PyTorch Dataset that loads and combines all CIFAR-10 batches from pickle files.

    Parameters
    ----------
    batch_paths : list of str
        List of file paths to CIFAR-10 batch files (pickle format).
    transform : callable, optional
        Transformation function or torchvision.transforms to apply on images.

    Attributes
    ----------
    images : torch.Tensor
        Tensor of all images concatenated from the batches.
    labels : torch.Tensor
        Tensor of all labels concatenated from the batches.
    """

    def __init__(self, batch_paths, transform=None):
        """
        Initialize the CIFAR-10 dataset by loading and concatenating all batches.

        Parameters
        ----------
        batch_paths : list of str
            Paths to CIFAR-10 batch files.
        transform : callable, optional
            Transformations to apply on images.
        """
        images = []
        labels = []

        for path in batch_paths:
            with open(path, "rb") as f:
                batch = pickle.load(f, encoding="latin1")
                data = batch["data"].reshape(-1, 3, 32, 32).astype(np.uint8)
                images.append(data)
                labels.extend(batch["labels"])

        images = np.concatenate(images, axis=0)
        super().__init__(images, labels, transform=transform, normalize=True)
