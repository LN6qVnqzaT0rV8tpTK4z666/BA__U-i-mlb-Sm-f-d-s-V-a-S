# BA__Projekt/BA__Programmierung/ml/datasets/dataset__torch__fmnist.py

import os

from BA__Programmierung.ml.datasets.dataset__torch__base_image_classification import (
    BaseImageCSVClassificationDataset,
)


def load_fashion_mnist_datasets(root_path, transform=None):
    """
    Load Fashion-MNIST train and test datasets from CSV files using BaseImageCSVClassificationDataset.

    Parameters
    ----------
    root_path : str
        Root directory path containing the Fashion-MNIST CSV files.
    transform : callable or None, optional
        Optional transform to be applied on a sample (image). Default is None.

    Returns
    -------
    tuple of (Dataset, Dataset)
        Tuple containing the training dataset and test dataset.

    Example
    -------
    >>> train_ds, test_ds = load_fashion_mnist_datasets("/path/to/fashion-mnist")
    """
    train_csv = os.path.join(root_path, "fashion-mnist_train.csv")
    test_csv = os.path.join(root_path, "fashion-mnist_test.csv")

    train_dataset = BaseImageCSVClassificationDataset(
        train_csv, image_shape=(1, 28, 28), transform=transform, normalize=True
    )
    test_dataset = BaseImageCSVClassificationDataset(
        test_csv, image_shape=(1, 28, 28), transform=transform, normalize=True
    )

    return train_dataset, test_dataset


if __name__ == "__main__":
    dataset_root = "/root/BA__U-i-mlb-Sm-f-d-s-V-a-S/BA__Projekt/assets/data/raw/dataset__fmnist"
    train_ds, test_ds = load_fashion_mnist_datasets(dataset_root)

    print(f"Train size: {len(train_ds)}, Test size: {len(test_ds)}")
    img, label = train_ds[0]
    print(f"Image shape: {img.shape}, Label: {label}")
