# BA__Projekt/BA__Programmierung/ml/datasets/dataset__torch__mnist.py

from BA__Programmierung.ml.datasets.dataset__torch__base_image_classification import (
    BaseImageCSVClassificationDataset,
)


def load_mnist_datasets(train_csv_path, test_csv_path):
    """
    Load MNIST train and test datasets from CSV files.

    Parameters
    ----------
    train_csv_path : str
        Path to the MNIST training CSV file.
    test_csv_path : str
        Path to the MNIST test CSV file.

    Returns
    -------
    tuple of (BaseImageCSVClassificationDataset, BaseImageCSVClassificationDataset)
        Tuple containing the training dataset and the test dataset.

    Notes
    -----
    Assumes grayscale images with shape (1, 28, 28) and pixel values normalized to [0,1].

    Example
    -------
    >>> train_ds, test_ds = load_mnist_datasets("mnist-train.csv", "mnist-test.csv")
    """
    image_shape = (1, 28, 28)  # channel-first

    train_dataset = BaseImageCSVClassificationDataset(
        train_csv_path, image_shape, normalize=True
    )
    test_dataset = BaseImageCSVClassificationDataset(
        test_csv_path, image_shape, normalize=True
    )

    return train_dataset, test_dataset


if __name__ == "__main__":
    train_path = "/root/BA__Projekt/assets/data/raw/dataset__mnist/mnist-train.csv"
    test_path = "/root/BA__Projekt/assets/data/raw/dataset__mnist/mnist-test.csv"

    train_ds, test_ds = load_mnist_datasets(train_path, test_path)
    print(f"Train: {len(train_ds)} samples, Test: {len(test_ds)} samples")
    img, label = train_ds[0]
    print(f"Image shape: {img.shape}, 🔢 Label: {label}")

