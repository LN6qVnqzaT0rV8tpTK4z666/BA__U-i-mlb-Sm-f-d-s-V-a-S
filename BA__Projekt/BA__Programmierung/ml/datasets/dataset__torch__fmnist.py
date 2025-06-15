# BA__Projekt/BA__Programmierung/ml/datasets/dataset__torch__fmnist.py


from BA__Programmierung.ml.datasets.dataset__torch__base_image_classification import (
    BaseImageCSVClassificationDataset,
)

def load_fashion_mnist_datasets(root_path, transform=None):

    train_ds = BaseImageCSVClassificationDataset(
        "assets/data/raw/dataset__fmnist/fashion-mnist_training.csv", image_shape=(1, 28, 28), transform=None, normalize=True
    )
    test_ds = BaseImageCSVClassificationDataset(
        "assets/data/raw/dataset__fmnist/fashion-mnist_test.csv", image_shape=(1, 28, 28), transform=None, normalize=True
    )

    return train_ds, test_ds


if __name__ == "__main__":

    train_ds, test_ds = load_fashion_mnist_datasets(None, None)

    print(f"Train size: {len(train_ds)}, Test size: {len(test_ds)}")
    img, label = train_ds[0]
    print(f"Image shape: {img.shape}, Label: {label}")
