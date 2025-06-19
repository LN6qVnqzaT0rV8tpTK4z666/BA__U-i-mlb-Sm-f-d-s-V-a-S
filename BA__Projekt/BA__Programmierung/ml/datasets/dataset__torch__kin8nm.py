# BA__Projekt/BA__Programmierung/ml/datasets/dataset__torch__kin8nm.py

import os

from BA__Programmierung.ml.datasets.dataset__torch__base_tabular import (
    BaseTabularDataset,
)


def load_kin8nm_dataset(csv_path):
    """
    Load the Kin8nm dataset from a CSV file into a PyTorch BaseTabularDataset.

    Parameters
    ----------
    csv_path : str
        Path to the Kin8nm CSV dataset file.

    Returns
    -------
    BaseTabularDataset
        PyTorch dataset containing the Kin8nm data with normalized features.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist at the given path.

    Example
    -------
    >>> dataset = load_kin8nm_dataset("path/to/kin8nm.csv")
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    import pandas as pd
    df = pd.read_csv(csv_path)

    input_cols = df.columns[:-1].tolist()
    target_cols = df.columns[-1]

    dataset = BaseTabularDataset(df, input_cols, target_cols, normalize=True, classification=False)
    return dataset


if __name__ == "__main__":
    dataset_path = "/root/BA__U-i-mlb-Sm-f-d-s-V-a-S/BA__Projekt/assets/data/raw/dataset__kin8nm/kin8nm.csv"
    dataset = load_kin8nm_dataset(dataset_path)

    print(f"Dataset size: {len(dataset)}")
    x, y = dataset[0]
    print(f"Features shape: {x.shape}, Target: {y.item()}")

