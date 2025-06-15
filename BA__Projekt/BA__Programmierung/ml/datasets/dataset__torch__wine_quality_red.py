# BA__Projekt/BA__Programmierung/ml/datasets/dataset__torch__wine_quality_red.py

import os
import pandas as pd

from BA__Programmierung.ml.datasets.dataset__torch__base_tabular import BaseTabularDataset


class WineQualityRedDataset(BaseTabularDataset):
    """
    PyTorch Dataset for the Red Wine Quality dataset loaded from a CSV file.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file containing the red wine quality data.
        The CSV is expected to be semicolon-separated (';').

    Raises
    ------
    FileNotFoundError
        If the specified CSV file does not exist.

    Notes
    -----
    - Assumes the last column is the target quality label.
    - Treated as a regression task (classification=False).
    - No normalization is applied by default.
    """

    def __init__(self, csv_path):
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"File not found: {csv_path}")

        df = pd.read_csv(csv_path, sep=";")
        input_cols = df.columns[:-1].tolist()
        target_col = df.columns[-1]

        super().__init__(
            dataframe=df,
            input_cols=input_cols,
            target_cols=target_col,
            normalize=False,
            classification=False
        )


def load_wine_quality_red_dataset(csv_path):
    """
    Helper function to load the WineQualityRedDataset.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file.

    Returns
    -------
    WineQualityRedDataset
        Instance of the WineQualityRedDataset.
    """
    return WineQualityRedDataset(csv_path)


if __name__ == "__main__":
    dataset_path = "/root/BA__Projekt/assets/data/raw/dataset__wine-quality/winequality-red.csv"
    dataset = load_wine_quality_red_dataset(dataset_path)

    print(f"Loaded {len(dataset)} samples.")
    x, y = dataset[0]
    print(f"Features shape: {x.shape}, Quality label: {y}")
