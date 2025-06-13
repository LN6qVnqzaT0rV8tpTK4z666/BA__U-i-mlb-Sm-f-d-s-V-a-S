# BA__Projekt/BA__Programmierung/ml/datasets/dataset__torch__wine_quality_white.py

import os

import pandas as pd

from BA__Programmierung.ml.datasets.dataset__torch__base_tabular import (
    BaseTabularDataset,
)


class WineQualityWhiteDataset(BaseTabularDataset):
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
            normalize=False,      # Klassifikation, keine Skalierung
            classification=False   # Labels sind Klassen
        )


def load_wine_quality_white_dataset(csv_path):
    return WineQualityWhiteDataset(csv_path)


if __name__ == "__main__":
    dataset_path = "/root/BA__Projekt/assets/data/raw/dataset__wine-quality/winequality-white.csv"
    dataset = load_wine_quality_white_dataset(dataset_path)

    print(f"Loaded {len(dataset)} samples.")
    x, y = dataset[0]
    print(f"Features shape: {x.shape}, Quality label: {y}")
