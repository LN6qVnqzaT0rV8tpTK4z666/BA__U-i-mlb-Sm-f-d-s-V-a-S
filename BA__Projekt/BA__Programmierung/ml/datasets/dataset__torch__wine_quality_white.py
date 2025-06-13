# BA__Projekt/BA__Programmierung/ml/datasets/dataset__torch__wine_quality_white.py

import os
import pandas as pd
import torch

from torch.utils.data import Dataset


class WineQualityWhiteDataset(Dataset):
    def __init__(self, csv_path):
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"File not found: {csv_path}")

        df = pd.read_csv(csv_path, sep=";")
        self.features = torch.tensor(
            df.iloc[:, :-1].values, dtype=torch.float32
        )
        self.labels = torch.tensor(
            df.iloc[:, -1].values, dtype=torch.long
        )  # classification target

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def load_wine_quality_white_dataset(csv_path):
    """
    Loads the Wine Quality White dataset from the provided CSV path.
    Returns a torch.utils.data.Dataset instance.
    """
    return WineQualityWhiteDataset(csv_path)


if __name__ == "__main__":
    dataset_path = "/root/BA__Projekt/assets/data/raw/dataset__wine-quality/winequality-white.csv"
    dataset = load_wine_quality_white_dataset(dataset_path)

    print(f"Loaded {len(dataset)} samples.")
    x, y = dataset[0]
    print(f"Features shape: {x.shape}, Quality label: {y}")
