# BA__Projekt/BA__Programmierung/generator__nmavani__func_1.py

import os
from pathlib import Path

import numpy as np
import pandas as pd


def generate_dataset(n_samples=100, x_min=-10, x_max=10, seed=42):
    """
    Generates dataset for: y = 7*sin(x) + 3*|x/2|*epsilon, where epsilon ~ N(0, 1)
    Returns: pandas DataFrame
    """
    np.random.seed(seed)

    x = np.linspace(x_min, x_max, n_samples)
    epsilon = np.random.randn(n_samples)
    y = 7 * np.sin(x) + 3 * np.abs(x / 2) * epsilon

    return pd.DataFrame({"x": x, "y": y})


def save_dataset(df: pd.DataFrame, output_paths):
    """
    Saves the DataFrame to all specified output paths.
    """
    for path in output_paths:
        os.makedirs(path.parent, exist_ok=True)
        df.to_csv(path, index=False)
        print(f"Datensatz gespeichert unter: {path}")


if __name__ == "__main__":
    # === Settings ===
    output_name = "dataset__generated__nmavani__func_1.csv"

    output_paths = [
        Path("assets/data/source") / output_name,
        Path("assets/data/raw/dataset__generated__nmavani__func_1/") / output_name,
    ]

    df = generate_dataset(n_samples=100)
    save_dataset(df, output_paths)

