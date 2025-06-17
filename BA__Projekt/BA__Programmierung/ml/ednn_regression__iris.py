# BA__Projekt/BA__Programmierung/ml/ednn_regression__iris.py
"""
Train an evidential deep ensemble regressor on the Iris dataset.

This script performs the following steps:

- Loads the Iris dataset from DuckDB.
- Splits into training and validation sets (80/20).
- Creates DataLoaders.
- Configures and initializes a GenericEnsembleRegressor model.
- Trains with early stopping and saves the best checkpoint.

Usage:
    Run this script directly to start training.
"""

import os
import torch
from torch.utils.data import DataLoader, random_split

from BA__Programmierung.ml.datasets.dataset__torch__duckdb_iris import DatasetTorchDuckDBIris
from models.model__generic_ensemble import GenericEnsembleRegressor
from BA__Programmierung.ml.utils.training_utils import train_with_early_stopping


def main():
    # === Dataset loading ===
    db_path = "assets/dbs/dataset__iris-dataset.duckdb"
    table_name = "iris_dataset_csv"
    dataset = DatasetTorchDuckDBIris(db_path=db_path, table_name=table_name)

    # === Split dataset (80/20) ===
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    # === Dataloaders ===
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False)

    # === Device and model config ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = 4  # Iris dataset has 4 features

    base_config = {
        "input_dim": input_dim,
        "hidden_dims": [64, 64],
        "output_type": "evidential",
        "use_dropout": False,
        "dropout_p": 0.2,
        "flatten_input": False,
        "use_batchnorm": False,
        "activation_name": "relu",
    }

    n_models = 5
    seed = 42
    # metric_bundles = Metrics.get_metric_bundles()
    loss_modes = ["nll", "abs", "mse", "kl", "scaled", "variational", "full"]
    model_save_base = "assets/models/pth/ednn_regression__iris_ensemble"

    # === Training ===
    for loss_mode in loss_modes:
        model_save_dir = os.path.join(model_save_base, loss_mode)
        os.makedirs(model_save_dir, exist_ok=True)

        for i in range(n_models):
            torch.manual_seed(seed + i)

            model = GenericEnsembleRegressor(base_config=base_config, n_models=n_models).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            model_path = os.path.join(model_save_dir, f"model_{i}.pth")
            print(f"[{loss_mode.upper()}] Training model {i + 1}/{n_models}...")

            train_with_early_stopping(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                model_path=model_path,
                device=device,
                epochs=100,
                patience=5,
                loss_mode=loss_mode
            )


if __name__ == "__main__":
    main()
