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

import torch
from torch.utils.data import DataLoader, random_split

from BA__Programmierung.ml.datasets.dataset__torch__duckdb_iris import DatasetTorchDuckDBIris
from models.model__generic_ensemble import GenericEnsembleRegressor
from BA__Programmierung.ml.utils.training_utils import train_with_early_stopping


def main():
    db_path = "assets/dbs/dataset__iris-dataset.duckdb"
    table_name = "iris_dataset_csv"
    dataset = DatasetTorchDuckDBIris(db_path=db_path, table_name=table_name)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_config = {
        "input_dim": 4,  # Iris features count
        "hidden_dims": [64, 64],
        "output_type": "evidential",
        "use_dropout": False,
        "dropout_p": 0.2,
        "flatten_input": False,
        "use_batchnorm": False,
        "activation_name": "relu",
    }

    model = GenericEnsembleRegressor(base_config=base_config, n_models=5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model_path = "assets/models/pth/ednn_regression__iris_ensemble/generic_ensemble__iris.pt"

    train_with_early_stopping(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        model_path=model_path,
        device=device,
        epochs=100,
        patience=5,
    )


if __name__ == "__main__":
    main()
