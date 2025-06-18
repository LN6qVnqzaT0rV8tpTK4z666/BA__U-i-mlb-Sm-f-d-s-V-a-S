# BA__Projekt/BA__Programmierung/ml/ednn_regression__concrete-compressive-strength.py
"""
Train a Generic Ensemble Evidential Regressor on the Concrete Compressive Strength dataset.

This script:
- Loads the concrete dataset from CSV
- Splits it into train/validation sets
- Initializes a GenericEnsembleRegressor
- Trains the model using early stopping

Usage:
    python ednn_regression__concrete_compressive_strength_ensemble.py
"""

import os
import torch

from models.model__generic import GenericRegressor

from BA__Programmierung.ml.datasets.dataset__torch__concrete_compressive_strength import (
    DatasetTorchConcreteCompressiveStrength,
)
from BA__Programmierung.ml.metrics.metrics_registry import MetricsRegistry
from BA__Programmierung.ml.utils.training_utils import train_with_early_stopping
from torch.utils.data import DataLoader, random_split


def main():
    # === Configuration ===
    csv_path = "assets/data/raw/dataset__concrete-compressive-strength/Concrete_Data.csv"
    batch_size = 64
    learning_rate = 1e-3
    n_models = 5
    seed = 42

    # === Load and split dataset ===
    dataset = DatasetTorchConcreteCompressiveStrength(csv_path)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    # === Device ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = dataset[0][0].shape[0]

    # === Base model config ===
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

    # === Paths and losses ===
    model_save_base = "assets/models/pth/ednn_regression__concrete_compressive_strength"
    metric_bundles = MetricsRegistry.get_metric_bundles()
    # loss_modes = ["nll", "abs", "mse", "kl", "scaled", "variational", "full"]
    loss_modes = ["mse"]
    
    print("Available tokens: ")
    print(metric_bundles)

    for loss_mode in loss_modes:
        model_save_dir = os.path.join(model_save_base, loss_mode)
        os.makedirs(model_save_dir, exist_ok=True)

        for i in range(n_models):
            torch.manual_seed(seed + i)

            model = GenericRegressor(**base_config).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            model_path = os.path.join(model_save_dir, f"model_{i}.pth")
            print(f"[{loss_mode.upper()}] Training model {i + 1}/{n_models}...")

            # Decide which token to use for metrics
            if loss_mode in ["nll", "full", "variational", "kl"]:
                metrics_token = "uq"
            elif loss_mode in ["mse", "abs"]:
                metrics_token = "regression"
            else:
                metrics_token = None  # or "probabilistic" depending on your setup

            train_with_early_stopping(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                model_path=model_path,
                device=device,
                epochs=100,
                patience=10,
                loss_mode=loss_mode,
                metrics_token=metrics_token
            )


if __name__ == "__main__":
    main()
