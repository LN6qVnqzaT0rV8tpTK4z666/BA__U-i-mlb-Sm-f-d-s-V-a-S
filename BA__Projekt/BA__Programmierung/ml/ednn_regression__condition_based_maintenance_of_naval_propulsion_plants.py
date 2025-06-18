# BA__Projekt/BA__Programmierung/ml/ednn_regression__condition-based-maintenance-of-naval-propulsion-plants.py
"""
Train a Generic Ensemble Evidential Regressor on the Condition-Based Maintenance dataset.

This script:
- Loads the CBM dataset with 16 features
- Splits it into train/validation sets
- Initializes a GenericEnsembleRegressor with correct input_dim=16
- Trains the model using early stopping
"""

import os
import torch

from BA__Programmierung.ml.metrics.metrics_registry import Metrics
from BA__Programmierung.ml.datasets.dataset__torch__condition_based_maintenance_of_naval_propulsion_plants import NavalPropulsionDataset
from BA__Programmierung.ml.utils.training_utils import train_with_early_stopping
from models.model__generic_ensemble import GenericEnsembleRegressor
from torch.utils.data import DataLoader, random_split


def main():
    # === Configuration ===
    csv_path = "assets/data/raw/dataset__condition-based-maintenance-of-naval-propulsion-plants/data.csv"
    batch_size = 64
    learning_rate = 1e-3
    ensemble_size = 5
    expected_input_dim = 16  # Dataset features count

    # Base model save directory
    model_save_base = "assets/models/pth/ednn_regression__condition_based_maintenance_of_naval_propulsion_plants"

    # === Load dataset ===
    dataset = NavalPropulsionDataset(csv_path)

    actual_input_dim = dataset[0][0].shape[0]
    assert actual_input_dim == expected_input_dim, f"Input dim mismatch: {actual_input_dim} != {expected_input_dim}"

    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Base model config
    base_config = {
        "input_dim": expected_input_dim,
        "hidden_dims": [64, 64],
        "output_type": "evidential",
        "use_dropout": False,
        "dropout_p": 0.2,
        "flatten_input": False,
        "use_batchnorm": False,
        "activation_name": "relu",
        "output_dim": 1,
    }

    # Training params
    n_models = ensemble_size
    seed = 42
    metric_bundles = Metrics.get_metric_bundles()
    loss_modes = ["nll", "abs", "mse", "kl", "scaled", "variational", "full"]

    for loss_mode in loss_modes:
        model_save_dir = os.path.join(model_save_base, loss_mode)
        os.makedirs(model_save_dir, exist_ok=True)

        for i in range(n_models):
            torch.manual_seed(seed + i)

            model = GenericEnsembleRegressor(base_config=base_config, n_models=ensemble_size).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

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
                patience=10,
                loss_mode=loss_mode
            )


if __name__ == "__main__":
    main()
