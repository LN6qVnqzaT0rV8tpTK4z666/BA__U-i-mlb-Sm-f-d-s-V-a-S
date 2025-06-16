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

import torch
from torch.utils.data import DataLoader, random_split

from BA__Programmierung.ml.datasets.dataset__torch__concrete_compressive_strength import (
    DatasetTorchConcreteCompressiveStrength,
)
from BA__Programmierung.ml.utils.training_utils import train_with_early_stopping
from models.model__generic_ensemble import GenericEnsembleRegressor


def main():
    # === Configuration ===
    csv_path = "assets/data/raw/dataset__concrete-compressive-strength/Concrete_Data.csv"
    batch_size = 64
    learning_rate = 1e-3
    ensemble_size = 5
    model_path = "assets/models/pth/ednn_regression__concrete_compressive_strength/generic_ensemble__concrete.pth"

    # === Load and split dataset ===
    dataset = DatasetTorchConcreteCompressiveStrength(csv_path)
    input_dim = dataset[0][0].shape[0]

    train_set, val_set = random_split(
        dataset,
        [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    # === Device ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Model config ===
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

    model = GenericEnsembleRegressor(base_config=base_config, n_models=ensemble_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # === Train ===
    train_with_early_stopping(model, train_loader, val_loader, optimizer, model_path, device)


if __name__ == "__main__":
    main()
