# BA__Projekt/BA__Programmierung/ml/ednn_regression__condition-based-maintenance-of-naval-propulsion-plants.py
"""
Train a Generic Ensemble Evidential Regressor on the Condition-Based Maintenance dataset.

This script:
- Loads the CBM dataset with 16 features
- Splits it into train/validation sets
- Initializes a GenericEnsembleRegressor with correct input_dim=16
- Trains the model using early stopping
"""

import torch
from torch.utils.data import DataLoader, random_split

from BA__Programmierung.ml.datasets.dataset__torch__condition_based_maintenance_of_naval_propulsion_plants import NavalPropulsionDataset
from BA__Programmierung.ml.utils.training_utils import train_with_early_stopping
from models.model__generic_ensemble import GenericEnsembleRegressor


def main():
    # === Configuration ===
    csv_path = "assets/data/raw/dataset__condition-based-maintenance-of-naval-propulsion-plants/data.csv"
    model_path = "assets/models/pth/ednn_regression__condition_based_maintenance_of_naval_propulsion_plants/ednn_regression__condition_based_maintenance_of_naval_propulsion_plants.pth"
    batch_size = 64
    learning_rate = 1e-3
    ensemble_size = 5
    expected_input_dim = 16  # Set to 16 if your dataset has 16 features

    # === Load dataset ===
    dataset = NavalPropulsionDataset(csv_path)

    # Verify dataset input dimension matches expected (16)
    actual_input_dim = dataset[0][0].shape[0]
    assert actual_input_dim == expected_input_dim, f"Dataset input dim {actual_input_dim} does not match expected {expected_input_dim}"

    # Split dataset into train and validation sets (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    # === Device setup ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Model configuration ===
    base_config = {
        "input_dim": expected_input_dim,  # 16 features input
        "hidden_dims": [64, 64],
        "output_type": "evidential",
        "use_dropout": False,
        "dropout_p": 0.2,
        "flatten_input": False,
        "use_batchnorm": False,
        "activation_name": "relu",
        "output_dim": 1
    }

    # Initialize ensemble model
    model = GenericEnsembleRegressor(base_config=base_config, n_models=ensemble_size).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # === Train model with early stopping ===
    train_with_early_stopping(model, train_loader, val_loader, optimizer, model_path, device)


if __name__ == "__main__":
    main()
