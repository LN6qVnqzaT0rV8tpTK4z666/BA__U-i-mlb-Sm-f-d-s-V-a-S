# BA__Projekt/BA__Programmierung/ml/ednn_regression__kin8nm.py
"""
Train a Generic Ensemble Evidential Deep Neural Network on the Kin8nm dataset.

This script loads the Kin8nm regression dataset, splits it into training and validation sets,
initializes a generic ensemble evidential regression model, and trains it using
early stopping. The trained model is saved to disk.

Modules:
- Dataset loader: `load_kin8nm_dataset`
- Model: `GenericEnsembleRegressor`
- Training utils: `train_with_early_stopping`

Usage:
    python ednn_regression__kin8nm.py

"""

import torch
from torch.utils.data import DataLoader, random_split

from BA__Programmierung.ml.datasets.dataset__torch__kin8nm import load_kin8nm_dataset
from models.model__generic_ensemble import GenericEnsembleRegressor
from BA__Programmierung.ml.utils.training_utils import train_with_early_stopping


def main():
    """
    Main entry point for training the Generic Ensemble Regressor on Kin8nm data.

    Steps:
    - Loads Kin8nm dataset from CSV.
    - Splits into train/validation sets (80/20).
    - Creates DataLoaders with batch size 32.
    - Configures the ensemble model with evidential output.
    - Trains model with Adam optimizer and early stopping.
    - Saves best model checkpoint to file.

    Args:
        None

    Returns:
        None
    """
    dataset_path = "assets/data/raw/dataset__kin8nm-dataset_2175/dataset__kin8nm-dataset_2175.csv"
    dataset = load_kin8nm_dataset(dataset_path)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_dim = dataset[0][0].shape[0]
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

    model = GenericEnsembleRegressor(base_config=base_config, n_models=5).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model_save_path = "assets/models/pth/ednn_regression__kin8nm/generic_ensemble__kin8nm.pt"

    train_with_early_stopping(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        model_path=model_save_path,
        device=device,
        epochs=50,
        patience=5,
    )


if __name__ == "__main__":
    main()
