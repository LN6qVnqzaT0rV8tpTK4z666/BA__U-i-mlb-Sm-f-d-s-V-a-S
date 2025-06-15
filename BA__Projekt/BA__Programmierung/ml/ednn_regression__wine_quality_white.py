# BA__Programmierung/ml/ednn_regression__wine-quality-white_ensemble.py
"""
Module for training an evidential deep neural network ensemble on the Wine Quality White dataset.

This script:
- Loads the Wine Quality White dataset
- Splits it into training and validation sets
- Defines and initializes a Generic Ensemble Regressor model
- Trains the model with early stopping using a utility function

Usage:
    Run the script directly to start training.

Example:
    $ python ednn_regression__wine-quality-white_ensemble.py
"""

import torch
from torch.utils.data import DataLoader, random_split
from BA__Programmierung.ml.datasets.dataset__torch__wine_quality_white import load_wine_quality_white_dataset
from models.model__generic_ensemble import GenericEnsembleRegressor
from BA__Programmierung.ml.utils.training_utils import train_with_early_stopping


def main():
    """
    Load dataset, prepare data loaders, initialize model and optimizer, then train with early stopping.

    Steps:
    - Load Wine Quality White dataset from CSV.
    - Split dataset 80/20 into training and validation sets with fixed random seed.
    - Create DataLoader instances for both splits.
    - Set device (GPU if available, else CPU).
    - Define model configuration and instantiate GenericEnsembleRegressor.
    - Setup Adam optimizer.
    - Train model using `train_with_early_stopping`, saving the best model checkpoint.

    No arguments are required. All paths and parameters are hardcoded for simplicity.
    """
    dataset = load_wine_quality_white_dataset("assets/data/raw/dataset__wine-quality/winequality-white.csv")

    train_set, val_set = random_split(dataset,
                                      [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))],
                                      generator=torch.Generator().manual_seed(42))
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
    model_path = "assets/models/pth/ednn_regression__wine_quality_white_ensemble/generic_ensemble__wine-quality-white.pt"

    train_with_early_stopping(model, train_loader, val_loader, optimizer, model_path, device)


if __name__ == "__main__":
    main()
