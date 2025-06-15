# BA__Programmierung/ml/ednn_regression__wine_quality_red.py
"""
Train an evidential deep ensemble regressor on the Wine Quality Red dataset.

This script performs the following steps:

- Loads the Wine Quality Red dataset from CSV.
- Splits the dataset into training and validation sets (80/20 split).
- Creates DataLoaders for batching.
- Configures and initializes a GenericEnsembleRegressor model.
- Trains the model with early stopping, saving the best model checkpoint.

Usage:
    Run this script directly to start training.

Example:
    $ python ednn_regression__wine_quality_red.py
"""

import torch
from torch.utils.data import DataLoader, random_split

from BA__Programmierung.ml.datasets.dataset__torch__wine_quality_red import load_wine_quality_red_dataset
from models.model__generic_ensemble import GenericEnsembleRegressor
from BA__Programmierung.ml.utils.training_utils import train_with_early_stopping


def main():
    """
    Main training routine.

    Loads the dataset, splits into train/val, initializes model and optimizer,
    then trains using early stopping.

    No parameters required; all paths and hyperparameters are hardcoded.

    Steps:
    1. Load dataset from CSV.
    2. Split dataset into train (80%) and validation (20%) with fixed seed.
    3. Create DataLoader instances for train and val sets.
    4. Setup device (GPU if available, else CPU).
    5. Configure model parameters and instantiate ensemble regressor.
    6. Setup Adam optimizer.
    7. Train model with early stopping; save best model.
    """
    dataset = load_wine_quality_red_dataset("assets/data/raw/dataset__wine-quality/winequality-red.csv")

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size],
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

    model_path = "assets/models/pth/ednn_regression__wine_quality_red/generic_ensemble__wine-quality-red.pt"

    train_with_early_stopping(model, train_loader, val_loader, optimizer, model_path, device)


if __name__ == "__main__":
    main()
