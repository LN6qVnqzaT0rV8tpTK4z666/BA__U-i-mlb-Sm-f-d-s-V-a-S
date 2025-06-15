# BA__Programmierung/ml/ednn_regression__nmavani_func1.py
"""
Train an evidential deep ensemble regressor on the nmavani_func1 dataset.

This script performs the following steps:

- Loads the nmavani_func1 dataset from a DuckDB database table.
- Splits the dataset into training and validation sets (80/20 split).
- Creates DataLoaders for batching.
- Configures and initializes a GenericEnsembleRegressor model.
- Trains the model with early stopping, saving the best model checkpoint.

Usage:
    Run this script directly to start training.

Example:
    $ python ednn_regression__nmavani_func1.py
"""

import torch
from torch.utils.data import DataLoader, random_split

from BA__Programmierung.ml.datasets.dataset__torch__nmavani_func1 import DatasetTorchDuckDBFunc1
from models.model__generic_ensemble import GenericEnsembleRegressor
from BA__Programmierung.ml.utils.training_utils import train_with_early_stopping


def main():
    """
    Main entry point for training.

    Loads the nmavani_func1 dataset from DuckDB, splits it into training and validation sets,
    prepares DataLoader instances, initializes the model and optimizer, then starts training
    using early stopping with checkpoint saving.

    No parameters required; paths and hyperparameters are hardcoded.

    Steps:
    1. Load dataset from DuckDB.
    2. Split dataset into train (80%) and validation (20%) sets using fixed random seed.
    3. Create DataLoader instances for train and val sets.
    4. Setup device (GPU if available, else CPU).
    5. Configure model parameters and instantiate GenericEnsembleRegressor.
    6. Setup Adam optimizer.
    7. Train model with early stopping, saving best checkpoint.
    """
    db_path = "assets/dbs/dataset__generated-nmavani-func_1.duckdb"
    table_name = "generated_nmavani_func_1_csv"
    dataset = DatasetTorchDuckDBFunc1(db_path=db_path, table_name=table_name)

    train_set, val_set = random_split(
        dataset,
        [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))],
        generator=torch.Generator().manual_seed(42),
    )
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
    model_path = "assets/models/pth/ednn_regression__nmavani_func1/generic_ensemble__nmavani_func1.pt"

    train_with_early_stopping(model, train_loader, val_loader, optimizer, model_path, device)


if __name__ == "__main__":
    main()
