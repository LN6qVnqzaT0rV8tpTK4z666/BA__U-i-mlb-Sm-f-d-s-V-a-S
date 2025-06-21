# BA__Projekt/BA__Programmierung/ml/ednn_regression__energy-efficiency.py

# BA__Projekt/BA__Programmierung/ml/ednn_regression__energy-efficiency.py

import os
import torch

from BA__Programmierung.ml.datasets.dataset__torch__energy_efficiency import EnergyEfficiencyDataset
from BA__Programmierung.ml.metrics.metrics_registry import MetricsRegistry
from BA__Programmierung.ml.utils.training_utils import load_model_checkpoint, train_with_early_stopping
from models.model__generic_ensemble import GenericEnsembleRegressor
from torch.utils.data import DataLoader, random_split

def main():
    """
    Main function for training an ensemble model on the energy efficiency dataset.

    This script performs the following steps:
    1. Loads the Energy Efficiency dataset.
    2. Splits the dataset into training and validation sets (80/20).
    3. Defines and configures the ensemble model.
    4. Trains the model using the `train_with_early_stopping` function.
    5. Saves the trained models.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Example
    -------
    >>> main()  # Run the main function for training and evaluating the model
    """
    dataset_path = "assets/data/raw/dataset__energy-efficiency/dataset__energy-efficiency.csv"
    dataset = EnergyEfficiencyDataset(dataset_path)

    # Split dataset (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = next(iter(train_loader))[0].shape[1]

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
    metric_bundles = MetricsRegistry.get_metric_bundles()

    # loss_modes determine the type of loss function to use for training
    loss_modes = ["mse"]

    model_save_base = "assets/models/pth/ednn_regression__energy_efficiency_ensemble"

    print("Available tokens: ")
    print(metric_bundles)

    # Training loop for multiple models
    for loss_mode in loss_modes:
        model_save_dir = os.path.join(model_save_base, loss_mode)
        os.makedirs(model_save_dir, exist_ok=True)

        for i in range(n_models):
            torch.manual_seed(seed + i)

            # Initialize and train the ensemble model
            model = GenericEnsembleRegressor(base_config=base_config, n_models=n_models).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            model_path = os.path.join(model_save_dir, f"model_{i}.pth")
            print(f"[{loss_mode.upper()}] Training model {i + 1}/{n_models}...")

            # Load checkpoint if it exists
            checkpoint, checkpoint_exists = load_model_checkpoint(model, optimizer, model_path, device)

            # Decide which token to use for metrics
            if loss_mode in ["nll", "full", "variational", "kl"]:
                metrics_token = "uq"
            elif loss_mode in ["mse", "abs"]:
                metrics_token = "regression"
            else:
                metrics_token = None  # or "probabilistic" depending on your setup

            # If checkpoint exists, resume from the last epoch
            if checkpoint_exists:
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
                    metrics_token=metrics_token,
                    resume_epoch=checkpoint['epoch']  # Resume from last checkpoint epoch
                )
            else:
                # If no checkpoint exists, train the model from scratch
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
                    metrics_token=metrics_token,
                    resume_epoch=0  # Start from epoch 0
                )

if __name__ == "__main__":
    main()
