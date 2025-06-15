# BA__Projekt/BA__Programmierung/ml/ednn_regression__energy-efficiency.py

import os
import torch
from torch.utils.data import DataLoader, random_split

from BA__Programmierung.ml.datasets.dataset__torch__energy_efficiency import EnergyEfficiencyDataset
from BA__Programmierung.ml.utils.training_utils import train_with_early_stopping
from models.model__generic_ensemble import GenericEnsembleRegressor


def main():
    dataset_path = "assets/data/raw/dataset__energy-efficiency/dataset__energy-efficiency.csv"
    dataset = EnergyEfficiencyDataset(dataset_path)

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
        "output_type": "evidential",  # assuming evidential regression here
        "use_dropout": False,
        "dropout_p": 0.2,
        "flatten_input": False,
        "use_batchnorm": False,
        "activation_name": "relu",
    }
    model = GenericEnsembleRegressor(base_config=base_config, n_models=5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model_path = "assets/models/pth/ednn_regression__energy_efficiency_ensemble/ednn_regression__energy_efficiency_ensemble.pth"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    train_with_early_stopping(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        model_path=model_path,
        device=device
    )

if __name__ == "__main__":
    main()
