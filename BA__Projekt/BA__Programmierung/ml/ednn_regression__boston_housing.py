# BA__Programmierung/ml/ednn_regression__boston_housing.py

import os
import torch
from torch.utils.data import DataLoader, random_split
from pathlib import Path

from BA__Programmierung.ml.datasets.dataset__torch__boston_housing import DatasetTorchBostonHousing
from BA__Programmierung.ml.utils.training_utils import train_with_early_stopping


def get_model_save_dir(script_name: str) -> str:
    base_dir = "/root/BA__U-i-mlb-Sm-f-d-s-V-a-S/BA__Projekt/assets/pth"
    subfolder = Path(script_name).stem
    model_dir = os.path.join(base_dir, subfolder)
    os.makedirs(model_dir, exist_ok=True)
    return model_dir


def main():
    # Load dataset
    dataset = DatasetTorchBostonHousing(
        csv_path="assets/data/raw/dataset__boston-housing/dataset__boston-housing.csv"
    )

    # Split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size],
                                      generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=16)

    # Ensemble config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_config = {"input_dim": 13, "hidden_dims": [64, 64], "output_type": "evidential"}
    n_models = 5
    seed = 42

    # Save directory
    script_name = os.path.basename(__file__)
    model_save_dir = get_model_save_dir(script_name)

    for i in range(n_models):
        # Seeded initialization
        torch.manual_seed(seed + i)

        from models.model__generic import GenericRegressor
        model = GenericRegressor(**base_config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Save path per model
        model_path = os.path.join(model_save_dir, f"model_{i}.pth")

        print(f"Training model {i + 1}/{n_models}...")
        train_with_early_stopping(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            model_path=model_path,
            device=device,
            epochs=100,
            patience=10
        )


if __name__ == "__main__":
    main()
