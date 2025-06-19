# BA__Projekt/tests/test__ednn_regression__combined_cycle_power_plant.py

import numpy as np
import pytest
import torch
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from BA__Programmierung.ml.datasets.dataset__torch__combined_cycle_power_plant import (
    DatasetTorchCombinedCyclePowerPlant,
)
from models.model__generic_ensemble import GenericEnsembleRegressor


@pytest.fixture(scope="module")
def data_and_model():
    # === Load and scale dataset ===
    db_path = "root/BA__U-i-mlb-Sm-f-d-s-V-a-S/BA__Projekt/assets/dbs/dataset__combined-cycle-power-plant.duckdb"
    table_name = "combined_cycle_power_plant__dataset_csv"
    dataset = DatasetTorchCombinedCyclePowerPlant(db_path=db_path, table_name=table_name)

    X = dataset.features.numpy()
    Y = dataset.labels.numpy().reshape(-1, 1)

    scaler_x = StandardScaler().fit(X)
    scaler_y = StandardScaler().fit(Y)

    X_scaled = torch.tensor(scaler_x.transform(X), dtype=torch.float32)
    Y_scaled = torch.tensor(scaler_y.transform(Y), dtype=torch.float32)

    full_dataset = TensorDataset(X_scaled, Y_scaled)
    loader = DataLoader(full_dataset, batch_size=64, shuffle=False)

    input_dim = X_scaled.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    # If pretrained model available:
    model_path = "assets/models/pth/ednn_regression__combined_cycle_power_plant/ednn_regression__combined_cycle_power_plant.pth"
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    except FileNotFoundError:
        print(f"[WARNING] Model not found at {model_path}. Continuing with randomly initialized model.")

    return model, loader, scaler_y, device


def test_model_output_shapes(data_and_model):
    model, loader, _, device = data_and_model

    for x, _ in loader:
        x = x.to(device)
        with torch.no_grad():
            mu, v, alpha, beta = model(x)
        assert mu.shape == (x.shape[0], 1)
        assert v.shape == (x.shape[0], 1)
        assert alpha.shape == (x.shape[0], 1)
        assert beta.shape == (x.shape[0], 1)
        break


def test_model_r2_and_mape(data_and_model):
    model, loader, scaler_y, device = data_and_model

    mu_all, targets_all = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            mu, _, _, _ = model(x)
            mu_all.append(mu.cpu().numpy())
            targets_all.append(y.numpy())

    mu_all = np.vstack(mu_all)
    targets_all = np.vstack(targets_all)

    mu_all = scaler_y.inverse_transform(mu_all)
    targets_all = scaler_y.inverse_transform(targets_all)

    r2 = r2_score(targets_all, mu_all)
    mape = mean_absolute_percentage_error(targets_all, mu_all)

    print(f"[TEST] R²: {r2:.4f}, MAPE: {mape * 100:.2f}%")

    assert r2 > 0.0, "Expected R² > 0.0"
    assert mape < 1.0, "Expected MAPE < 100%"


def test_model_class_type(data_and_model):
    model, _, _, _ = data_and_model
    assert isinstance(model, GenericEnsembleRegressor), "Model should be of type GenericEnsembleRegressor"

