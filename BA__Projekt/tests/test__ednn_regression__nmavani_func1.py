# BA__Projekt/tests/test__ednn_regression__nmavani_func1.py

import numpy as np
import pytest
import torch

from sklearn.metrics import r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from BA__Programmierung.config import (
    DB_PATH__GENERATED__MAVANI__FUNC_1,
    CSV_PATH__GENERATED__MAVANI__FUNC_1,
)
from BA__Programmierung.ml.datasets.dataset__torch__nmavani_func1 import DatasetTorchDuckDBFunc1
from models.model__generic_ensemble import GenericEnsembleRegressor


@pytest.fixture(scope="module")
def data_and_model():
    # === Load dataset ===
    dataset = DatasetTorchDuckDBFunc1(
        db_path=DB_PATH__GENERATED__MAVANI__FUNC_1,
        table_name=CSV_PATH__GENERATED__MAVANI__FUNC_1,
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    # === Normalize labels (targets) ===
    targets = torch.stack([y for _, y in dataset]).unsqueeze(1).numpy()
    scaler_y = StandardScaler().fit(targets)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Load model ===
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
    model_path = "assets/models/pth/ednn_regression__nmavani_func1/generic_ensemble__nmavani_func1.pt"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return model, loader, scaler_y, device


def test_output_shapes(data_and_model):
    model, loader, _, device = data_and_model

    for X, _ in loader:
        X = X.to(device)
        with torch.no_grad():
            mu, v, alpha, beta = model(X)
        assert mu.shape == (X.shape[0], 1)
        assert v.shape == (X.shape[0], 1)
        assert alpha.shape == (X.shape[0], 1)
        assert beta.shape == (X.shape[0], 1)
        break


def test_model_performance(data_and_model):
    model, loader, scaler_y, device = data_and_model

    mu_all, y_all = [], []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            mu, _, _, _ = model(X)
            mu_all.append(mu.cpu().numpy())
            y_all.append(y.numpy())

    mu_all = np.vstack(mu_all)
    y_all = np.vstack(y_all)

    # Inverse transform
    mu_all = scaler_y.inverse_transform(mu_all)
    y_all = scaler_y.inverse_transform(y_all)

    r2 = r2_score(y_all, mu_all)
    mape = mean_absolute_percentage_error(y_all, mu_all)

    print(f"[TEST] R²: {r2:.4f}, MAPE: {mape * 100:.2f}%")

    assert r2 > 0.0, "Expected R² > 0.0"
    assert mape < 1.0, "Expected MAPE < 100%"


def test_model_type(data_and_model):
    model, _, _, _ = data_and_model
    assert isinstance(model, GenericEnsembleRegressor)

