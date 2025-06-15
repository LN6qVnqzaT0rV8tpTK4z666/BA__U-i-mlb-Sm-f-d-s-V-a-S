# BA__Projekt/tests/test__ednn_regression__iris.py

import numpy as np
import pytest
import torch
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from BA__Programmierung.ml.datasets.dataset__torch__duckdb_iris import DatasetTorchDuckDBIris
from models.model__generic_ensemble import GenericEnsembleRegressor


@pytest.fixture(scope="module")
def data_and_model():
    # === Load dataset ===
    dataset = DatasetTorchDuckDBIris(
        db_path="/root/BA__U-i-mlb-Sm-f-d-s-V-a-S/BA__Projekt/assets/dbs/dataset__iris-dataset.duckdb",
        table_name="iris_dataset_csv"
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    # === Normalize labels ===
    targets = torch.stack([y for _, y in dataset]).unsqueeze(1).numpy()
    scaler_y = StandardScaler().fit(targets)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Initialize model ===
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
    model_path = "assets/models/pth/ednn_regression__iris/generic_ensemble__iris.pt"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return model, loader, scaler_y, device


def test_output_shapes(data_and_model):
    model, loader, _, device = data_and_model

    for X, _ in loader:
        X = X.to(device)
        with torch.no_grad():
            mu, v, alpha, beta = model(X)

        assert mu.shape == (X.shape[0], 1), "mu shape mismatch"
        assert v.shape == (X.shape[0], 1), "v shape mismatch"
        assert alpha.shape == (X.shape[0], 1), "alpha shape mismatch"
        assert beta.shape == (X.shape[0], 1), "beta shape mismatch"
        break


def test_parameter_ranges(data_and_model):
    model, loader, _, device = data_and_model

    for X, _ in loader:
        X = X.to(device)
        with torch.no_grad():
            _, v, alpha, beta = model(X)

        assert torch.all(v > 0), "v should be positive"
        assert torch.all(alpha > 1), "alpha should be > 1"
        assert torch.all(beta > 0), "beta should be positive"
        break


def test_r2_and_mape(data_and_model):
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

    mu_all = scaler_y.inverse_transform(mu_all)
    y_all = scaler_y.inverse_transform(y_all)

    r2 = r2_score(y_all, mu_all)
    mape = mean_absolute_percentage_error(y_all, mu_all)

    print(f"[TEST] R²: {r2:.4f}, MAPE: {mape * 100:.2f}%")

    assert r2 > 0.0, "Expected R² > 0.0"
    assert mape < 1.0, "Expected MAPE < 100%"


def test_model_type(data_and_model):
    model, _, _, _ = data_and_model
    assert isinstance(model, GenericEnsembleRegressor), "Model should be of type GenericEnsembleRegressor"
