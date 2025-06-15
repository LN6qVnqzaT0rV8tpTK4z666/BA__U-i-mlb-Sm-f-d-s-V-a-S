# BA__Projekt/tests/test__ednn_regression__kin8nm.py
import os
import numpy as np
import pytest
import torch
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from BA__Programmierung.ml.datasets.dataset__torch__kin8nm import load_kin8nm_dataset
from models.model__generic_ensemble import GenericEnsembleRegressor


@pytest.fixture(scope="module")
def data_and_model():
    csv_path = "/root/BA__Projekt/assets/data/raw/dataset__kin8nm/kin8nm.csv"
    dataset = load_kin8nm_dataset(csv_path)

    X = dataset.features.numpy()
    Y = dataset.labels.numpy().reshape(-1, 1)

    scaler_x = StandardScaler().fit(X)
    scaler_y = StandardScaler().fit(Y)

    X_scaled = torch.tensor(scaler_x.transform(X), dtype=torch.float32)
    Y_scaled = torch.tensor(scaler_y.transform(Y), dtype=torch.float32)

    full_dataset = torch.utils.data.TensorDataset(X_scaled, Y_scaled)
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

    model_path = "assets/models/pth/ednn_regression__kin8nm/generic_ensemble__kin8nm.pt"
    if not os.path.isfile(model_path):
        pytest.skip(f"Model file not found at {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

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
        break  # test one batch is enough


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
    assert isinstance(model, GenericEnsembleRegressor), "Model should be GenericEnsembleRegressor"
