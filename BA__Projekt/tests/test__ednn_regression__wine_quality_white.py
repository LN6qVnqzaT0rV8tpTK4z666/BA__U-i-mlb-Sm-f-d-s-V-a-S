import torch
import pytest
import numpy as np
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_percentage_error

from BA__Programmierung.ml.datasets.dataset__torch__wine_quality_white import WineQualityWhiteDataset
from models.model__ednn_basic import EvidentialNet


@pytest.fixture(scope="module")
def data_and_model():
    # === Load and scale dataset ===
    csv_path = "assets/data/raw/dataset__wine-quality/winequality-white.csv"
    dataset = WineQualityWhiteDataset(csv_path)

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

    model = EvidentialNet(input_dim).to(device)
    model.load_state_dict(torch.load("models/ednn__wine-quality-white.pt", map_location=device))
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
        break  # Test one batch is sufficient


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

    # Inverse transform
    mu_all = scaler_y.inverse_transform(mu_all)
    targets_all = scaler_y.inverse_transform(targets_all)

    r2 = r2_score(targets_all, mu_all)
    mape = mean_absolute_percentage_error(targets_all, mu_all)

    print(f"[TEST] R²: {r2:.4f}, MAPE: {mape * 100:.2f}%")

    assert r2 > 0.0, "Expected R² > 0.0"
    assert mape < 1.0, "Expected MAPE < 100%"


def test_model_class_type(data_and_model):
    model, _, _, _ = data_and_model
    assert isinstance(model, EvidentialNet), "Model should be of type EvidentialNet"
