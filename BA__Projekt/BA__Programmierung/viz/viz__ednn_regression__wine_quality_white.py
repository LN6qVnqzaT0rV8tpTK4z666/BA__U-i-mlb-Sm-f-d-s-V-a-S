# BA__Projekt/BA__Programmierung/viz/test__ednn_regression__wine-quality-white.py
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from BA__Programmierung.ml.datasets.dataset__torch__wine_quality_white import WineQualityWhiteDataset
from models.model__ednn_basic import EvidentialNet
from BA__Programmierung.config import VIZ_PATH


def evaluate_and_save_dashboard(model, dataloader, scaler_y, device, save_dir):
    model.eval()
    mu_list, v_list, alpha_list, beta_list, targets_list = [], [], [], [], []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            mu, v, alpha, beta = model(x)
            mu_list.append(mu.cpu().numpy())
            v_list.append(v.cpu().numpy())
            alpha_list.append(alpha.cpu().numpy())
            beta_list.append(beta.cpu().numpy())
            targets_list.append(y.numpy())

    mu = np.vstack(mu_list)
    v = np.vstack(v_list)
    alpha = np.vstack(alpha_list)
    beta = np.vstack(beta_list)
    targets = np.vstack(targets_list)

    # Inverse transform if needed
    if scaler_y is not None:
        mu = scaler_y.inverse_transform(mu)
        targets = scaler_y.inverse_transform(targets)
        sigma_squared = beta / (v * (alpha - 1 + 1e-6))
        sigma = np.sqrt(sigma_squared) * scaler_y.scale_[0]
    else:
        sigma = np.sqrt(beta / (v * (alpha - 1 + 1e-6)))

    r2 = r2_score(targets, mu)
    mape = mean_absolute_percentage_error(targets, mu)

    print(f"R² score: {r2:.4f}")
    print(f"MAPE: {mape * 100:.2f}%")

    os.makedirs(save_dir, exist_ok=True)

    # Plot 1: True vs Predicted
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=targets.flatten(), y=mu.flatten(), alpha=0.6)
    plt.errorbar(targets.flatten(), mu.flatten(), yerr=sigma.flatten(), fmt='o', alpha=0.2, color='gray')
    plt.plot([min(targets), max(targets)], [min(targets), max(targets)], 'r--')
    plt.xlabel("True Quality")
    plt.ylabel("Predicted Quality")
    plt.title(f"True vs. Predicted Wine Quality (R²={r2:.2f})")
    plt.grid(True)
    plt.tight_layout()
    path = os.path.join(save_dir, "true_vs_pred.png")
    plt.savefig(path)
    print(f"Saved: {path}")
    plt.close()

    # Plot 2: Residual Distribution
    residuals = mu.flatten() - targets.flatten()
    plt.figure(figsize=(8, 5))
    sns.histplot(residuals, bins=30, kde=True, color="orange")
    plt.axvline(0, color='red', linestyle='--')
    plt.title("Residual Distribution (Prediction Error)")
    plt.xlabel("Residual (Pred - True)")
    plt.grid(True)
    plt.tight_layout()
    path = os.path.join(save_dir, "residual_hist.png")
    plt.savefig(path)
    print(f"Saved: {path}")
    plt.close()

    # Plot 3: Residuals vs True
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=targets.flatten(), y=residuals, alpha=0.6)
    plt.axhline(0, linestyle="--", color="red")
    plt.title("Residuals vs True Quality")
    plt.xlabel("True Quality")
    plt.ylabel("Residual (Pred - True)")
    plt.grid(True)
    plt.tight_layout()
    path = os.path.join(save_dir, "residuals_vs_true.png")
    plt.savefig(path)
    print(f"Saved: {path}")
    plt.close()

    # Plot 4: Uncertainty vs Absolute Error
    abs_error = np.abs(mu.flatten() - targets.flatten())
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=sigma.flatten(), y=abs_error, alpha=0.5)
    plt.xlabel("Predicted Std Dev (Uncertainty)")
    plt.ylabel("Absolute Error")
    plt.title("Uncertainty vs Absolute Error")
    plt.grid(True)
    plt.tight_layout()
    path = os.path.join(save_dir, "uncertainty_vs_error.png")
    plt.savefig(path)
    print(f"Saved: {path}")
    plt.close()

    # Plot 5–8: Histograms for EDNN parameters
    hist_params = [("mu", mu), ("v", v), ("alpha", alpha), ("beta", beta)]
    for name, arr in hist_params:
        plt.figure(figsize=(7, 5))
        sns.histplot(arr.flatten(), bins=30, kde=True)
        plt.title(f"Distribution of {name}")
        plt.grid(True)
        plt.tight_layout()
        path = os.path.join(save_dir, f"dist_{name}.png")
        plt.savefig(path)
        print(f"Saved: {path}")
        plt.close()


def main():
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
    model.load_state_dict(torch.load(
        "assets/pth/ednn_regression__wine_quality_white/ednn__wine-quality-white.pt",
        map_location=device
    ))

    # Use consistent output path based on filename
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    save_dir = os.path.join(VIZ_PATH, script_name)
    evaluate_and_save_dashboard(model, loader, scaler_y, device, save_dir)


if __name__ == "__main__":
    main()
