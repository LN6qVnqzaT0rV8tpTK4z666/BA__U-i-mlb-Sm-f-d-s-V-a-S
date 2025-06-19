# BA__Projekt/BA__Programmierung/viz/viz__ednn_regression__iris.py

import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from BA__Programmierung.config import VIZ_PATH
from BA__Programmierung.ml.datasets.dataset__torch__duckdb_iris import DatasetTorchDuckDBIris
from models.model__generic_ensemble import GenericEnsembleRegressor
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler


def evaluate_and_save_dashboard_ensemble(model, dataloader, device, save_dir, scaler_y=None):
    """
    Evaluates a generic ensemble EDNN regression model and saves diagnostic visualizations.

    The function computes:
    - R² and MAPE scores
    - Predicted mean and uncertainty (σ)
    - Residual diagnostics
    - Parameter distributions
    - PCA plot colored by aleatoric uncertainty

    It then saves the following plots to `save_dir`:
    - true_vs_pred.png
    - residual_hist.png
    - residuals_vs_true.png
    - uncertainty_vs_error.png
    - dist_mu.png, dist_v.png, dist_alpha.png, dist_beta.png
    - pca_aleatoric.png

    Args:
        model (GenericEnsembleRegressor): Trained ensemble regression model.
        dataloader (DataLoader): Torch dataloader providing test data.
        device (torch.device): Computation device (CPU or CUDA).
        save_dir (str): Path to save all generated plots.
        scaler_y (StandardScaler, optional): Target scaler for inverse transformation.

    Returns:
        None
    """
    model.eval()
    mu_list, v_list, alpha_list, beta_list, targets_list = [], [], [], [], []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            mu, v, alpha, beta = model(x)

            # Average across ensemble members if applicable
            if mu.dim() == 3:
                mu = mu.mean(dim=0)
                v = v.mean(dim=0)
                alpha = alpha.mean(dim=0)
                beta = beta.mean(dim=0)

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

    if scaler_y is not None:
        mu = scaler_y.inverse_transform(mu)
        targets = scaler_y.inverse_transform(targets)
        sigma_squared = beta / (v * (alpha - 1 + 1e-6))
        sigma = np.sqrt(sigma_squared) * scaler_y.scale_[0]
    else:
        sigma = np.sqrt(beta / (v * (alpha - 1 + 1e-6)))

    residuals = mu.flatten() - targets.flatten()
    abs_error = np.abs(residuals)
    r2 = r2_score(targets, mu)
    mape = mean_absolute_percentage_error(targets, mu)

    print(f"R² score: {r2:.4f}")
    print(f"MAPE: {mape * 100:.2f}%")

    os.makedirs(save_dir, exist_ok=True)

    # 1. True vs Predicted
    plt.figure(figsize=(8, 6))
    plt.errorbar(targets.flatten(), mu.flatten(), yerr=sigma.flatten(), fmt="o", alpha=0.3)
    sns.scatterplot(x=targets.flatten(), y=mu.flatten(), alpha=0.6)
    plt.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--')
    plt.title("Predicted μ vs True y ± Uncertainty")
    plt.xlabel("True y")
    plt.ylabel("Predicted μ")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "true_vs_pred.png"))
    plt.close()

    # 2. Residual Histogram
    plt.figure(figsize=(7, 5))
    sns.histplot(residuals, bins=30, kde=True)
    plt.title("Residual Distribution")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "residual_hist.png"))
    plt.close()

    # 3. Residuals vs True
    plt.figure(figsize=(7, 5))
    sns.scatterplot(x=targets.flatten(), y=residuals, alpha=0.5)
    plt.axhline(0, linestyle="--", color="red")
    plt.title("Residuals vs True y")
    plt.xlabel("True y")
    plt.ylabel("Residual")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "residuals_vs_true.png"))
    plt.close()

    # 4. Uncertainty vs Absolute Error
    plt.figure(figsize=(7, 5))
    sns.scatterplot(x=sigma.flatten(), y=abs_error, alpha=0.5)
    plt.title("Uncertainty vs Absolute Error")
    plt.xlabel("Predicted Std Dev")
    plt.ylabel("Absolute Error")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "uncertainty_vs_error.png"))
    plt.close()

    # 5-8. Parameter Distributions
    for name, arr in [("mu", mu), ("v", v), ("alpha", alpha), ("beta", beta)]:
        plt.figure(figsize=(7, 5))
        sns.histplot(arr.flatten(), bins=30, kde=True)
        plt.title(f"Distribution of {name}")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"dist_{name}.png"))
        plt.close()

    # 9. PCA Plot with aleatoric uncertainty
    aleatoric = beta / (alpha - 1 + 1e-6)
    dataset_X = np.vstack([x.numpy() for x, _ in dataloader])
    pca_proj = PCA(n_components=2).fit_transform(dataset_X)

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(pca_proj[:, 0], pca_proj[:, 1], c=aleatoric.flatten(), cmap="viridis", alpha=0.8)
    plt.colorbar(sc, label="Aleatoric Uncertainty")
    plt.title("PCA of Input Colored by Aleatoric Uncertainty")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "pca_aleatoric.png"))
    plt.close()


def main():
    dataset = DatasetTorchDuckDBIris(
        db_path="/root/BA__U-i-mlb-Sm-f-d-s-V-a-S/BA__Projekt/assets/dbs/dataset__iris-dataset.duckdb",
        table_name="iris_dataset_csv"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = dataset.X.numpy()
    y = dataset.y.numpy().reshape(-1, 1)

    scaler_x = StandardScaler().fit(X)
    scaler_y = StandardScaler().fit(y)

    X_scaled = torch.tensor(scaler_x.transform(X), dtype=torch.float32)
    y_scaled = torch.tensor(scaler_y.transform(y), dtype=torch.float32)

    full_dataset = torch.utils.data.TensorDataset(X_scaled, y_scaled)
    dataloader = DataLoader(full_dataset, batch_size=64, shuffle=False)

    base_config = {
        "input_dim": 4,  # Iris features count
        "hidden_dims": [64, 64],
        "output_type": "evidential",
        "use_dropout": False,
        "dropout_p": 0.2,
        "flatten_input": False,
        "use_batchnorm": False,
        "activation_name": "relu",
    }

    model = GenericEnsembleRegressor(base_config=base_config, n_models=5).to(device)
    # model.load_state_dict(torch.load("path_to_trained_model.pth"))  # Optional

    save_dir = os.path.join(VIZ_PATH, "viz__ednn_regression__iris")
    evaluate_and_save_dashboard_ensemble(model, dataloader, device=device, save_dir=save_dir, scaler_y=scaler_y)


if __name__ == "__main__":
    main()

