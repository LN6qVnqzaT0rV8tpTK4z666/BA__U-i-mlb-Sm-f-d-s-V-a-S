# BA__Projekt/BA__Programmierung/viz/viz__ednn_regression__boston-housing.py
"""
Visualize predictions of a trained Evidential Deep Neural Network Ensemble on the Boston Housing dataset.
Saves multiple diagnostic plots analogous zum wine-quality-white Beispiel.
"""

import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from torch.utils.data import DataLoader
from BA__Programmierung.ml.datasets.dataset__torch__boston_housing import DatasetTorchBostonHousing
from models.model__generic_ensemble import GenericEnsembleRegressor


def load_ensemble_models(model_dir, base_config, device):
    ensemble_files = sorted([f for f in os.listdir(model_dir) if f.startswith("model_") and f.endswith(".pth")])
    ensemble = GenericEnsembleRegressor(base_config=base_config, n_models=len(ensemble_files))

    for i, model_file in enumerate(ensemble_files):
        model_path = os.path.join(model_dir, model_file)
        state_dict = torch.load(model_path, map_location=device)
        ensemble.models[i].load_state_dict(state_dict)
        ensemble.models[i].to(device)
        ensemble.models[i].eval()

    return ensemble


def predict_ensemble(ensemble, dataloader, device):
    n_models = len(ensemble.models)
    mu_list = [[] for _ in range(n_models)]
    v_list = [[] for _ in range(n_models)]
    alpha_list = [[] for _ in range(n_models)]
    beta_list = [[] for _ in range(n_models)]
    y_true_list = []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_true_list.append(y_batch.cpu())

            for i, model in enumerate(ensemble.models):
                mu, v, alpha, beta = model(X_batch)

                mu_list[i].append(mu.cpu())
                v_list[i].append(v.cpu())
                alpha_list[i].append(alpha.cpu())
                beta_list[i].append(beta.cpu())

    mu_stack = torch.stack([torch.cat(m) for m in mu_list])    # (n_models, n_samples, 1)
    v_stack = torch.stack([torch.cat(m) for m in v_list])
    alpha_stack = torch.stack([torch.cat(m) for m in alpha_list])
    beta_stack = torch.stack([torch.cat(m) for m in beta_list])
    y_true = torch.cat(y_true_list)

    return mu_stack, v_stack, alpha_stack, beta_stack, y_true


def calculate_uncertainties(mu, v, alpha, beta):
    aleatoric_var = beta / (v * (alpha - 1 + 1e-6))  # shape (n_models, n_samples)
    mu_np = mu.numpy()
    epistemic_var = np.var(mu_np, axis=0)
    mu_mean = np.mean(mu_np, axis=0)
    total_uncertainty = aleatoric_var.mean(axis=0) + epistemic_var

    return mu_mean, aleatoric_var.mean(axis=0), epistemic_var, total_uncertainty


def evaluate_and_save_dashboard_ensemble(mu_mean, aleatoric, epistemic, total_uncertainty, y_true, dataset, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    y_pred = dataset.inverse_transform_y(torch.tensor(mu_mean).unsqueeze(1)).numpy().flatten()
    y_true_np = dataset.inverse_transform_y(y_true).numpy().flatten()
    # convert total_uncertainty (NumPy array) to tensor properly:
    sigma = dataset.inverse_transform_y(
        total_uncertainty.sqrt().unsqueeze(1)
    ).numpy().flatten()

    residuals = y_pred - y_true_np
    abs_error = np.abs(residuals)

    r2 = np.corrcoef(y_true_np, y_pred)[0, 1] ** 2  # rough R² approximation
    print(f"Approximate R²: {r2:.4f}")

    # Plot 1: True vs Predicted with uncertainty
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_true_np, y=y_pred, alpha=0.6)
    plt.errorbar(y_true_np, y_pred, yerr=sigma, fmt='o', alpha=0.3, color='gray')
    plt.plot([min(y_true_np), max(y_true_np)], [min(y_true_np), max(y_true_np)], 'r--')
    plt.xlabel("True MEDV")
    plt.ylabel("Predicted MEDV")
    plt.title(f"True vs. Predicted Boston Housing (R²={r2:.2f})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "true_vs_pred.png"))
    plt.close()

    # Plot 2: Residual Distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(residuals, bins=30, kde=True, color="orange")
    plt.axvline(0, color='red', linestyle='--')
    plt.title("Residual Distribution (Prediction Error)")
    plt.xlabel("Residual (Predicted - True)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "residual_hist.png"))
    plt.close()

    # Plot 3: Residuals vs True
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_true_np, y=residuals, alpha=0.6)
    plt.axhline(0, linestyle="--", color="red")
    plt.title("Residuals vs True MEDV")
    plt.xlabel("True MEDV")
    plt.ylabel("Residual (Pred - True)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "residuals_vs_true.png"))
    plt.close()

    # Plot 4: Uncertainty vs Absolute Error
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=sigma, y=abs_error, alpha=0.5)
    plt.xlabel("Predicted Std Dev (Uncertainty)")
    plt.ylabel("Absolute Error")
    plt.title("Uncertainty vs Absolute Error")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "uncertainty_vs_error.png"))
    plt.close()

    # Plot 5–8: Histograms for aleatoric and epistemic uncertainties
    for name, arr in [
        ("aleatoric", aleatoric),
        ("epistemic", epistemic),
        ("total_uncertainty", total_uncertainty),
        ("predictions", mu_mean),
    ]:
        plt.figure(figsize=(7, 5))
        sns.histplot(arr.flatten(), bins=30, kde=True)
        plt.title(f"Distribution of {name}")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"dist_{name}.png"))
        plt.close()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_config = {
        "input_dim": 13,
        "hidden_dims": [64, 64],
        "output_type": "evidential"
    }
    model_dir = "/root/BA__U-i-mlb-Sm-f-d-s-V-a-S/BA__Projekt/assets/models/pth/ednn_regression__boston_housing"

    dataset = DatasetTorchBostonHousing(
        csv_path="assets/data/raw/dataset__boston-housing/dataset__boston-housing.csv"
    )
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

    ensemble = load_ensemble_models(model_dir, base_config, device)
    mu, v, alpha, beta, y_true = predict_ensemble(ensemble, dataloader, device)

    mu_mean, aleatoric, epistemic, total_uncertainty = calculate_uncertainties(
        mu.squeeze(-1), v.squeeze(-1), alpha.squeeze(-1), beta.squeeze(-1)
    )

    save_dir = os.path.join(
        "assets", "viz", "ednn_regression__boston_housing"
    )
    evaluate_and_save_dashboard_ensemble(mu_mean, aleatoric, epistemic, total_uncertainty, y_true, dataset, save_dir)
    print(f"Visualizations saved to: {save_dir}")


if __name__ == "__main__":
    main()
