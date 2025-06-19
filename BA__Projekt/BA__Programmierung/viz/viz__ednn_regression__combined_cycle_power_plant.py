# BA__Projekt/BA__Programmierung/viz/viz__ednn_regression__combined_cycle_power_plant.py

import os
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from torch.utils.data import DataLoader

from BA__Programmierung.ml.datasets.dataset__torch__combined_cycle_power_plant import DatasetTorchCombinedCyclePowerPlant
from models.model__generic_ensemble import GenericEnsembleRegressor


def load_ensemble(model_path, base_config, device):
    model = GenericEnsembleRegressor(base_config=base_config, n_models=5).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def predict_ensemble(model, dataloader, device):
    mu_list, v_list, alpha_list, beta_list, targets_list = [], [], [], [], []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            outputs = model(x)
            mu, v, alpha, beta = outputs

            if mu.dim() == 3:  # Ensemble dimension
                mu = mu.mean(dim=0)
                v = v.mean(dim=0)
                alpha = alpha.mean(dim=0)
                beta = beta.mean(dim=0)

            mu_list.append(mu.cpu().numpy())
            v_list.append(v.cpu().numpy())
            alpha_list.append(alpha.cpu().numpy())
            beta_list.append(beta.cpu().numpy())
            targets_list.append(y.numpy())

    return (
        np.vstack(mu_list),
        np.vstack(v_list),
        np.vstack(alpha_list),
        np.vstack(beta_list),
        np.vstack(targets_list),
    )


def evaluate_and_save_dashboard(mu, v, alpha, beta, targets, scaler_y, save_dir):
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

    print(f"R²: {r2:.4f}")
    print(f"MAPE: {mape * 100:.2f}%")

    os.makedirs(save_dir, exist_ok=True)

    def save_plot(fig, name):
        filepath = os.path.join(save_dir, name)
        fig.savefig(filepath)
        plt.close(fig)
        print(f"Saved plot: {filepath}")

    # Plot 1: True vs Predicted with uncertainty
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=targets.flatten(), y=mu.flatten(), alpha=0.6, ax=ax)
    ax.errorbar(targets.flatten(), mu.flatten(), yerr=sigma.flatten(), fmt='o', alpha=0.3, color='gray')
    ax.plot([min(targets), max(targets)], [min(targets), max(targets)], 'r--')
    ax.set_title("True vs Predicted Power Output")
    ax.set_xlabel("True Power Output")
    ax.set_ylabel("Predicted Power Output")
    ax.grid(True)
    save_plot(fig, "true_vs_pred.png")

    # Plot 2: Residual Histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(residuals, bins=30, kde=True, color='orange', ax=ax)
    ax.axvline(0, color='red', linestyle='--')
    ax.set_title("Residual Distribution")
    ax.set_xlabel("Residual (Predicted - True)")
    save_plot(fig, "residual_hist.png")

    # Plot 3: Residuals vs True
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=targets.flatten(), y=residuals, alpha=0.6, ax=ax)
    ax.axhline(0, color='red', linestyle='--')
    ax.set_title("Residuals vs True Power Output")
    ax.set_xlabel("True Power Output")
    ax.set_ylabel("Residual (Pred - True)")
    save_plot(fig, "residuals_vs_true.png")

    # Plot 4: Uncertainty vs Absolute Error
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=sigma.flatten(), y=abs_error, alpha=0.5, ax=ax)
    ax.set_title("Uncertainty vs Absolute Error")
    ax.set_xlabel("Predicted Std Dev")
    ax.set_ylabel("Absolute Error")
    save_plot(fig, "uncertainty_vs_error.png")

    # Plot 5–8: Parameter Distributions
    for name, arr in [("mu", mu), ("v", v), ("alpha", alpha), ("beta", beta)]:
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.histplot(arr.flatten(), bins=30, kde=True, ax=ax)
        ax.set_title(f"Distribution of {name}")
        save_plot(fig, f"dist_{name}.png")


def main():
    csv_path = "assets/data/raw/dataset__combined-cycle-power-plant/dataset__combined-cycle-power-plant.csv"
    model_path = "assets/models/pth/ednn_regression__combined_cycle_power_plant/ednn_regression__combined_cycle_power_plant.pth"
    save_dir = "assets/viz/ednn_regression__combined_cycle_power_plant"

    dataset = DatasetTorchCombinedCyclePowerPlant(csv_path)
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

    model = load_ensemble(model_path, base_config, device)
    mu, v, alpha, beta, targets = predict_ensemble(model, loader, device)
    evaluate_and_save_dashboard(mu, v, alpha, beta, targets, scaler_y, save_dir)


if __name__ == "__main__":
    main()

