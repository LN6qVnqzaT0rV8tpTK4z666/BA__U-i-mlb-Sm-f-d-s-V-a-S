# BA__Projekt/BA__Programmierung/viz/test__ednn_regression__wine-quality-white.py

import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from BA__Programmierung.config import VIZ_PATH
from BA__Programmierung.ml.datasets.dataset__torch__wine_quality_white import (
    WineQualityWhiteDataset,
)
from models.model__generic_ensemble import GenericEnsembleRegressor
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader


def evaluate_and_save_dashboard_ensemble(model, dataloader, scaler_y, device, save_dir):
    """
    Evaluates the ensemble model, calculates performance metrics (R², MAPE), and generates
    various visualizations including plots of residuals, true vs predicted values, and model parameters.

    This function saves the following visualizations:
    - True vs Predicted with uncertainty (error bars).
    - Residual Distribution (Prediction Error).
    - Residuals vs True Quality.
    - Uncertainty vs Absolute Error.
    - Histograms for the model parameters (`mu`, `v`, `alpha`, `beta`).

    Parameters
    ----------
    model : torch.nn.Module
        The trained ensemble model used for making predictions.
    dataloader : torch.utils.data.DataLoader
        DataLoader providing the input data for evaluation.
    scaler_y : sklearn.preprocessing.StandardScaler
        Scaler used to reverse transform the target variable.
    device : torch.device
        The device (CPU or GPU) to run the model on.
    save_dir : str
        Directory path where the plots will be saved.

    Returns
    -------
    None
    """
    model.eval()
    mu_list, v_list, alpha_list, beta_list, targets_list = [], [], [], [], []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            outputs = model(x)
            mu, v, alpha, beta = outputs

            # If the model is an ensemble (outputs have an extra dimension)
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

    r2 = r2_score(targets, mu)
    mape = mean_absolute_percentage_error(targets, mu)

    print(f"R² score: {r2:.4f}")
    print(f"MAPE: {mape * 100:.2f}%")

    os.makedirs(save_dir, exist_ok=True)

    # === Plot 1: True vs Predicted with uncertainty
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=targets.flatten(), y=mu.flatten(), alpha=0.6)
    plt.errorbar(targets.flatten(), mu.flatten(), yerr=sigma.flatten(), fmt='o', alpha=0.2, color='gray')
    plt.plot([min(targets), max(targets)], [min(targets), max(targets)], 'r--')
    plt.xlabel("True Quality")
    plt.ylabel("Predicted Quality")
    plt.title(f"True vs. Predicted Wine Quality (R²={r2:.2f})")
    plt.grid(True)
    plt.tight_layout()
    filename = "true_vs_pred.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath)
    plt.close()
    print(f"Saved plot: {filepath}")

    # === Plot 2: Residual Distribution
    residuals = mu.flatten() - targets.flatten()
    plt.figure(figsize=(8, 5))
    sns.histplot(residuals, bins=30, kde=True, color="orange")
    plt.axvline(0, color='red', linestyle='--')
    plt.title("Residual Distribution (Prediction Error)")
    plt.xlabel("Residual (Pred - True)")
    plt.grid(True)
    plt.tight_layout()
    filename = "residual_hist.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath)
    plt.close()
    print(f"Saved plot: {filepath}")

    # === Plot 3: Residuals vs True
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=targets.flatten(), y=residuals, alpha=0.6)
    plt.axhline(0, linestyle="--", color="red")
    plt.title("Residuals vs True Quality")
    plt.xlabel("True Quality")
    plt.ylabel("Residual (Pred - True)")
    plt.grid(True)
    plt.tight_layout()
    filename = "residuals_vs_true.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath)
    plt.close()
    print(f"Saved plot: {filepath}")

    # === Plot 4: Uncertainty vs Absolute Error
    abs_error = np.abs(residuals)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=sigma.flatten(), y=abs_error, alpha=0.5)
    plt.xlabel("Predicted Std Dev (Uncertainty)")
    plt.ylabel("Absolute Error")
    plt.title("Uncertainty vs Absolute Error")
    plt.grid(True)
    plt.tight_layout()
    filename = "uncertainty_vs_error.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath)
    plt.close()
    print(f"Saved plot: {filepath}")

    # === Plot 5–8: Histograms for EDNN parameters
    for name, arr in [("mu", mu), ("v", v), ("alpha", alpha), ("beta", beta)]:
        plt.figure(figsize=(7, 5))
        sns.histplot(arr.flatten(), bins=30, kde=True)
        plt.title(f"Distribution of {name}")
        plt.grid(True)
        plt.tight_layout()
        filename = f"dist_{name}.png"
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath)
        plt.close()
        print(f"Saved plot: {filepath}")


def main():
    """
    Main function for loading the Wine Quality White dataset, training the ensemble model,
    evaluating it, and saving the results as visualizations.

    The function:
    1. Loads the Wine Quality White dataset from a CSV file.
    2. Preprocesses the data (standard scaling).
    3. Defines the model configuration and loads the pre-trained model.
    4. Evaluates the model and saves the visualizations (e.g., True vs Predicted, Residuals).

    Parameters
    ----------
    None

    Returns
    -------
    None

    Example
    -------
    >>> main()  # This will execute the entire evaluation and visualization process
    """
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

    model_path = "assets/models/pth/ednn_regression__wine_quality_white_ensemble/generic_ensemble__wine-quality-white.pt"
    if os.path.isfile(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"Model file not found at '{model_path}'. Initializing and saving new model.")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(model.state_dict(), model_path)

    save_dir = os.path.join(VIZ_PATH, "ednn_regression__wine_quality_white")
    evaluate_and_save_dashboard_ensemble(model, loader, scaler_y, device, save_dir)


if __name__ == "__main__":
    main()

