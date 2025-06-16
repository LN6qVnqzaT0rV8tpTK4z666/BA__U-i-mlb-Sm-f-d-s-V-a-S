# BA__Projekt/BA__Programmierung/viz/viz__ednn_regression__condition_based_maintenance_of_naval_propulsion_plants.py

import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from torch.utils.data import DataLoader

from BA__Programmierung.ml.datasets.dataset__torch__condition_based_maintenance_of_naval_propulsion_plants import NavalPropulsionDataset
from models.model__generic_ensemble import GenericEnsembleRegressor


def load_model_ensemble(model_path, base_config, device, n_models=5):
    """Load the trained ensemble model from checkpoint."""
    model = GenericEnsembleRegressor(base_config=base_config, n_models=n_models).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def predict_ensemble(model, dataloader, device):
    """Get ensemble predictions and parameters for the dataset."""
    mu_list, v_list, alpha_list, beta_list, targets_list = [], [], [], [], []

    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            mu, v, alpha, beta = model(x)

            # Average over ensemble models if 3D output
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

    return (
        np.concatenate(mu_list, axis=0),
        np.concatenate(v_list, axis=0),
        np.concatenate(alpha_list, axis=0),
        np.concatenate(beta_list, axis=0),
        np.concatenate(targets_list, axis=0),
    )


def evaluate_and_save_dashboard(
    mu, v, alpha, beta, targets, scaler_y, save_dir, dataset_name="Naval Propulsion"
):
    """Generate evaluation metrics and save a dashboard of plots."""
    # Inverse transform targets and predictions if scaler provided
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

    print(f"{dataset_name} Evaluation:")
    print(f"  R²: {r2:.4f}")
    print(f"  MAPE: {mape * 100:.2f}%")

    os.makedirs(save_dir, exist_ok=True)

    def save_plot(fig, filename):
        path = os.path.join(save_dir, filename)
        fig.savefig(path)
        plt.close(fig)
        print(f"Saved plot: {path}")

    # Plot: True vs Predicted with Uncertainty
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=targets.flatten(), y=mu.flatten(), alpha=0.6, ax=ax)
    ax.errorbar(targets.flatten(), mu.flatten(), yerr=sigma.flatten(), fmt='o', alpha=0.3, color='gray')
    t_flat = targets.flatten()
    min_t = t_flat.min()
    max_t = t_flat.max()
    ax.plot([min_t, max_t], [min_t, max_t], 'r--')
    ax.set_title(f"{dataset_name}: True vs Predicted")
    ax.set_xlabel("True Power Output")
    ax.set_ylabel("Predicted Power Output")
    ax.grid(True)
    save_plot(fig, "true_vs_pred.png")

    # Plot: Residual Histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(residuals, bins=30, kde=True, color='orange', ax=ax)
    ax.axvline(0, color='red', linestyle='--')
    ax.set_title(f"{dataset_name}: Residual Distribution")
    ax.set_xlabel("Residual (Predicted - True)")
    save_plot(fig, "residual_hist.png")

    # Plot: Residuals vs True
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=targets.flatten(), y=residuals, alpha=0.6, ax=ax)
    ax.axhline(0, color='red', linestyle='--')
    ax.set_title(f"{dataset_name}: Residuals vs True Output")
    ax.set_xlabel("True Power Output")
    ax.set_ylabel("Residual (Predicted - True)")
    save_plot(fig, "residuals_vs_true.png")

    # Plot: Uncertainty vs Absolute Error
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=sigma.flatten(), y=abs_error, alpha=0.5, ax=ax)
    ax.set_title(f"{dataset_name}: Uncertainty vs Absolute Error")
    ax.set_xlabel("Predicted Std Dev")
    ax.set_ylabel("Absolute Error")
    save_plot(fig, "uncertainty_vs_error.png")

    # Plots: Parameter Distributions
    for name, arr in [("mu", mu), ("v", v), ("alpha", alpha), ("beta", beta)]:
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.histplot(arr.flatten(), bins=30, kde=True, ax=ax)
        ax.set_title(f"{dataset_name}: Distribution of {name}")
        save_plot(fig, f"dist_{name}.png")


def main():
    csv_path = "assets/data/raw/dataset__condition-based-maintenance-of-naval-propulsion-plants/data.csv"
    model_path = "assets/models/pth/ednn_regression__condition_based_maintenance_of_naval_propulsion_plants/generic_ensemble__cbm.pth"
    save_dir = "assets/viz/ednn_regression__condition_based_maintenance_of_naval_propulsion_plants"
    batch_size = 64

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = NavalPropulsionDataset(csv_path, normalize=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

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
        "output_dim": 2
    }

    model = load_model_ensemble(model_path, base_config, device)

    mu, v, alpha, beta, targets = predict_ensemble(model, dataloader, device)

    # scaler_y is None here, but if you add scaler support, pass it in
    evaluate_and_save_dashboard(mu, v, alpha, beta, targets, scaler_y=None, save_dir=save_dir)


if __name__ == "__main__":
    main()
