# BA__Projekt/BA__Programmierung/viz/viz__ednn_regression__nmavani__func1.py
import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from BA__Programmierung.config import VIZ_PATH
from BA__Programmierung.ml.datasets.dataset__torch__nmavani_func1 import DatasetTorchDuckDBFunc1
from models.model__generic_ensemble import GenericEnsembleRegressor
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


def evaluate_and_visualize(model, dataloader, scaler_y, save_dir, device):
    """
    Evaluates the model, calculates various performance metrics, and visualizes the results with multiple plots.

    This function generates and saves the following plots:
    - True vs Predicted plot with uncertainty (error bars).
    - Residual histogram (distribution of prediction errors).
    - Residuals vs True target plot.
    - Uncertainty vs Absolute error plot.
    - Histograms for model parameters (`mu`, `v`, `alpha`, `beta`).

    Parameters
    ----------
    model : torch.nn.Module
        The trained ensemble model used for predictions.
    dataloader : torch.utils.data.DataLoader
        DataLoader containing the input data to evaluate the model.
    scaler_y : sklearn.preprocessing.StandardScaler
        Scaler used to inverse transform the predicted and true values of the target variable.
    save_dir : str
        Directory path where the plots will be saved.
    device : torch.device
        The device (CPU or GPU) to run the model on.

    Returns
    -------
    None
    """
    model.eval()
    mu_list, v_list, alpha_list, beta_list, y_list = [], [], [], [], []

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
            y_list.append(y.numpy())

    mu = np.vstack(mu_list)
    v = np.vstack(v_list)
    alpha = np.vstack(alpha_list)
    beta = np.vstack(beta_list)
    y_true = np.vstack(y_list)

    if scaler_y is not None:
        mu = scaler_y.inverse_transform(mu)
        y_true = scaler_y.inverse_transform(y_true)
        sigma_squared = beta / (v * (alpha - 1 + 1e-6))
        sigma = np.sqrt(sigma_squared) * scaler_y.scale_[0]
    else:
        sigma = np.sqrt(beta / (v * (alpha - 1 + 1e-6)))

    r2 = r2_score(y_true, mu)
    mape = mean_absolute_percentage_error(y_true, mu)

    print(f"R²: {r2:.4f}, MAPE: {mape * 100:.2f}%")
    os.makedirs(save_dir, exist_ok=True)

    # Plot 1: True vs Predicted with uncertainty
    plt.figure(figsize=(8, 6))
    plt.errorbar(y_true.flatten(), mu.flatten(), yerr=sigma.flatten(), fmt='o', alpha=0.3, color='gray')
    sns.scatterplot(x=y_true.flatten(), y=mu.flatten(), alpha=0.7)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(f"True vs Predicted (R²={r2:.2f})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "true_vs_pred.png"))
    plt.close()

    # Plot 2: Residuals Histogram
    residuals = mu.flatten() - y_true.flatten()
    plt.figure(figsize=(8, 5))
    sns.histplot(residuals, bins=30, kde=True, color="orange")
    plt.axvline(0, color='red', linestyle='--')
    plt.title("Residual Distribution")
    plt.xlabel("Residual (Pred - True)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "residual_hist.png"))
    plt.close()

    # Plot 3: Residuals vs True
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_true.flatten(), y=residuals, alpha=0.6)
    plt.axhline(0, linestyle="--", color="red")
    plt.title("Residuals vs True")
    plt.xlabel("True")
    plt.ylabel("Residual")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "residuals_vs_true.png"))
    plt.close()

    # Plot 4: Uncertainty vs Absolute Error
    abs_error = np.abs(residuals)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=sigma.flatten(), y=abs_error, alpha=0.5)
    plt.xlabel("Predicted Std Dev")
    plt.ylabel("Absolute Error")
    plt.title("Uncertainty vs Absolute Error")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "uncertainty_vs_error.png"))
    plt.close()

    # Plot 5–8: EDNN parameters
    for name, arr in [("mu", mu), ("v", v), ("alpha", alpha), ("beta", beta)]:
        plt.figure(figsize=(7, 5))
        sns.histplot(arr.flatten(), bins=30, kde=True)
        plt.title(f"Distribution of {name}")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"dist_{name}.png"))
        plt.close()


def main():
    """
    Main function for loading the dataset, training the model, and evaluating it with visualization.

    This function:
    1. Loads the dataset from a DuckDB database.
    2. Scales the input features and target.
    3. Defines the ensemble model configuration.
    4. Loads the pre-trained model checkpoint.
    5. Evaluates the model and saves various visualizations (e.g., True vs Predicted, Residual Distribution).

    Parameters
    ----------
    None

    Returns
    -------
    None

    Example
    -------
    >>> main()  # This will run the entire evaluation and visualization pipeline
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Dataset
    dataset = DatasetTorchDuckDBFunc1(
        db_path="assets/dbs/dataset__generated__nmavani__func_1.duckdb",
        table_name="generated_nmavani_func_1_csv"
    )
    X = dataset.X.numpy()
    Y = dataset.y.numpy().reshape(-1, 1)

    scaler_x = StandardScaler().fit(X)
    scaler_y = StandardScaler().fit(Y)

    X_scaled = torch.tensor(scaler_x.transform(X), dtype=torch.float32)
    Y_scaled = torch.tensor(scaler_y.transform(Y), dtype=torch.float32)

    loader = DataLoader(TensorDataset(X_scaled, Y_scaled), batch_size=64, shuffle=False)

    # Build Model
    input_dim = X_scaled.shape[1]
    model = GenericEnsembleRegressor(
        base_config={
            "input_dim": input_dim,
            "hidden_dims": [64, 64],
            "output_type": "evidential",
            "activation_name": "relu",
            "use_dropout": False,
        },
        n_models=5
    ).to(device)

    # Load Checkpoint
    ckpt_path = "assets/models/pth/ednn_regression__nmavani_func1/generic_ensemble__nmavani_func1.pt"
    if os.path.isfile(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"[✔] Loaded checkpoint: {ckpt_path}")
    else:
        print(f"[⚠] No model found at: {ckpt_path}")
        return

    # Run Evaluation + Save Visuals
    save_dir = os.path.join(VIZ_PATH, "ednn_regression__nmavani_func1")
    evaluate_and_visualize(model, loader, scaler_y, save_dir, device)


if __name__ == "__main__":
    main()

