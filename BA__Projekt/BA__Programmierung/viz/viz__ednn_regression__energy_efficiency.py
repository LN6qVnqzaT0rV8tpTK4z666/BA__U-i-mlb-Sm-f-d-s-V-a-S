# BA__Projekt/BA__Programmierung/viz/viz__ednn_regression__energy-efficiency.py

import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from BA__Programmierung.ml.datasets.dataset__torch__energy_efficiency import EnergyEfficiencyDataset
from models.model__generic_ensemble import GenericEnsembleRegressor
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import random_split, DataLoader


def evaluate_and_visualize_and_save(model, dataloader, scaler_y, device, save_dir):
    """
    Evaluates the given model and visualizes the results by generating and saving plots.
    
    This function performs the following:
    1. Evaluates the model on the provided dataloader.
    2. Calculates R² and Mean Absolute Percentage Error (MAPE).
    3. Creates three types of plots:
        - True vs Predicted values scatter plot
        - Residual distribution (prediction error)
        - Residuals vs True values scatter plot
    4. Saves all the generated plots in the specified directory.

    :param model: The model to be evaluated.
    :param dataloader: DataLoader providing the test dataset.
    :param scaler_y: StandardScaler instance used for scaling the target variable.
    :param device: The device to perform the evaluation on (e.g., 'cuda' or 'cpu').
    :param save_dir: Directory where the plots will be saved.
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            output = model(x)
            preds = output[0].cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(y.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    if scaler_y is not None:
        all_preds = scaler_y.inverse_transform(all_preds)
        all_targets = scaler_y.inverse_transform(all_targets)

    r2 = r2_score(all_targets, all_preds)
    mape = mean_absolute_percentage_error(all_targets, all_preds)

    print(f"R² score: {r2:.4f}")
    print(f"MAPE: {mape*100:.2f}%")

    # Plot 1: True vs Predicted
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=all_targets.flatten(), y=all_preds.flatten(), alpha=0.6)
    plt.plot([min(all_targets), max(all_targets)], [min(all_targets), max(all_targets)], 'r--')
    plt.xlabel("True Target")
    plt.ylabel("Predicted Target")
    plt.title(f"True vs. Predicted Targets (R²={r2:.2f})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "true_vs_pred.png"))
    plt.close()

    # Plot 2: Residual Distribution
    residuals = all_preds.flatten() - all_targets.flatten()
    plt.figure(figsize=(8, 5))
    sns.histplot(residuals, bins=30, kde=True, color="orange")
    plt.axvline(0, color='red', linestyle='--')
    plt.title("Residual Distribution (Prediction Error)")
    plt.xlabel("Residual (Pred - True)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "residual_hist.png"))
    plt.close()

    # Plot 3: Residuals vs True
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=all_targets.flatten(), y=residuals, alpha=0.6)
    plt.axhline(0, linestyle="--", color="red")
    plt.title("Residuals vs True Target")
    plt.xlabel("True Target")
    plt.ylabel("Residual (Pred - True)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "residuals_vs_true.png"))
    plt.close()

    print(f"Plots saved in {save_dir}")


def main():
    """
    Main function to load the Energy Efficiency dataset, train a regression model,
    and evaluate it using the `evaluate_and_visualize_and_save` function.

    This function performs the following steps:
    1. Loads the Energy Efficiency dataset.
    2. Scales the features and targets.
    3. Evaluates the model on the test data and generates plots.
    """
    dataset_path = "assets/data/raw/dataset__energy-efficiency/dataset__energy-efficiency.csv"
    full_dataset = EnergyEfficiencyDataset(dataset_path)

    # Train-Test Split
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )

    def extract_xy(dataset):
        """
        Extracts input features (X) and target variables (Y) from a dataset.

        :param dataset: A torch dataset containing (input, target) pairs.
        :return: A tuple of numpy arrays (X, Y).
        """
        X = torch.stack([item[0] for item in dataset])
        Y = torch.stack([item[1] for item in dataset])
        return X.numpy(), Y.numpy()

    # Extract and scale data
    X_train, Y_train = extract_xy(train_dataset)
    X_test, Y_test = extract_xy(test_dataset)

    scaler_x = StandardScaler().fit(X_train)
    scaler_y = StandardScaler().fit(Y_train)

    X_test_scaled = torch.tensor(scaler_x.transform(X_test), dtype=torch.float32)
    Y_test_scaled = torch.tensor(scaler_y.transform(Y_test), dtype=torch.float32)

    test_loader = DataLoader(torch.utils.data.TensorDataset(X_test_scaled, Y_test_scaled), batch_size=64, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = X_test_scaled.shape[1]

    # Model Configuration
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

    # Load model
    model = GenericEnsembleRegressor(base_config=base_config, n_models=5).to(device)
    model.load_state_dict(torch.load("/root/BA__Projekt/assets/models/pth/ednn_regression__energy_efficiency_ensemble.pth", map_location=device))

    # Evaluate and visualize
    evaluate_and_visualize_and_save(model, test_loader, scaler_y, device, "assets/viz/ednn_regression__energy_efficiency_ensemble/")

if __name__ == "__main__":
    main()

