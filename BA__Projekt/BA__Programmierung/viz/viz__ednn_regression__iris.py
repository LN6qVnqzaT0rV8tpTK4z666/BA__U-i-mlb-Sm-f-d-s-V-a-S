# BA__Projekt/BA__Programmierung/viz/vis__ednn_regression__iris.py

import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import torch

from BA__Programmierung.config import VIZ_PATH, DATA_DIR__PROCESSED
from BA__Programmierung.ml.ednn_regression__iris import EvidentialNet
from BA__Programmierung.ml.datasets.dataset__torch_duckdb_iris import DatasetTorchDuckDBIris
from sklearn.decomposition import PCA
from tensorboard.backend.event_processing import event_accumulator


def plot_loss_curves():
    # ============
    # Visualization
    # ============

    # === Dynamisches Logverzeichnis finden ===
    processed_root = DATA_DIR__PROCESSED
    log_folders = sorted(
        glob.glob(os.path.join(processed_root, "ednn_iris_*")),
        key=os.path.getmtime,
        reverse=True,
    )

    if not log_folders:
        raise FileNotFoundError(
            "No log directory found. Please run training first."
        )

    log_dir = log_folders[0]
    print(f"Verwende Log-Verzeichnis: {log_dir}")

    # === Zielverzeichnis für Visualisierung ===
    SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]
    SCRIPT_VIZ_DIR = os.path.join(VIZ_PATH, SCRIPT_NAME)
    os.makedirs(SCRIPT_VIZ_DIR, exist_ok=True)
    output_path = os.path.join(SCRIPT_VIZ_DIR, "loss_curve.png")

    # === TensorBoard-Logs einlesen ===
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()

    # Verfügbare Tags anzeigen (optional)
    print("Verfügbare Scalar-Tags:", ea.Tags()["scalars"])

    # Daten extrahieren
    train_scalars = ea.Scalars("Loss/train")
    val_scalars = (
        ea.Scalars("Loss/val") if "Loss/val" in ea.Tags()["scalars"] else []
    )

    train_steps = [e.step for e in train_scalars]
    train_vals = [e.value for e in train_scalars]

    plt.figure(figsize=(10, 6))
    plt.plot(train_steps, train_vals, label="Train Loss", color="blue")

    if val_scalars:
        val_steps = [e.step for e in val_scalars]
        val_vals = [e.value for e in val_scalars]
        plt.plot(val_steps, val_vals, label="Val Loss", color="orange")

    plt.title("Train & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # === Bild speichern ===
    plt.savefig(output_path)
    print(f"Plot gespeichert unter: {output_path}")


def evidential_dashboard(model, dataset, device="cpu"):
    model.eval()
    X = dataset.X.to(device)
    y = dataset.y.to(device)

    with torch.no_grad():
        mu, v, alpha, beta = model(X)

    mu = mu.squeeze().cpu().numpy()
    y_true = y.squeeze().cpu().numpy()
    v = v.squeeze().cpu().numpy()
    alpha = alpha.squeeze().cpu().numpy()
    beta = beta.squeeze().cpu().numpy()

    aleatoric = beta / (alpha - 1 + 1e-6)
    epistemic = beta / ((v + 1e-6) * (alpha - 1 + 1e-6))  # Optional

    fig, axs = plt.subplots(3, 2, figsize=(14, 14))
    fig.suptitle("📊 Evidential Regression Dashboard", fontsize=16)

    axs[0, 0].scatter(mu, y_true, alpha=0.7, label="Prediction")
    axs[0, 0].errorbar(
        mu, y_true, yerr=aleatoric, fmt="o", alpha=0.2, label="Uncertainty"
    )
    axs[0, 0].plot(
        [min(y_true), max(y_true)], [min(y_true), max(y_true)], "r--"
    )
    axs[0, 0].set_xlabel("Predicted μ")
    axs[0, 0].set_ylabel("True y")
    axs[0, 0].set_title("μ vs True y ± Uncertainty")
    axs[0, 0].legend()
    axs[0, 0].grid()

    axs[0, 1].hist(mu, bins=30, color="cornflowerblue")
    axs[0, 1].set_title("Distribution of μ")
    axs[0, 1].grid()

    axs[1, 0].hist(v, bins=30, color="orange")
    axs[1, 0].set_title("Distribution of v (inverse variance)")
    axs[1, 0].grid()

    axs[1, 1].hist(alpha, bins=30, color="mediumseagreen")
    axs[1, 1].set_title("Distribution of α")
    axs[1, 1].grid()

    axs[2, 0].hist(beta, bins=30, color="slateblue")
    axs[2, 0].set_title("Distribution of β")
    axs[2, 0].grid()

    X_pca = PCA(n_components=2).fit_transform(X.cpu())
    sc = axs[2, 1].scatter(
        X_pca[:, 0], X_pca[:, 1], c=aleatoric, cmap="viridis", alpha=0.8
    )
    axs[2, 1].set_title("PCA Projection with Aleatoric Uncertainty")
    axs[2, 1].set_xlabel("PCA 1")
    axs[2, 1].set_ylabel("PCA 2")
    fig.colorbar(sc, ax=axs[2, 1], label="Uncertainty")

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]
    SCRIPT_VIZ_DIR = os.path.join(VIZ_PATH, SCRIPT_NAME)
    os.makedirs(SCRIPT_VIZ_DIR, exist_ok=True)
    output_path = os.path.join(SCRIPT_VIZ_DIR, "evidential_dashboard.png")
    plt.savefig(output_path)
    print(f"Dashboard gespeichert unter: {output_path}")


def main():
    dataset = DatasetTorchDuckDBIris(
        db_path="/root/BA__U-i-mlb-Sm-f-d-s-V-a-S/BA__Projekt/assets/dbs/dataset__iris__dataset.duckdb",
        table_name="iris__dataset_csv"  # ggf. anpassen
    )
    device = "cpu"
    model = EvidentialNet(input_dim=4)
    plot_loss_curves()
    evidential_dashboard(model, dataset)


if __name__ == "__main__":
    main()
