# BA__Projekt/BA__Programmierung/ml/ednn_regression__iris.py

import os
from datetime import datetime

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from BA__Programmierung.ml.datasets.dataset__torch__duckdb_iris import (
    DatasetTorchDuckDBIris,
)
from BA__Programmierung.ml.losses.evidential_loss import evidential_loss
from models.model__ednn_deep import EvidentialNetDeep as EvidentialNet


def train(model, train_loader, val_loader, epochs=100, lr=1e-3, device="cpu"):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # === TensorBoard Writer ===
    log_dir = os.path.join(
        "/root/BA__U-i-mlb-Sm-f-d-s-V-a-S/BA__Projekt/assets/data/processed",
        "ednn_iris_" + datetime.now().strftime("%Y%m%d-%H%M%S"),
    )
    writer = SummaryWriter(log_dir=log_dir)

    model.to(device)

    best_val_loss = float("inf")
    epochs_no_improve = 0
    patience = 5
    model_save_path = "/root/BA__Projekt/assets/models/pth/ednn_regression__iris/ednn_regression__iris.pth"
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            mu, v, alpha, beta = model(X)
            loss = evidential_loss(y, mu, v, alpha, beta)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        writer.add_scalar("Loss/train", avg_loss, epoch)

        # === Validation ===
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                mu, v, alpha, beta = model(X_val)
                loss = evidential_loss(y_val, mu, v, alpha, beta)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        writer.add_scalar("Loss/val", avg_val_loss, epoch)

        print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

        # === Early Stopping ===
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_save_path)
            print(f"Validation improved. Model saved to {model_save_path}")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s).")
            if epochs_no_improve >= patience:
                print("Early stopping triggered!")
                break

    writer.close()


def main():
    dataset = DatasetTorchDuckDBIris(
        db_path="/root/BA__U-i-mlb-Sm-f-d-s-V-a-S/BA__Projekt/assets/dbs/dataset__iris__dataset.duckdb",
        table_name="iris__dataset_csv",
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=16)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EvidentialNet(input_dim=4)
    train(model, train_loader, val_loader, epochs=100, device=device)


if __name__ == "__main__":
    main()
