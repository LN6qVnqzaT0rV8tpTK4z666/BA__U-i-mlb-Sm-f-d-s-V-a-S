# BA__Projekt/BA__Programmierung/ml/ednn_regression__iris.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator


# from models.model__ednn_basic import EvidentialNet
from models.model__ednn_deep import EvidentialNetDeep as EvidentialNet
from BA__Programmierung.ml.losses.evidential_loss import evidential_loss


# ============
# Dataset
# ============


class IrisDataset(Dataset):
    def __init__(self):
        data = load_iris()
        X = data.data
        y = data.target.astype(float)  # regression-style: float targets

        # Only use samples from two classes (binary regression) or simplify the task
        mask = y < 2
        X, y = X[mask], y[mask]

        self.scaler = StandardScaler()
        self.X = torch.tensor(
            self.scaler.fit_transform(X), dtype=torch.float32
        )
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ============
# Training
# ============


def train(model, train_loader, val_loader, epochs=100, lr=1e-3, device="cpu"):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # === TensorBoard Writer ===
    log_dir = os.path.join(
        "/root/BA__U-i-mlb-Sm-f-d-s-V-a-S/BA__Projekt/assets/data/processed",
        "ednn_iris_" + datetime.now().strftime("%Y%m%d-%H%M%S"),
    )
    writer = SummaryWriter(log_dir=log_dir)

    model.to(device)

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

        # Optional: Add val loss tracking
        if val_loader:
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

        print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_loss:.4f}")

    writer.close()


# ============
# Main
# ============


def main():
    dataset = IrisDataset()
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=16)

    model = EvidentialNet(input_dim=4)
    train(model, train_loader, val_loader, epochs=100)


if __name__ == "__main__":
    main()
