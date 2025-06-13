# BA__Projekt/BA__Programmierung/ml/ednn_regression__wine-quality-white.py

import os
import torch

from BA__Programmierung.ml.losses.evidential_loss__nll_reg import evidential_regression_loss
from BA__Programmierung.ml.datasets.dataset__torch__wine_quality_white import load_wine_quality_white_dataset
from models.model__ednn_basic import EvidentialNet
# from models.model__ednn_deep import EvidentialNetDeep as EvidentialNet
# from models.model__ednn_deep_dropout import EvidentialNetDeep as EvidentialNet
from torch.utils.data import random_split, DataLoader


def train_model(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.float().unsqueeze(1).to(device)

        optimizer.zero_grad()
        mu, v, alpha, beta = model(inputs)
        loss = evidential_regression_loss(targets, mu, v, alpha, beta)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate_model(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.float().unsqueeze(1).to(device)
            mu, v, alpha, beta = model(inputs)
            loss = evidential_regression_loss(targets, mu, v, alpha, beta)
            total_loss += loss.item()
    return total_loss / len(dataloader)


def main():
    dataset_path = "assets/data/raw/dataset__wine-quality/winequality-white.csv"
    dataset = load_wine_quality_white_dataset(dataset_path)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EvidentialNet(input_dim=dataset[0][0].shape[0]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 50
    patience = 5
    best_val_loss = float("inf")
    epochs_no_improve = 0

    model_dir = "assets/models/pth/ednn_regression__wine_quality_white/"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "ednn__wine-quality-white.pt")

    for epoch in range(epochs):
        train_loss = train_model(model, train_loader, optimizer, device)
        val_loss = evaluate_model(model, val_loader, device)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_path)
            print(f"Validation improved. Model saved at {model_path}")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break


if __name__ == "__main__":
    main()
