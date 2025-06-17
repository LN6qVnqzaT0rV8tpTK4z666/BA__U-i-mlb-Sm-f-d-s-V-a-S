# BA__Programmierung/ml/utils/training_utils.py
"""
training_utils.py

Utilities for training and evaluating PyTorch models with evidential regression loss.

Functions:
- train_one_epoch: Train for one epoch
- evaluate: Evaluate on validation set
- train_with_early_stopping: Train with early stopping
"""

import os
from BA__Programmierung.ml.metrics.metrics_registry import Metrics
import torch

from BA__Programmierung.ml.losses.evidential_loss import evidential_loss


def train_one_epoch(model, dataloader, optimizer, device, loss_mode="nll"):
    """
    Train the model for one epoch.

    Parameters
    ----------
    model : torch.nn.Module
    dataloader : DataLoader
    optimizer : Optimizer
    device : torch.device
    loss_mode : str, optional

    Returns
    -------
    float
        Average training loss.
    """
    model.train()
    total_loss = 0.0
    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.float().unsqueeze(1).to(device)

        optimizer.zero_grad()
        mu, v, alpha, beta = model(inputs)
        loss = evidential_loss(targets, mu, v, alpha, beta, mode=loss_mode)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device, loss_mode="nll", metrics_token=None):
    """
    Evaluate model performance and optionally accumulate registered metrics.

    Parameters
    ----------
    model : torch.nn.Module
    dataloader : DataLoader
    device : torch.device
    loss_mode : str
    metrics_token : str or None

    Returns
    -------
    float
        Average validation loss.
    """
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.float().unsqueeze(1).to(device)

            mu, v, alpha, beta = model(inputs)

            # ----- Compute validation loss -----
            loss = evidential_loss(targets, mu, v, alpha, beta, mode=loss_mode)
            total_loss += loss.item()

            # ----- Optional: compute extra metrics -----
            if metrics_token:
                y_pred = mu  # or: torch.cat([mu, v, alpha, beta], dim=-1) depending on metrics
                for metric in Metrics.get_metrics(metrics_token):
                    metric(*y_pred, y_true=targets)

    return total_loss / len(dataloader)



def train_with_early_stopping(model, train_loader, val_loader, optimizer, model_path,
                              device, epochs=50, patience=5, loss_mode="nll", metrics_token=None):
    """
    Train with early stopping.

    Parameters
    ----------
    model : torch.nn.Module
    train_loader : DataLoader
    val_loader : DataLoader
    optimizer : Optimizer
    model_path : str
    device : torch.device
    epochs : int, optional
    patience : int, optional
    loss_mode : str, optional
    metrics_token : str, optional
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, loss_mode)
        val_loss = evaluate(model, val_loader, device, loss_mode, metrics_token)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if metrics_token:
            print(f"📊 Evaluation Metrics [{metrics_token}]:")
            Metrics.report(metrics_token)

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
