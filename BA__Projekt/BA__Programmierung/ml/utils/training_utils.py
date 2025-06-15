# BA__Programmierung/ml/utils/training_utils.py

"""
training_utils.py

Utility functions for training and evaluating PyTorch models with evidential regression loss.

This module provides reusable functions for:
- Training a single epoch
- Evaluating a model on a validation set
- Managing training with early stopping

Author: [Your Name or Team]
"""

import os
import torch

from BA__Programmierung.ml.losses.evidential_loss import evidential_loss
from pathlib import Path


def train_one_epoch(model, dataloader, optimizer, device):
    """
    Train a model for one epoch using evidential loss.

    :param model: The PyTorch model to train.
    :type model: torch.nn.Module
    :param dataloader: DataLoader for training data.
    :type dataloader: torch.utils.data.DataLoader
    :param optimizer: Optimizer for updating model weights.
    :type optimizer: torch.optim.Optimizer
    :param device: Device to perform training on ('cpu' or 'cuda').
    :type device: torch.device

    :return: Average training loss for the epoch.
    :rtype: float
    """
    model.train()
    total_loss = 0.0
    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.float().unsqueeze(1).to(device)

        optimizer.zero_grad()
        mu, v, alpha, beta = model(inputs)
        loss = evidential_loss(targets, mu, v, alpha, beta, mode="nll")
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    """
    Evaluate a model on a validation or test dataset.

    :param model: The PyTorch model to evaluate.
    :type model: torch.nn.Module
    :param dataloader: DataLoader for validation or test data.
    :type dataloader: torch.utils.data.DataLoader
    :param device: Device to perform evaluation on ('cpu' or 'cuda').
    :type device: torch.device

    :return: Average validation loss.
    :rtype: float
    """
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.float().unsqueeze(1).to(device)
            mu, v, alpha, beta = model(inputs)
            loss = evidential_loss(targets, mu, v, alpha, beta, mode="nll")
            total_loss += loss.item()
    return total_loss / len(dataloader)


def train_with_early_stopping(
    model,
    train_loader,
    val_loader,
    optimizer,
    model_path,
    device,
    epochs=50,
    patience=5,
):
    """
    Train a model with early stopping based on validation loss.

    :param model: The PyTorch model to train.
    :type model: torch.nn.Module
    :param train_loader: DataLoader for training data.
    :type train_loader: torch.utils.data.DataLoader
    :param val_loader: DataLoader for validation data.
    :type val_loader: torch.utils.data.DataLoader
    :param optimizer: Optimizer for updating model weights.
    :type optimizer: torch.optim.Optimizer
    :param model_path: File path to save the best model.
    :type model_path: str
    :param device: Device to perform training on ('cpu' or 'cuda').
    :type device: torch.device
    :param epochs: Maximum number of epochs to train. Default is 50.
    :type epochs: int, optional
    :param patience: Number of epochs with no improvement after which training will be stopped. Default is 5.
    :type patience: int, optional

    :return: None
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)

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


def get_dataset_root(path):
    """
    Extract the dataset root directory containing the specified folder.

    This function resolves the given file path, searches for the
    folder named "BA__U-i-mlb-Sm-f-d-s-V-a-S" in the path parts, and returns
    the directory path up to and including that folder. If the folder is not
    found, it returns the parent directory of the given path.

    Parameters
    ----------
    path : str or Path
        The input file path (usually to a CSV file).

    Returns
    -------
    str
        The resolved dataset root directory path containing the target folder.

    Raises
    ------
    None

    Examples
    --------
    >>> get_dataset_root("/root/BA__U-i-mlb-Sm-f-d-s-V-a-S/BA__Projekt/assets/data/raw/dataset__fmnist/fashion-mnist_train.csv")
    '/root/BA__U-i-mlb-Sm-f-d-s-V-a-S/BA__Projekt/assets/data/raw/dataset__fmnist'

    >>> get_dataset_root("/some/other/path/file.csv")
    '/some/other/path'
    """
    p = Path(path).resolve()

    parts = p.parts
    try:
        idx = parts.index("BA__U-i-mlb-Sm-f-d-s-V-a-S")
    except ValueError:
        return str(p.parent)

    root = Path(*parts[: idx + 1])
    relative = Path(*parts[idx + 1 :])

    return str(root / relative.parent)
