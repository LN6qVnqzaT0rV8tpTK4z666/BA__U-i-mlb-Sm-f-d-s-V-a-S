# BA__Projekt/BA__Programmierung/ml/utils/training_utils.py

"""
training_utils.py

Utility functions for training and evaluating PyTorch models with evidential regression loss.

This module supports single-epoch training, full training with early stopping, and
evaluation with extensive uncertainty quantification and metric logging.

Functions
---------
train_one_epoch(model, dataloader, optimizer, device, loss_mode="nll")
    Train the model for a single epoch using the specified evidential loss mode.

evaluate(model, data_loader, device, loss_mode="nll", metrics_token=None)
    Evaluate the model on a validation set and collect uncertainty metrics.

train_with_early_stopping(model, train_loader, val_loader, optimizer, model_path, device, ...)
    Perform training with early stopping and metric tracking.
"""
import os
import torch

from BA__Programmierung.ml.metrics.metrics_registry import MetricsRegistry, metrics_registry
from BA__Programmierung.ml.losses.evidential_loss import evidential_loss


def train_one_epoch(model, dataloader, optimizer, device, loss_mode="nll"):
    """
    Train the model for one epoch.

    Parameters
    ----------
    model : torch.nn.Module
        The model to train.
    dataloader : torch.utils.data.DataLoader
        DataLoader providing training data batches.
    optimizer : torch.optim.Optimizer
        Optimizer used for updating model weights.
    device : torch.device
        Device to run training on.
    loss_mode : str, optional
        Mode to compute evidential loss (e.g., "nll", "mse", "full"). Default is "nll".

    Returns
    -------
    float
        The average training loss for this epoch.
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


def evaluate(model, data_loader, device, loss_mode="nll", metrics_token=None):
    """
    Evaluate the model and compute UQ metrics.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model to evaluate.
    data_loader : torch.utils.data.DataLoader
        Validation data loader.
    device : torch.device
        Device to perform evaluation on.
    loss_mode : str, optional
        Loss mode to use during evaluation.
    metrics_token : str, optional
        Token to identify which metrics to evaluate.

    Returns
    -------
    float
        The average loss over the validation dataset.
    """
    model.eval()
    total_loss = 0.0

    all_outputs = {
        "y_true": [],
        "y_pred": [],
        "y_pred_mean": [],
        "y_pred_std": [],
        "logvar": [],
        "mean": [],
        "target": [],
        "pred_probs": [],
        "y_samples": [],
        "mc_preds": [],
        "y_lower": [],
        "y_upper": [],
        "pred_mean": [],
        "pred_std": [],
        "ref_mean": [],
        "ref_std": [],
        "uncertainty": [],
        "pred_uncert_aleatoric": [],
        "pred_uncert_epistemic": [],
        "true_errors": [],
        "confidence_scores": [],
        "baseline_confidence_scores": [],
        "meta_calibration_score": [],
        "corr_err_epistemic": [],
        "uda": [],
        "ncg": [],
        "n_bins": [],
        "kl_div": [],
        "v": [],
        "alpha": [],
        "beta": [],
    }

    with torch.no_grad():
        for batch in data_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            mu, v, alpha, beta = model(inputs)

            logvar = torch.log(1.0 / v)
            std = torch.sqrt(1.0 / v)

            # Monte Carlo Samples
            samples = mu.unsqueeze(1) + std.unsqueeze(1) * torch.randn(mu.size(0), 100, mu.size(1), device=mu.device)
            all_outputs["y_samples"].append(samples)

            # Prediction Probabilities (falls Klassifikation)
            if samples.ndim == 3 and samples.shape[-1] > 1:
                all_outputs["pred_probs"].append(samples.softmax(dim=-1))
            else:
                all_outputs["pred_probs"].append(torch.full_like(samples, fill_value=1.0))

            # Klassisch
            all_outputs["mean"].append(mu)
            all_outputs["logvar"].append(logvar)
            all_outputs["y_true"].append(targets)
            all_outputs["target"].append(targets)
            all_outputs["y_pred"].append(mu)
            all_outputs["y_pred_mean"].append(mu)
            all_outputs["y_pred_std"].append(std)
            all_outputs["pred_std"] = all_outputs["y_pred_std"]

            # Unsicherheiten
            aleatoric = 1.0 / (alpha - 1).clamp(min=1e-8)
            epistemic = beta / (v * (alpha - 1).clamp(min=1e-8))
            uncertainty = torch.sqrt(aleatoric + epistemic)

            all_outputs["uncertainty"].append(uncertainty)
            all_outputs["pred_uncert_aleatoric"].append(torch.sqrt(aleatoric))
            all_outputs["pred_uncert_epistemic"].append(torch.sqrt(epistemic))

            all_outputs["v"].append(v)
            all_outputs["alpha"].append(alpha)
            all_outputs["beta"].append(beta)

            # KL-Divergenz: KL(N(mu, std^2) || N(0,1))
            ref_mean = torch.zeros_like(mu)
            ref_std = torch.ones_like(std)

            kl = 0.5 * (
                torch.log(ref_std / std) +
                (std.pow(2) + (mu - ref_mean).pow(2)) / ref_std.pow(2) - 1
            )

            # Reduziere ggf. über Features (z. B. dim=1 bei shape [B, D])
            if kl.ndim == 2:
                kl = kl.sum(dim=1)  # [B]

            kl_div = kl.view(-1)  # Sicherstellen, dass es 1D ist
            assert kl_div.ndim == 1, f"Expected 1D kl_div, got {kl_div.shape}"
            all_outputs["kl_div"].append(kl_div)
            # Konfidenz
            all_outputs["confidence_scores"].append(1.0 / (uncertainty + 1e-8))
            all_outputs["baseline_confidence_scores"].append(1.0 / (std + 1e-8))

            # Intervallgrenzen
            z = 1.96
            all_outputs["y_lower"].append(mu - z * std)
            all_outputs["y_upper"].append(mu + z * std)

            # Referenz (für KL-Divergenz)
            all_outputs["ref_mean"].append(torch.zeros_like(mu))
            all_outputs["ref_std"].append(torch.ones_like(std))

            # true error
            all_outputs["true_errors"].append(torch.abs(mu - targets))

            # künstliche n_bins für ECE & Co.
            #all_outputs["n_bins"].append(torch.tensor(10, device=device).expand(mu.size(0)))  # oder 15 für regression_ece
            all_outputs["n_bins"].append(10)

            # KL-Term (nur Dummy, du kannst echten Wert einfügen wenn verfügbar)
            # all_outputs["kl_div"].append(torch.zeros_like(mu))

            # Optional: MC Dropout Predictions (z. B. 5x forward pass)
            # Hier nur ein Dummy-Vektor – echtes Sampling wäre: model.forward_n(...)
            all_outputs["mc_preds"].append(samples[:, :5, :])  # 5 Samples

            # Meta-Kalibrierungs-Mocks (nur falls notwendig)
            all_outputs["meta_calibration_score"].append(torch.ones_like(mu) * 0.5)
            all_outputs["corr_err_epistemic"].append(torch.ones_like(mu) * 0.5)
            all_outputs["uda"].append(torch.ones_like(mu) * 0.5)
            all_outputs["ncg"].append(torch.ones_like(mu) * 0.5)

            # Verlust
            loss = evidential_loss(targets, mu, v, alpha, beta, mode=loss_mode)
            total_loss += loss.item()


    # Merge collected outputs (robust)
    merged_outputs = {}
    for k, v in all_outputs.items():
        if isinstance(v, list) and len(v) > 0:
            first = v[0]
            try:
                if isinstance(first, torch.Tensor):
                    merged_outputs[k] = torch.cat(v, dim=0)
                elif isinstance(first, (list, tuple)) and isinstance(first[0], torch.Tensor):
                    merged_outputs[k] = torch.cat([item for sublist in v for item in sublist], dim=0)
                else:
                    merged_outputs[k] = torch.tensor(v)  # Fallback (z. B. floats)
            except Exception as e:
                print(f"⚠️ Could not merge {k}: {e}")
                merged_outputs[k] = v  # Zur Not: gib einfach die Liste weiter
        else:
            merged_outputs[k] = v
    try:
        if "kl_div" in merged_outputs:
            merged_outputs["kl_div"] = merged_outputs["kl_div"].view(-1)  # ensure it's 1D
    except Exception as e:
        print(f"⚠️ Fehler beim Zusammenführen von 'kl_div': {e}")

    if "mean" in merged_outputs:
        merged_outputs["mu"] = merged_outputs["mean"]

    if "n_bins" not in merged_outputs:
        merged_outputs["n_bins"] = torch.tensor(10)

    metrics_registry = MetricsRegistry()

    # Fix: inject 'eps' only if mape is expected to be computed
    if "mape" in metrics_registry.list("regression")["regression"]:
        y_pred = merged_outputs.get("y_pred_mean", merged_outputs.get("y_pred"))
        merged_outputs["eps"] = torch.full_like(y_pred, 1e-8)

    merged_outputs["y_pred"] = merged_outputs.get("y_pred_mean")

    # # Normalize shape of y_pred and y_true
    # if "y_pred" in merged_outputs and merged_outputs["y_pred"].ndim == 2:
    #     if merged_outputs["y_pred"].shape[1] == 1:
    #         merged_outputs["y_pred"] = merged_outputs["y_pred"].squeeze(1)

    # if "y_true" in merged_outputs and merged_outputs["y_true"].ndim == 2:
    #     if merged_outputs["y_true"].shape[1] == 1:
    #         merged_outputs["y_true"] = merged_outputs["y_true"].squeeze(1)

    # Run metrics
    if metrics_token:
        metrics = metrics_registry.get_metrics(metrics_token)

        for metric in metrics:
            try:
                needed_args = metric.arg_names
                args_for_metric = {
                    name: merged_outputs[name]
                    for name in needed_args
                    if name in merged_outputs and merged_outputs[name] is not None
                }

                if len(args_for_metric) < len(needed_args):
                    print(f"{metric.name:<25}: ⏭ Skipped due to missing args.")
                    print(f"🧪 {metric.name:<25} ← args: {list(args_for_metric.keys())}")
                    continue

                metric(**args_for_metric)

            except Exception as e:
                print(f"{metric.name:<25}: ❌ Error during computation: {e}")
                print(f"{metric.name:<25}: (batch-only or failed)")


    for key, value in all_outputs.items():
        if isinstance(value, list) and len(value) > 0 and isinstance(value[0], torch.Tensor):
            try:
                if value[0].dim() == value[-1].dim():  # Einfache Prüfung auf konsistente Shapes
                    all_outputs[key] = torch.cat(value, dim=0)
                else:
                    all_outputs[key] = torch.stack(value, dim=0)
            except Exception as e:
                print(f"⚠️ Fehler beim Zusammenführen von '{key}': {e}")

    # Ensure required regression inputs exist
    if "y_pred" not in merged_outputs and "y_pred_mean" in merged_outputs:
        merged_outputs["y_pred"] = merged_outputs["y_pred_mean"]
    if "y_true" not in merged_outputs and "target" in merged_outputs:
        merged_outputs["y_true"] = merged_outputs["target"]

    metrics_registry.add_batch(metrics_token, **merged_outputs)

    return total_loss / len(data_loader)


def train_with_early_stopping(model, train_loader, val_loader, optimizer, model_path,
                              device, epochs=50, patience=5, loss_mode="nll", metrics_token=None, resume_epoch=0):
    """
    Train a model with early stopping and metric tracking.

    Parameters
    ----------
    model : torch.nn.Module
        The model to train.
    train_loader : torch.utils.data.DataLoader
        DataLoader for training data.
    val_loader : torch.utils.data.DataLoader
        DataLoader for validation data.
    optimizer : torch.optim.Optimizer
        Optimizer for model parameter updates.
    model_path : str
        Path to save the best performing model.
    device : torch.device
        The device (CPU or GPU) for computation.
    epochs : int, optional
        Maximum number of training epochs. Default is 50.
    patience : int, optional
        Number of epochs to wait for improvement before stopping. Default is 5.
    loss_mode : str, optional
        Loss variant to use. Default is "nll".
    metrics_token : str, optional
        Token to track and report relevant metrics.
    resume_epoch : int, optional
        The epoch number to resume training from (default is 0, meaning start from the beginning).

    Returns
    -------
    None
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    best_val_loss = float("inf")
    epochs_no_improve = 0

    # Resume training from the resume_epoch
    for epoch in range(resume_epoch, epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, loss_mode)
        val_loss = evaluate(model, val_loader, device, loss_mode, metrics_token)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if metrics_token:
            print(f"📊 Evaluation Metrics [{metrics_token}]:")
            metrics_registry.report(metrics_token)

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


def load_model_checkpoint(model, optimizer, model_path, device):
    """
    Load the model and optimizer state from a checkpoint if it exists.
    
    Args:
        model (torch.nn.Module): The model to load the checkpoint into.
        optimizer (torch.optim.Optimizer): The optimizer to load the state into.
        model_path (str): Path to the model checkpoint.
        device (torch.device): The device to load the model onto.
        
    Returns:
        bool: Whether the checkpoint was loaded successfully.
    """
    if os.path.exists(model_path):
        print(f"Loading checkpoint from {model_path}...")
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"Checkpoint loaded (Epoch: {epoch}, Loss: {loss})")
        return True
    return False
