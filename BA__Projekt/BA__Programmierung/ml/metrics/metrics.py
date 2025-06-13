# BA__Projekt/BA__Programmierung/ml/metrics/metrics.py
"""
Metrics module for evaluating machine learning models.

This module provides a registry-based system for defining, storing, and retrieving
metrics for classification, regression, and uncertainty quantification (UQ) tasks.
It supports both batch-level and accumulated evaluation across multiple batches.

Classes:
    Metric: Wraps a metric function and handles optional accumulation.
    MetricsRegistry: Singleton registry for registering and retrieving metrics.

Functions:
    accuracy, top_k_accuracy, mse, rmse, mae, mape, r2_score, nll_gaussian, energy_score

Usage:
    Register metrics for a task using `Metrics.register(...)` and
    compute them via `metric(y_pred, y_true)`.
"""

from collections import defaultdict
from typing import Callable, Dict, List, Optional, Union

import torch
import torch.nn.functional as F

from BA__Programmierung.util.singleton import Singleton


class Metric:
    """
    A class to wrap metric functions and support optional accumulation.

    Args:
        fn (Callable): The metric function to wrap.
        name (str, optional): Custom name for the metric. Defaults to function name.
        accumulate (bool): Whether to accumulate results across multiple batches.
    """

    def __init__(self, fn: Callable, name: Optional[str] = None, accumulate: bool = False):
        self.fn = fn
        self.name = name or fn.__name__
        self.accumulate = accumulate
        self.reset()

    def __call__(self, y_pred, y_true, **kwargs):
        """
        Evaluates or accumulates the metric.

        Args:
            y_pred (Tensor): Predicted values.
            y_true (Tensor): True target values.
        """
        if self.accumulate:
            self._accumulate(y_pred, y_true)
        else:
            return self.fn(y_pred, y_true, **kwargs)

    def _accumulate(self, y_pred, y_true):
        """
        Stores batch predictions and targets for later evaluation.
        """
        self.preds.append(y_pred.detach().cpu())
        self.trues.append(y_true.detach().cpu())

    def compute(self):
        """
        Computes the accumulated metric value.

        Returns:
            float: The computed metric.

        Raises:
            ValueError: If the metric is not set to accumulate.
        """
        if not self.accumulate:
            raise ValueError(f"Metric '{self.name}' does not support accumulation.")
        preds = torch.cat(self.preds)
        trues = torch.cat(self.trues)
        return self.fn(preds, trues)

    def reset(self):
        """
        Resets internal state for accumulation.
        """
        self.preds = []
        self.trues = []


class MetricsRegistry(Singleton):
    """
    A singleton registry for managing multiple metric functions across tasks.

    Methods:
        register: Adds a metric function under a specific task.
        get: Retrieves registered metrics by name for a task.
        list: Lists all metric names for a task.
        all: Returns all Metric instances for a task.
        report: Computes and prints all accumulated metrics for a task.
    """

    def __init__(self):
        if not hasattr(self, "_initialized"):
            self._registry: Dict[str, Dict[str, Metric]] = defaultdict(dict)
            self._initialized = True

    def register(self, task: str, fn: Callable, name: Optional[str] = None, accumulate=False):
        """
        Registers a new metric under a task.

        Args:
            task (str): The task name (e.g., "classification").
            fn (Callable): The metric function.
            name (str, optional): Custom name for the metric.
            accumulate (bool): Whether the metric should accumulate over batches.
        """
        metric = Metric(fn, name, accumulate)
        self._registry[task][metric.name] = metric

    def get(self, task: str, names: Union[str, List[str]]) -> Dict[str, Metric]:
        """
        Retrieves metric(s) by name for a specific task.

        Args:
            task (str): The task name.
            names (Union[str, List[str]]): Metric name or list of names.

        Returns:
            Dict[str, Metric]: The retrieved metrics.
        """
        if isinstance(names, str):
            names = [names]
        return {name: self._registry[task][name] for name in names}

    def list(self, task: Optional[str] = None) -> Dict[str, List[str]]:
        """
        Lists all registered metrics.

        Args:
            task (str, optional): Specific task to list metrics for.

        Returns:
            Dict[str, List[str]]: Metric names by task.
        """
        if task:
            return {task: list(self._registry[task].keys())}
        return {task: list(metrics.keys()) for task, metrics in self._registry.items()}

    def all(self, task: str) -> Dict[str, Metric]:
        """
        Returns all metrics for a task.

        Args:
            task (str): The task name.

        Returns:
            Dict[str, Metric]: All registered metrics for the task.
        """
        return self._registry[task]

    def report(self, task: str, verbose=True) -> Dict[str, float]:
        """
        Computes and prints all accumulated metrics for a task.

        Args:
            task (str): The task name.
            verbose (bool): Whether to print results.

        Returns:
            Dict[str, float]: Metric values.
        """
        out = {}
        for name, metric in self._registry[task].items():
            if metric.accumulate:
                result = metric.compute()
                metric.reset()
            else:
                result = None
            out[name] = result
            if verbose:
                print(f"{name:20s}: {result:.4f}" if result is not None else f"{name:20s}: (batch-only)")
        return out


# ───── Global Singleton Registry ───── #
Metrics = MetricsRegistry()


# ───── Metric Functions ───── #

def accuracy(y_pred, y_true):
    """
    Computes classification accuracy.

    Returns:
        float: Accuracy in [0, 1].
    """
    return (torch.argmax(y_pred, dim=1) == y_true).float().mean().item()

def top_k_accuracy(y_pred, y_true, k=3):
    """
    Computes top-k classification accuracy.

    Args:
        k (int): Number of top predictions to consider.

    Returns:
        float: Top-k accuracy.
    """
    topk = y_pred.topk(k, dim=1).indices
    correct = sum(y_true[i] in topk[i] for i in range(len(y_true)))
    return correct / len(y_true)

def mse(y_pred, y_true):
    """Mean Squared Error."""
    return F.mse_loss(y_pred, y_true).item()

def rmse(y_pred, y_true):
    """Root Mean Squared Error."""
    return torch.sqrt(F.mse_loss(y_pred, y_true)).item()

def mae(y_pred, y_true):
    """Mean Absolute Error."""
    return F.l1_loss(y_pred, y_true).item()

def mape(y_pred, y_true, eps=1e-8):
    """
    Mean Absolute Percentage Error.

    Args:
        eps (float): Small value to prevent division by zero.

    Returns:
        float: MAPE value.
    """
    return (torch.abs((y_true - y_pred) / (y_true + eps)).mean() * 100).item()

def r2_score(y_pred, y_true):
    """
    Coefficient of Determination (R²).

    Returns:
        float: R² score.
    """
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    return (1 - ss_res / ss_tot).item()

def nll_gaussian(mean, logvar, target):
    """
    Negative Log-Likelihood for a Gaussian distribution.

    Returns:
        float: NLL value.
    """
    precision = torch.exp(-logvar)
    return 0.5 * torch.mean(precision * (target - mean) ** 2 + logvar).item()

def energy_score(y_samples, y_true):
    """
    Computes the energy score for probabilistic forecasts.

    Args:
        y_samples (Tensor): Sample predictions (B, S, D).
        y_true (Tensor): True target values (B, D).

    Returns:
        float: Energy score.
    """
    B, S, D = y_samples.shape
    y_true = y_true.unsqueeze(1).expand(-1, S, -1)
    t1 = torch.norm(y_samples - y_true, dim=-1).mean()
    t2 = torch.norm(y_samples.unsqueeze(2) - y_samples.unsqueeze(1), dim=-1).mean()
    return (t1 - 0.5 * t2).item()


# ───── Register Metrics ───── #

Metrics.register("classification", accuracy, accumulate=True)
Metrics.register("classification", top_k_accuracy, name="top3_acc", accumulate=True)

Metrics.register("regression", mse, accumulate=True)
Metrics.register("regression", rmse, accumulate=True)
Metrics.register("regression", mae, accumulate=True)
Metrics.register("regression", mape, accumulate=True)
Metrics.register("regression", r2_score, name="r2_score", accumulate=True)

Metrics.register("uq", nll_gaussian, accumulate=True)
Metrics.register("uq", energy_score, accumulate=True)


# ───── Example Usage ───── #

if __name__ == "__main__":
    torch.manual_seed(42)

    print("🔢 Classification Metrics")
    cls_metrics = Metrics.get("classification", ["accuracy", "top3_acc"])
    for _ in range(3):
        y_pred = torch.randn(32, 10)
        y_true = torch.randint(0, 10, (32,))
        for metric in cls_metrics.values():
            metric(y_pred, y_true)
    Metrics.report("classification")

    print("\n📈 Regression Metrics")
    reg_metrics = Metrics.get("regression", ["mse", "rmse", "mae", "mape", "r2_score"])
    for _ in range(3):
        y_pred = torch.randn(64, 1)
        y_true = y_pred + 0.1 * torch.randn_like(y_pred)
        for metric in reg_metrics.values():
            metric(y_pred, y_true)
    Metrics.report("regression")

    print("\n📊 Uncertainty Quantification (UQ) Metrics")
    y_true = torch.randn(16, 1)
    mean = y_true + 0.05 * torch.randn_like(y_true)
    logvar = torch.zeros_like(y_true)
    y_samples = mean.unsqueeze(1) + 0.1 * torch.randn(16, 100, 1)
    Metrics.all("uq")["nll_gaussian"](mean, logvar, y_true)
    Metrics.all("uq")["energy_score"](y_samples, y_true)
    Metrics.report("uq")
