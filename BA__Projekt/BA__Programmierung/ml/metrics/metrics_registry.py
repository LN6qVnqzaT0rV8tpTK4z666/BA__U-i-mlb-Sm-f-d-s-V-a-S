# BA__Projekt/BA__Programmierung/ml/metrics/metrics_registry.py

"""
Metrics module for evaluating machine learning models.

This module provides a registry-based system for defining, storing, and retrieving
metrics for classification, regression, and uncertainty quantification (UQ) tasks.
It supports both batch-level and accumulated evaluation across multiple batches.

Classes:
    Metric: Wraps a metric function and handles optional accumulation.
    MetricsRegistry: Singleton registry for registering and retrieving metrics.

Usage:
    Register metrics for a task using `Metrics.register(...)` and
    compute them via `metric(y_pred, y_true)` or via accumulation with `metric(...)` + `metric.compute()`.

Register a new metric and evaluate it:

>>> from your_module.metrics_registry import metrics_registry
>>> metrics_registry.register(
...     token="regression",
...     fn=your_custom_metric,
...     name="my_metric",
...     accumulate=True,
...     arg_names=["y_pred", "y_true"]
... )
>>> metric = metrics_registry.get("regression", "my_metric")["my_metric"]
>>> metric(y_pred, y_true)
>>> metrics_registry.report("regression")
"""

from collections import defaultdict
from collections.abc import Callable
from BA__Programmierung.ml.metrics.metrics_registry_definitions import (
    accuracy,
    ace,
    aleatoric_variance,
    calibration_error,
    continuous_ranked_probability_score,
    ece,
    elbo,
    energy_score,
    epistemic_variance,
    evidence,
    kl_divergence_normal,
    mae,
    mape,
    marginal_likelihood,
    mean_pred_variance,
    meta_metric__bnn_ednn,
    mpiw,
    mse,
    mutual_information,
    ncg,
    nll_gaussian,
    picp,
    predictive_entropy,
    r2_score,
    regression_ece,
    rmse,
    statistical__mod_pred__mean,
    statistical__mod_pred__variance_band,
    statistical__mod_pred__standard_error,
    statistical__mod_pred__quantiles,
    statistical__mod_pred__predictive_mean,
    statistical__mod_pred__plus_minus_sigma,
    statistical__mod_pred__plus_minus_1_sigma,
    statistical__mod_pred__plus_minus_2_sigma,
    statistical__mod_pred__plus_minus_3_sigma,
    statistical__mod_pred__plus_minus_4_sigma,
    statistical__mod_pred__plus_minus_5_sigma,
    top_k_accuracy,
    uda
)
from BA__Programmierung.ml.metrics.metrics import Metric
from BA__Programmierung.util.singleton import Singleton
from typing import Union

import numpy as np
import builtins
import torch


class MetricsRegistry(metaclass=Singleton):
    """
    Singleton registry for storing and retrieving evaluation metrics.

    The registry supports grouping by task tokens (e.g., 'classification', 'regression', 'uq').
    Each metric is stored with metadata such as whether it supports batch accumulation.

    Attributes
    ----------
    _registry : dict[str, dict[str, Metric]]
        Internal mapping from task tokens to metric dictionaries.
    storage : dict[str, dict[str, list]]
        Stores accumulated inputs for batch-based metric computation.

    Methods
    -------
    register(token, fn, name=None, accumulate=False, arg_names=None)
        Registers a new metric function under the given token.
    
    get(token, names)
        Retrieves one or multiple registered metrics by name.
    
    get_metrics(token)
        Returns all Metric objects for the given token.
    
    list_tokens()
        Returns a list of all registered tokens.
    
    list(token=None)
        Returns all metric names for a specific token or for all tokens.
    
    all(token)
        Returns all registered Metric instances for a token.
    
    report(token, verbose=True)
        Computes and prints all metrics for a given token based on accumulated data.
    
    add_batch(token, **kwargs)
        Adds data (e.g., predictions and targets) for accumulation across batches.
    
    get_metric_bundles()
        Returns a dictionary of token → metric names (available groups).
    """

    def __init__(self):
        """
        Initialize the internal registry.
        """
        self._registry: dict[str, dict[str, Metric]] = defaultdict(dict)
        self.storage: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))

    def register(self, token: str, fn: Callable, name: str = None, accumulate: bool = False, arg_names: list[str] = None):
        """
        Register a metric function under a token.

        :param token: Task or dataset identifier.
        :param fn: Metric function.
        :param name: Optional metric name.
        :param accumulate: If True, accumulate over batches.
        :param arg_names: List of argument names to match metric function's signature.
        """
        metric = Metric(fn, name=name, accumulate=accumulate, arg_names=arg_names)
        self._registry[token][metric.name] = metric

    def get(self, token: str, names: Union[str, list[str]]) -> dict[str, Metric]:
        """
        Retrieve specific metrics by name.

        :param token: Token to identify metric group.
        :param names: Metric name or list of names.
        :return: Dictionary of metrics.
        """
        if isinstance(names, str):
            names = [names]
        return {name: self._registry[token][name] for name in names}

    def get_metrics(self, token: str) -> list[Metric]:
        """
        Return all metrics for a token.

        :param token: Metric group identifier.
        :return: List of Metric instances.
        """
        return list(self._registry.get(token, {}).values())

    def list_tokens(self) -> list[str]:
        """
        List all registered tokens.

        :return: List of token strings.
        """
        return list(self._registry.keys())

    def list(self, token: str = None) -> dict[str, list[str]]:
        """
        List metric names under a token or all tokens.

        :param token: Specific token or None for all.
        :return: Dictionary of token to metric names.
        """
        if token:
            return {token: list(self._registry[token].keys())}
        return {token: list(metrics.keys()) for token, metrics in self._registry.items()}

    def all(self, token: str) -> dict[str, Metric]:
        """
        Get all metrics as a dict for a token.

        :param token: Token identifier.
        :return: Dictionary of metrics.
        """
        return self._registry[token]

    def report(self, token: str, verbose: bool = True) -> dict[str, float | None]:
        out = {}

        if token not in self._registry or token not in self.storage:
            print(f"⚠️ No data found for token '{token}'")
            return out

        inputs = {}

        # Merge inputs
        for key, values in self.storage[token].items():
            try:
                if isinstance(values[0], torch.Tensor):
                    inputs[key] = torch.cat(values, dim=0)
                elif isinstance(values[0], np.ndarray):
                    inputs[key] = np.concatenate(values, axis=0)
                else:
                    print(f"⚠️ Unsupported type in {key}: {type(values[0])}")
            except Exception as e:
                print(f"⚠️ Could not merge {key}: {e}")

        # Run each metric
        for name, metric in self._registry[token].items():
            result = None

            if not metric.accumulate:
                out[name] = None
                if verbose:
                    print(f"{name:25s}: (batch-only or not accumulated)")
                continue

            try:
                # Select args
                args = [inputs[arg] for arg in metric.arg_names]

                # Optional: Check for arg length mismatch
                if len(metric.arg_names) != len(args):
                    raise ValueError(f"Mismatch in arg_names vs. inputs for {name}")

                result = metric.fn(*args)

            except Exception as e:
                result = None
                if verbose:
                    print(f"{name:25s}: ❌ Error during computation: {e}")

            out[name] = result
            if verbose:
                if result is not None:
                    if isinstance(result, (float, int)):
                        print(f"{name:25s}: {result:.4f}")
                    elif isinstance(result, torch.Tensor):
                        result = result.item() if result.numel() == 1 else result.cpu().numpy()
                    elif isinstance(result, torch.Tensor) and result.ndim == 2 and result.shape[1] == 1 and torch.all(result == result[0]):
                        print(f"{name:25s}: tensor([[{result[0].item():.4f}]] * {result.shape[0]})")
                    elif isinstance(result, np.ndarray) and result.size == 1:
                        print(f"{name:25s}: {result.item():.4f}")
                    else:
                        print(f"{name:25s}: {result}")
                else:
                    print(f"{name:25s}: (batch-only or failed)")

        return out

    def add_batch(self, token: str, **kwargs):
        """
        Store inputs for accumulation by token.

        Args:
            token (str): Metric group name.
            kwargs: Named inputs used by metrics (e.g., y_pred, y_true).
        """
        for k, v in kwargs.items():
            if isinstance(v, (torch.Tensor, np.ndarray)):
                self.storage[token][k].append(v)

    @staticmethod
    def get_metric_bundles() -> dict[str, builtins.list[str]]:
            """
            Generate a dictionary of all metric names grouped by token (task type).

            Returns
            -------
            dict[str, list[str]]
                Dictionary of token → list of metric names.
            """
            return metrics_registry.list()


# ───── Global Singleton Registry ───── #
metrics_registry = MetricsRegistry()

# ───── Register Statistical Functions ───── #
metrics_registry.register(token="statistical", fn=statistical__mod_pred__mean, accumulate=False, arg_names=["y_pred"], name="statistical__mod_pred__mean")
metrics_registry.register(token="statistical", fn=statistical__mod_pred__variance_band, accumulate=False, arg_names=["y_pred"], name="statistical__mod_pred__variance_band")
metrics_registry.register(token="statistical", fn=statistical__mod_pred__standard_error, accumulate=False, arg_names=["y_pred"], name="statistical__mod_pred__standard_error")
metrics_registry.register(token="statistical", fn=statistical__mod_pred__quantiles, accumulate=False, arg_names=["y_pred", "quantiles"], name="statistical__mod_pred__quantiles")
metrics_registry.register(token="statistical", fn=statistical__mod_pred__predictive_mean, accumulate=False, arg_names=["y_pred"], name="statistical__mod_pred__predictive_mean")
metrics_registry.register(token="statistical", fn=statistical__mod_pred__plus_minus_sigma, accumulate=False, arg_names=["y_pred", "num_sigma"], name="statistical__mod_pred__plus_minus_sigma")
metrics_registry.register(token="statistical", fn=statistical__mod_pred__plus_minus_1_sigma, accumulate=False, arg_names=["y_pred"], name="statistical__mod_pred__plus_minus_1_sigma")
metrics_registry.register(token="statistical", fn=statistical__mod_pred__plus_minus_2_sigma, accumulate=False, arg_names=["y_pred"], name="statistical__mod_pred__plus_minus_2_sigma")
metrics_registry.register(token="statistical", fn=statistical__mod_pred__plus_minus_3_sigma, accumulate=False, arg_names=["y_pred"], name="statistical__mod_pred__plus_minus_3_sigma")
metrics_registry.register(token="statistical", fn=statistical__mod_pred__plus_minus_4_sigma, accumulate=False, arg_names=["y_pred"], name="statistical__mod_pred__plus_minus_4_sigma")
metrics_registry.register(token="statistical", fn=statistical__mod_pred__plus_minus_5_sigma, accumulate=False, arg_names=["y_pred"], name="statistical__mod_pred__plus_minus_5_sigma")

# ───── Register Metrics ───── #
#.register(self, token: str, fn: Callable, name: str = None, accumulate: bool = False)

# ───── Classification ─────
metrics_registry.register(token="classification", fn=accuracy, accumulate=True, arg_names=["y_pred", "y_true"], name="accuracy")
# metrics_registry.register(token="classification", fn=brier_score, accumulate=True, arg_names=["y_pred", "y_true"], name="brier_score")
metrics_registry.register(token="classification", fn=lambda y_pred, y_true: top_k_accuracy(y_pred, y_true, k=3), accumulate=True, arg_names=["y_pred", "y_true"], name="top_k_accuracy")

# ───── Regression ─────
metrics_registry.register(token="regression", fn=mse, accumulate=True, arg_names=["y_pred", "y_true"], name="mse")
metrics_registry.register(token="regression", fn=rmse, accumulate=True, arg_names=["y_pred", "y_true"], name="rmse")
metrics_registry.register(token="regression", fn=mae, accumulate=True, arg_names=["y_pred", "y_true"], name="mae")
metrics_registry.register(token="regression", fn=mape, accumulate=True, arg_names=["y_pred", "y_true", "eps"], name="mape")
metrics_registry.register(token="regression", fn=r2_score, accumulate=True, arg_names=["y_pred", "y_true"], name="r2_score")

# ───── UQ ─────
metrics_registry.register(token="uq", fn=nll_gaussian, accumulate=True, arg_names=["mean", "logvar", "y_true"], name="nll_gaussian")
metrics_registry.register(token="uq", fn=energy_score, accumulate=True, arg_names=["y_samples", "y_true"], name="energy_score")
metrics_registry.register(token="uq", fn=ece, accumulate=True, arg_names=["y_pred", "y_true", "n_bins"], name="ece")
metrics_registry.register(token="uq", fn=ace, accumulate=True, arg_names=["y_pred", "y_true", "n_bins"], name="ace")
metrics_registry.register(token="uq", fn=regression_ece, accumulate=True, arg_names=["y_pred_mean", "y_pred_std", "y_true", "n_bins"], name="regression_ece")
metrics_registry.register(token="uq", fn=elbo, accumulate=True, arg_names=["mu", "v", "alpha", "beta", "target", "kl_div"], name="elbo")
metrics_registry.register(token="uq", fn=evidence, accumulate=True, arg_names=["y_pred"], name="evidence")
metrics_registry.register(token="uq", fn=marginal_likelihood, accumulate=True, arg_names=["y_pred"], name="marginal_likelihood")
metrics_registry.register(token="uq", fn=picp, accumulate=True, arg_names=["y_lower", "y_upper", "y_true"], name="picp")
metrics_registry.register(token="uq", fn=mpiw, accumulate=True, arg_names=["y_lower", "y_upper"], name="mpiw")
metrics_registry.register(token="uq", fn=continuous_ranked_probability_score, accumulate=True, arg_names=["y_pred_mean", "pred_std", "y_true"], name="continuous_ranked_probability_score")
metrics_registry.register(token="uq", fn=kl_divergence_normal, accumulate=True, arg_names=["y_pred_mean", "pred_std", "ref_mean", "ref_std"], name="kl_divergence_normal")
metrics_registry.register(token="uq", fn=mean_pred_variance, accumulate=True, arg_names=["pred_std"], name="mean_pred_variance")
metrics_registry.register(token="uq", fn=predictive_entropy, accumulate=True, arg_names=["pred_probs"], name="predictive_entropy")
metrics_registry.register(token="uq", fn=mutual_information, accumulate=True, arg_names=["pred_probs"], name="mutual_information")
metrics_registry.register(token="uq", fn=epistemic_variance, accumulate=True, arg_names=["mc_preds"], name="epistemic_variance")
metrics_registry.register(token="uq", fn=aleatoric_variance, accumulate=True, arg_names=["pred_std"], name="aleatoric_variance")
metrics_registry.register(token="uq", fn=calibration_error, accumulate=True, arg_names=["y_true", "y_pred", "uncertainty", "n_bins"], name="calibration_error")
metrics_registry.register(token="uq", fn=uda, accumulate=True, arg_names=["pred_uncert_epistemic", "pred_uncert_aleatoric", "true_errors"], name="uda")
metrics_registry.register(token="uq", fn=ncg, accumulate=True, arg_names=["confidence_scores", "baseline_confidence_scores"], name="ncg")
metrics_registry.register(token="uq", fn=meta_metric__bnn_ednn, accumulate=True, arg_names=["uda", "meta_calibration_score", "corr_err_epistemic", "ncg"], name="meta_metric__bnn_ednn")

# ───── Probabilistic ─────
metrics_registry.register(token="probabilistic", fn=nll_gaussian, accumulate=True, arg_names=["mean", "logvar", "target"], name="nll_gaussian")
metrics_registry.register(token="probabilistic", fn=energy_score, accumulate=True, arg_names=["samples", "target"], name="energy_score")


# ───── Example Usage ───── #

if __name__ == "__main__":
    torch.manual_seed(42)

    print("🔢 Classification Metrics")
    cls_metrics = metrics_registry.get("classification", ["accuracy", "top3_acc"])
    for _ in range(3):
        y_pred = torch.randn(32, 10)
        y_true = torch.randint(0, 10, (32,))
        for metric in cls_metrics.values():
            metric(y_pred, y_true)
    metrics_registry.report("classification")

    print("\n📈 Regression Metrics")
    reg_metrics = metrics_registry.get("regression", ["mse", "rmse", "mae", "mape", "r2_score"])
    for _ in range(3):
        y_pred = torch.randn(64, 1)
        y_true = y_pred + 0.1 * torch.randn_like(y_pred)
        for metric in reg_metrics.values():
            metric(y_pred, y_true)
    metrics_registry.report("regression")

    print("\n📊 Uncertainty Quantification (UQ) Metrics")
    y_true = torch.randn(16, 1)
    mean = y_true + 0.05 * torch.randn_like(y_true)
    logvar = torch.zeros_like(y_true)
    y_samples = mean.unsqueeze(1) + 0.1 * torch.randn(16, 100, 1)
    metrics_registry.all("uq")["nll_gaussian"](mean, logvar, y_true)
    metrics_registry.all("uq")["energy_score"](y_samples, y_true)
    metrics_registry.report("uq")
