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
    # Classification
    accuracy
    top_k_accuracy

    # Regression
    mse
    rmse
    mae
    mape
    r2_score

    # Probabilistic Regression
    nll_gaussian
    energy_score
    crps                 # Continuous Ranked Probability Score
    kl_divergence        # Kullback-Leibler Divergence (for Gaussian predictions)

    # Uncertainty Quantification (UQ)
    ece
    regression_ece
    ace
    brier_score
    elbo
    evidence
    marginal_likelihood
    picp
    mpiw

    # Predictive Uncertainty Metrics
    mean_pred_variance   # Average predictive variance
    predictive_entropy   # Predictive entropy (for classification)
    mutual_information   # Epistemic uncertainty via mutual information (BALD)

    # Variance Decomposition
    epistemic_variance   # Epistemic uncertainty from MC predictions
    aleatoric_variance   # Aleatoric uncertainty (model-internal)

    total_uncertainty

    uda
    ncg
    meta_metric__bnn_ednn

Usage:
    Register metrics for a task using `Metrics.register(...)` and
    compute them via `metric(y_pred, y_true)` or via accumulation with `metric(...)` + `metric.compute()`.
"""

import matplotlib.pyplot as plt
import numpy as np
import properscoring as ps
import torch
import torchmetrics.functional as MF
import torch.nn.functional as F

from BA__Programmierung.ml.losses.evidential_loss import evidential_loss
from BA__Programmierung.util.singleton import Singleton
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple, Union


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

def mse(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Mean Squared Error."""
    return MF.mean_squared_error(y_pred, y_true)


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


def ece(y_pred, y_true, n_bins=10):
    """
    Expected Calibration Error (ECE) für Regressionsmodelle.
    """
    confidences = torch.max(y_pred, dim=1).values
    predictions = torch.argmax(y_pred, dim=1)
    accuracies = predictions.eq(y_true)

    ece = torch.zeros(1, device=y_pred.device)
    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=y_pred.device)

    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if mask.any():
            acc = accuracies[mask].float().mean()
            conf = confidences[mask].mean()
            ece += (conf - acc).abs() * mask.float().mean()

    return ece.item()


def regression_ece(y_pred_mean, y_pred_std, y_true, n_bins=10):
    """
    Calibration Error für Regression, basierend auf Vorhersage-Unsicherheit.

    Args:
        y_pred_mean (Tensor): Erwartete Werte.
        y_pred_std (Tensor): Standardabweichungen (Unsicherheit).
        y_true (Tensor): Wahre Werte.
        n_bins (int): Anzahl Konfidenzintervalle.

    Returns:
        float: Expected Calibration Error.
    """
    probs = torch.distributions.Normal(y_pred_mean, y_pred_std).cdf(y_true)
    errors = torch.abs(probs - 0.5) * 2  # Erwartungswert bei perfekter Kalibrierung: 0.5

    bins = torch.linspace(0, 1, n_bins + 1)
    bin_ids = torch.bucketize(probs, bins)

    ece = 0.0
    for i in range(1, n_bins + 1):
        mask = bin_ids == i
        if mask.any():
            bin_conf = (bins[i] + bins[i - 1]) / 2
            bin_acc = (probs[mask] < bins[i]).float().mean().item()
            ece += abs(bin_acc - bin_conf) * mask.float().mean().item()
    return ece


def ace(y_pred, y_true, n_bins=10):
    """
    Adaptive Calibration Error (ACE).
    """
    confidences = torch.max(y_pred, dim=1).values
    predictions = torch.argmax(y_pred, dim=1)
    accuracies = predictions.eq(y_true)

    sorted_conf, sorted_idx = torch.sort(confidences)
    sorted_acc = accuracies[sorted_idx]
    bins = torch.chunk(sorted_conf, n_bins)
    acc_bins = torch.chunk(sorted_acc, n_bins)

    ace = 0
    for b_conf, b_acc in zip(bins, acc_bins):
        if len(b_conf) > 0:
            acc = b_acc.float().mean()
            conf = b_conf.mean()
            ace += (acc - conf).abs()
    return (ace / n_bins).item()


def brier_score(y_pred, y_true):
    """
    Brier Score für probabilistische Klassifikation.
    """
    y_true_oh = F.one_hot(y_true, num_classes=y_pred.shape[1]).float()
    return torch.mean((y_pred - y_true_oh) ** 2).item()


def elbo(y_pred, y_true, kl_div):
    """
    Evidence Lower Bound Loss (negativer ELBO).
    """
    nll = evidential_loss(y_pred, y_true)
    return (nll + kl_div).item()


def evidence(y_pred):
    """
    Erwartete Evidenz: Summe der Alphas (Dirichlet-Parameter).
    """
    alpha = y_pred + 1  # Ensure alphas > 1
    return alpha.sum(dim=1).mean().item()


def marginal_likelihood(y_pred, y_true):
    """
    Marginale Likelihood-Schätzung.
    """
    alpha = y_pred + 1
    S = torch.sum(alpha, dim=1, keepdim=True)
    likelihood = torch.exp(torch.lgamma(S) - torch.lgamma(alpha).sum(dim=1, keepdim=True))
    return likelihood.mean().item()


def picp(y_lower, y_upper, y_true):
    """
    Prediction Interval Coverage Probability (PICP).
    """
    covered = ((y_true >= y_lower) & (y_true <= y_upper)).float()
    return covered.mean().item()


def mpiw(y_lower, y_upper):
    """
    Mean Prediction Interval Width (MPIW).
    """
    return (y_upper - y_lower).mean().item()


# Continuous Ranked Probability Score (CRPS)
def continuous_ranked_probability_score(pred_mean: np.ndarray, pred_std: np.ndarray, y_true: np.ndarray) -> float:
    """Compute CRPS for Gaussian predictions"""
    return np.mean(ps.crps_gaussian(y_true, mu=pred_mean, sig=pred_std))


# # KL-Divergenz
def kl_divergence_normal(pred_mean: np.ndarray, pred_std: np.ndarray, ref_mean: np.ndarray, ref_std: np.ndarray) -> float:
    """KL divergence between two Gaussians (prediction vs reference)"""
    var_ratio = (pred_std ** 2) / (ref_std ** 2)
    kl = np.log(ref_std / pred_std) + (var_ratio + (pred_mean - ref_mean) ** 2 / (ref_std ** 2) - 1) / 2
    return np.mean(kl)


# Mittlere Vorhergesagte Varianz
def mean_pred_variance(pred_std: np.ndarray, **kwargs) -> float:
    """Mean predictive variance (aleatoric + epistemic)"""
    return float(np.mean(pred_std ** 2))


# Entropie der Vorhersage (Klassifikation)
def predictive_entropy(pred_probs: np.ndarray, **kwargs) -> float:
    """Entropy of the predictive distribution (for classification)"""
    entropy = -np.sum(pred_probs * np.log(pred_probs + 1e-12), axis=1)
    return float(np.mean(entropy))


# Epistemic uncertainty
def mutual_information(pred_probs: np.ndarray, **kwargs) -> float:
    """Epistemic uncertainty estimate via Mutual Information"""
    mean_probs = np.mean(pred_probs, axis=0)
    entropy_mean = -np.sum(mean_probs * np.log(mean_probs + 1e-12))
    mean_entropy = np.mean([-np.sum(p * np.log(p + 1e-12)) for p in pred_probs])
    return float(entropy_mean - mean_entropy)


# Varianz-Dekomposition, Epistemische Varianz numpy
def epistemic_variance(mc_preds: np.ndarray, **kwargs) -> float:
    """Epistemic variance: Varianz über Modelle (MC-Samples)"""
    return float(np.mean(np.var(mc_preds, axis=0)))


# Varianz-Dekomposition, Epistemische Varianz torchmetrics
def epistemic_uncertainty(predictions: torch.Tensor) -> torch.Tensor:
    """
    Estimate epistemic uncertainty from a set of predictions (e.g., MC Dropout).
    Expects shape: (num_samples, batch_size)
    """
    return predictions.std(dim=0)


# Varianz-Dekomposition, Aleartorische Varianz
def aleatoric_variance(pred_std: np.ndarray, **kwargs) -> float:
    """Aleatoric variance: Modellinterne Unsicherheit"""
    return float(np.mean(pred_std ** 2))


# Varianz-Dekomposition, Aleatorische Varianz torchmetrics
def aleatoric_uncertainty(variance_pred: torch.Tensor) -> torch.Tensor:
    """
    Aleatoric uncertainty: predicted variance per sample.
    """
    return torch.sqrt(variance_pred)


# Totale Unsicherheit
def total_uncertainty(epistemic: torch.Tensor, aleatoric: torch.Tensor) -> torch.Tensor:
    """
    Total uncertainty = sqrt(aleatoric^2 + epistemic^2)
    """
    return torch.sqrt(epistemic ** 2 + aleatoric ** 2)


# Kalibrierungsfehler
def calibration_error(y_true: torch.Tensor, y_pred: torch.Tensor, uncertainty: torch.Tensor, num_bins: int = 10) -> torch.Tensor:
    """
    Empirical calibration error: fraction of true values within predicted uncertainty intervals.
    """
    assert y_true.shape == y_pred.shape == uncertainty.shape, "Shapes must match."
    z = torch.abs(y_pred - y_true) / (uncertainty + 1e-8)
    inside = (z < 1.0).float()
    return 1.0 - inside.mean()


def interval_coverage(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    uncertainty: torch.Tensor,
    alphas: List[float] = [0.05, 0.1, 0.2]
) -> Tuple[List[float], List[float]]:
    """
    Computes actual coverage of predictive intervals for various alphas (1 - confidence).
    
    Parameters
    ----------
    y_true : torch.Tensor
        Ground truth targets.
    y_pred : torch.Tensor
        Predicted means.
    uncertainty : torch.Tensor
        Predictive standard deviation (not variance!).
    alphas : list of float
        Alpha values corresponding to confidence levels (e.g., 0.05 for 95%).

    Returns
    -------
    levels : list of float
        Confidence levels (1 - alpha).
    coverages : list of float
        Actual fraction of targets falling within predicted interval.
    """
    from scipy.stats import norm

    levels = [1 - a for a in alphas]
    coverages = []

    for alpha in alphas:
        z = norm.ppf(1 - alpha / 2)
        lower = y_pred - z * uncertainty
        upper = y_pred + z * uncertainty
        in_interval = ((y_true >= lower) & (y_true <= upper)).float()
        coverage = in_interval.mean().item()
        coverages.append(coverage)

    return levels, coverages


def uda(pred_uncert_epistemic, pred_uncert_aleatoric, true_errors):
    """
    Compute Uncertainty Decomposition Accuracy (UDA).

    Args:
        pred_uncert_epistemic (np.ndarray): Predicted epistemic uncertainty per sample.
        pred_uncert_aleatoric (np.ndarray): Predicted aleatoric uncertainty per sample.
        true_errors (np.ndarray): True errors or error components per sample.

    Returns:
        float: UDA score (e.g., correlation or accuracy)
    """
    # For example, you might correlate epistemic uncertainty with error magnitude
    # This is a placeholder implementation — adjust according to your setup
    
    # Normalize uncertainties and errors
    epistemic_norm = (pred_uncert_epistemic - np.min(pred_uncert_epistemic)) / (np.ptp(pred_uncert_epistemic) + 1e-8)
    aleatoric_norm = (pred_uncert_aleatoric - np.min(pred_uncert_aleatoric)) / (np.ptp(pred_uncert_aleatoric) + 1e-8)
    errors_norm = (true_errors - np.min(true_errors)) / (np.ptp(true_errors) + 1e-8)
    
    # Compute correlation between epistemic uncertainty and error as proxy for UDA
    corr_epistemic = np.corrcoef(epistemic_norm, errors_norm)[0, 1]
    
    # Compute correlation between aleatoric uncertainty and error (or noise)
    corr_aleatoric = np.corrcoef(aleatoric_norm, errors_norm)[0, 1]
    
    # Average or weighted average of correlations as UDA
    uda_score = 0.5 * corr_epistemic + 0.5 * corr_aleatoric
    
    # Clamp score between 0 and 1 (optional)
    uda_score = max(0.0, min(1.0, uda_score))
    
    return uda_score


def ncg(confidence_scores, baseline_confidence_scores):
    """
    Compute Normalized Confidence Gain (NCG).

    Args:
        confidence_scores (np.ndarray): Confidence scores after calibration.
        baseline_confidence_scores (np.ndarray): Baseline confidence scores.

    Returns:
        float: NCG value between 0 and 1.
    """
    # For instance, use mean confidence improvement normalized by max possible gain
    gain = np.mean(confidence_scores) - np.mean(baseline_confidence_scores)
    
    # Normalize by max possible gain (e.g., 1 - baseline_mean_confidence)
    max_gain = 1.0 - np.mean(baseline_confidence_scores)
    
    if max_gain == 0:
        return 0.0
    
    ncg = gain / max_gain
    ncg = max(0.0, min(1.0, ncg))  # clamp between 0 and 1
    
    return ncg



def meta_metric__bnn_ednn(UDA, meta_calibration_score, corr_err_epistemic, NCG):
    """
    Computes the meta-metric as a weighted average of four submetrics.

    Args:
        UDA (float): Uncertainty Decomposition Accuracy
        meta_calibration_score (float): Calibration score (e.g. ECE, ACE)
        corr_err_epistemic (float): Correlation between error and epistemic uncertainty
        NCG (float): Normalized Confidence Gain

    Returns:
        float: Meta-metric value
    """
    return (
        0.25 * UDA +
        0.25 * meta_calibration_score +
        0.25 * corr_err_epistemic +
        0.25 * (1 - NCG)
    )


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
Metrics.register("uq", ece, accumulate=True)
Metrics.register("uq", ace, accumulate=True)
Metrics.register("uq", regression_ece, accumulate=True)
Metrics.register("uq", brier_score, accumulate=True)
Metrics.register("uq", elbo, accumulate=True)
Metrics.register("uq", evidence, accumulate=True)
Metrics.register("uq", marginal_likelihood, accumulate=True)
Metrics.register("uq", picp, accumulate=True)
Metrics.register("uq", mpiw, accumulate=True)

Metrics.register("uq", continuous_ranked_probability_score, accumulate=True)
Metrics.register("uq", kl_divergence_normal, accumulate=True)

Metrics.register("uq", mean_pred_variance, accumulate=True)
Metrics.register("uq", predictive_entropy, accumulate=True)
Metrics.register("uq", mutual_information, accumulate=True)

Metrics.register("uq", epistemic_variance, accumulate=True)
Metrics.register("uq", aleatoric_variance, accumulate=True)

# Metrics.register("uq", latent_function, accumulate=True)
# Metrics.register("uq", cov_train, name="cov_train", accumulate=True)
# Metrics.register("uq", cov_test, name="cov_test", accumulate=True)
# Metrics.register("uq", cross_cov, name="cross_cov", accumulate=True)

# Probabilistic Regression
Metrics.register("probabilistic", nll_gaussian, accumulate=True)
Metrics.register("probabilistic", energy_score, accumulate=True)
Metrics.register("probabilistic", lambda mean, std, y: continuous_ranked_probability_score(mean.cpu().numpy(), std.cpu().numpy(), y.cpu().numpy()), name="crps", accumulate=True)
Metrics.register("probabilistic", lambda mean, std, ref_m, ref_s: kl_divergence_normal(mean, std, ref_m, ref_s), name="kl_div", accumulate=True)

Metrics.register("uq", mean_pred_variance, accumulate=True)
Metrics.register("uq", predictive_entropy, accumulate=True)
Metrics.register("uq", mutual_information, accumulate=True)
Metrics.register("uq", epistemic_variance, accumulate=True)
Metrics.register("uq", aleatoric_variance, accumulate=True)
Metrics.register("uq", calibration_error, accumulate=True)

Metrics.register("uq", uda, accumulate=True)
Metrics.register("uq", ncg, accumulate=True)
Metrics.register("uq", meta_metric__bnn_ednn, accumulate=True)


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
