# BA__Projekt/BA__Programmierung/ml/metrics/metrics_registry_definitions.py

"""
metrics_registry_definitions
============================

This module defines a comprehensive set of metrics for evaluating models in
classification, regression, and uncertainty quantification (UQ) contexts.

Metrics are implemented as standalone functions and are compatible with the
`MetricsRegistry` system to enable accumulation and standardized reporting.

Includes:
- Classification: accuracy, top-k accuracy, ECE, ACE, Brier score
- Regression: MSE, RMSE, MAE, MAPE, R², CRPS
- UQ: NLL, ELBO, energy score, predictive entropy, mutual information, UDA, NCG, meta-metric, etc.

External Libraries Used:
- torchmetrics
- numpy
- properscoring
- scipy.stats
"""
import torch
import torch.nn.functional as tnf
import torchmetrics.functional as tmf
import numpy as np
import pandas as pd
import properscoring as ps
from scipy.stats import norm

# If you reference this elsewhere, keep or adjust this import
from BA__Programmierung.ml.losses.evidential_loss import evidential_loss


# ───── Helper Functions ───── #

def to_numpy(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x

# ───── Metric Functions ───── #

def accuracy(y_pred, y_true) -> float:
    """
    Computes classification accuracy.

    Returns:
        float: Accuracy in [0, 1].
    """
    return (torch.argmax(y_pred, dim=1) == y_true).float().mean().item()


# ───── Statistical Functions: Model Prediction ───── #

def statistical__mod_pred__mean(y_pred: torch.Tensor) -> float:
    """
    Computes the statistical function, which could be the raw output of a model.
    
    Args:
        y_pred (torch.Tensor): Predicted values by the model.
        
    Returns:
        float: The mean of the predictions.
    """
    return y_pred.mean().item()


def statistical__mod_pred__variance_band(y_pred: torch.Tensor) -> float:
    """
    Computes the variance-band for the predictions.
    
    Args:
        y_pred (torch.Tensor): Predicted values by the model.
        
    Returns:
        float: The variance of the predictions.
    """
    return y_pred.var(dim=0).mean().item()  # Variance along the first dimension (batch)


def statistical__mod_pred__standard_error(y_pred: torch.Tensor) -> float:
    """
    Computes the standard error for the predictions.
    
    Args:
        y_pred (torch.Tensor): Predicted values by the model.
        
    Returns:
        float: The standard error of the predictions.
    """
    return (y_pred.std(dim=0) / torch.sqrt(torch.tensor(len(y_pred)))).mean().item()


def statistical__mod_pred__quantiles(y_pred: torch.Tensor, quantiles: list = [0.25, 0.5, 0.75]) -> dict:
    """
    Computes the quantiles (e.g., 25th, 50th, 75th percentiles) of the predictions.
    
    Args:
        y_pred (torch.Tensor): Predicted values by the model.
        quantiles (list): List of quantiles to compute (default is [0.25, 0.5, 0.75]).
        
    Returns:
        dict: Dictionary with quantiles as keys and their corresponding values.
    """
    quantile_values = {}
    for q in quantiles:
        quantile_values[f'quantile_{int(q*100)}'] = torch.quantile(y_pred, q).item()
    return quantile_values


def statistical__mod_pred__predictive_mean(y_pred: torch.Tensor) -> float:
    """
    Computes the statistical predictive mean for the predictions.
    
    Args:
        y_pred (torch.Tensor): Predicted values by the model.
        
    Returns:
        float: The mean of the predictions.
    """
    return y_pred.mean().item()


def statistical__mod_pred__plus_minus_sigma(y_pred: torch.Tensor, num_sigma: int = 2) -> tuple:
    """
    Computes the interval defined by mean +/- num_sigma * standard deviation.
    
    Args:
        y_pred (torch.Tensor): Predicted values by the model.
        num_sigma (int): Number of standard deviations for the interval (default is 2).
        
    Returns:
        tuple: Lower and upper bounds of the interval.
    """
    mean = y_pred.mean()
    std_dev = y_pred.std()
    lower = mean - num_sigma * std_dev
    upper = mean + num_sigma * std_dev
    return lower.item(), upper.item()


# ───── Extensions for 1 to 5 Sigmas ───── #

def statistical__mod_pred__plus_minus_1_sigma(y_pred: torch.Tensor) -> tuple:
    """Computes the interval for mean +/- 1 standard deviation."""
    return statistical__mod_pred__plus_minus_sigma(y_pred, num_sigma=1)


def statistical__mod_pred__plus_minus_2_sigma(y_pred: torch.Tensor) -> tuple:
    """Computes the interval for mean +/- 2 standard deviations."""
    return statistical__mod_pred__plus_minus_sigma(y_pred, num_sigma=2)


def statistical__mod_pred__plus_minus_3_sigma(y_pred: torch.Tensor) -> tuple:
    """Computes the interval for mean +/- 3 standard deviations."""
    return statistical__mod_pred__plus_minus_sigma(y_pred, num_sigma=3)


def statistical__mod_pred__plus_minus_4_sigma(y_pred: torch.Tensor) -> tuple:
    """Computes the interval for mean +/- 4 standard deviations."""
    return statistical__mod_pred__plus_minus_sigma(y_pred, num_sigma=4)


def statistical__mod_pred__plus_minus_5_sigma(y_pred: torch.Tensor) -> tuple:
    """Computes the interval for mean +/- 5 standard deviations."""
    return statistical__mod_pred__plus_minus_sigma(y_pred, num_sigma=5)

# ───── Statistical Functions: Real Data ───── #

def statistical__real_data__mean(real_data: pd.Series) -> float:
    """
    Computes the statistical mean for the real data.
    
    Args:
        real_data (pd.Series): Real data values from a column in a DataFrame.
        
    Returns:
        float: The mean of the real data.
    """
    return real_data.mean()


def statistical__real_data__variance_band(real_data: pd.Series) -> float:
    """
    Computes the variance-band for the real data.
    
    Args:
        real_data (pd.Series): Real data values from a column in a DataFrame.
        
    Returns:
        float: The variance of the real data.
    """
    return real_data.var()


def statistical__real_data__standard_error(real_data: pd.Series) -> float:
    """
    Computes the standard error for the real data.
    
    Args:
        real_data (pd.Series): Real data values from a column in a DataFrame.
        
    Returns:
        float: The standard error of the real data.
    """
    return real_data.std() / (len(real_data) ** 0.5)


def statistical__real_data__quantiles(real_data: pd.Series, quantiles: list = [0.25, 0.5, 0.75]) -> dict:
    """
    Computes the quantiles (e.g., 25th, 50th, 75th percentiles) for the real data.
    
    Args:
        real_data (pd.Series): Real data values from a column in a DataFrame.
        quantiles (list): List of quantiles to compute (default is [0.25, 0.5, 0.75]).
        
    Returns:
        dict: Dictionary with quantiles as keys and their corresponding values.
    """
    quantile_values = {}
    for q in quantiles:
        quantile_values[f'quantile_{int(q*100)}'] = real_data.quantile(q)
    return quantile_values


def statistical__real_data__predictive_mean(real_data: pd.Series) -> float:
    """
    Computes the statistical predictive mean for the real data.
    
    Args:
        real_data (pd.Series): Real data values from a column in a DataFrame.
        
    Returns:
        float: The mean of the real data.
    """
    return real_data.mean()


def statistical__real_data__plus_minus_sigma(real_data: pd.Series, num_sigma: int = 2) -> tuple:
    """
    Computes the interval defined by mean +/- num_sigma * standard deviation for real data.
    
    Args:
        real_data (pd.Series): Real data values from a column in a DataFrame.
        num_sigma (int): Number of standard deviations for the interval (default is 2).
        
    Returns:
        tuple: Lower and upper bounds of the interval.
    """
    mean = real_data.mean()
    std_dev = real_data.std()
    lower = mean - num_sigma * std_dev
    upper = mean + num_sigma * std_dev
    return lower, upper


# ───── Extensions for 1 to 5 Sigmas ───── #

def statistical__real_data__plus_minus_1_sigma(real_data: pd.Series) -> tuple:
    """Computes the interval for mean +/- 1 standard deviation for real data."""
    return statistical__real_data__plus_minus_sigma(real_data, num_sigma=1)


def statistical__real_data__plus_minus_2_sigma(real_data: pd.Series) -> tuple:
    """Computes the interval for mean +/- 2 standard deviations for real data."""
    return statistical__real_data__plus_minus_sigma(real_data, num_sigma=2)


def statistical__real_data__plus_minus_3_sigma(real_data: pd.Series) -> tuple:
    """Computes the interval for mean +/- 3 standard deviations for real data."""
    return statistical__real_data__plus_minus_sigma(real_data, num_sigma=3)


def statistical__real_data__plus_minus_4_sigma(real_data: pd.Series) -> tuple:
    """Computes the interval for mean +/- 4 standard deviations for real data."""
    return statistical__real_data__plus_minus_sigma(real_data, num_sigma=4)


def statistical__real_data__plus_minus_5_sigma(real_data: pd.Series) -> tuple:
    """Computes the interval for mean +/- 5 standard deviations for real data."""
    return statistical__real_data__plus_minus_sigma(real_data, num_sigma=5)

# ───── Metric Functions ───── #

def top_k_accuracy(y_pred, y_true, k=3) -> float:
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


def mse(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Mean Squared Error."""
    return float(tmf.mean_squared_error(y_pred, y_true))


def rmse(y_pred, y_true):
    """Root Mean Squared Error."""
    return torch.sqrt(tnf.mse_loss(y_pred, y_true)).item()


def mae(y_pred, y_true):
    """Mean Absolute Error."""
    return tnf.l1_loss(y_pred, y_true).item()


def mape(y_pred, y_true, eps=1e-8) -> float:
    """
    Mean Absolute Percentage Error.

    Args:
        eps (float): Small value to prevent division by zero.

    Returns:
        float: MAPE value.
    """
    return (torch.abs((y_true - y_pred) / (y_true + eps)).mean() * 100).item()


def r2_score(y_pred, y_true) -> float:
    """
    Coefficient of Determination (R²).

    Returns:
        float: R² score.
    """
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    return (1 - ss_res / ss_tot).item()


def nll_gaussian(mean, logvar, target) -> float:
    """
    Negative Log-Likelihood for a Gaussian distribution.

    Returns:
        float: NLL value.
    """
    precision = torch.exp(-logvar)
    return 0.5 * torch.mean(precision * (target - mean) ** 2 + logvar).item()


def energy_score(y_samples, y_true) -> float:
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


def ece(y_pred, y_true, n_bins=10) -> float:
    """
    Expected Calibration Error (ECE) for classification.
    """
    if isinstance(n_bins, torch.Tensor):
        n_bins = int(n_bins.flatten()[0].item())

    confidences = torch.max(y_pred, dim=1).values
    predictions = torch.argmax(y_pred, dim=1)
    accuracies = predictions.eq(y_true)

    ece = torch.zeros(1, device=y_pred.device)
    bin_boundaries = torch.linspace(0, 1, steps=n_bins + 1, device=y_pred.device)

    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if mask.any():
            acc = accuracies[mask].float().mean()
            conf = confidences[mask].mean()
            ece += (conf - acc).abs() * mask.float().mean()

    return ece.item()


def regression_ece(y_pred_mean, y_pred_std, y_true, n_bins=10) -> float:
    if isinstance(n_bins, torch.Tensor):
        n_bins = int(n_bins.flatten()[0].item())  # safely extract scalar

    probs = torch.distributions.Normal(y_pred_mean, y_pred_std).cdf(y_true)
    errors = torch.abs(probs - 0.5) * 2

    bins = torch.linspace(0, 1, steps=n_bins + 1)
    bin_ids = torch.bucketize(probs, bins)

    ece = 0.0
    for i in range(1, n_bins + 1):
        mask = bin_ids == i
        if mask.any():
            bin_conf = (bins[i] + bins[i - 1]) / 2
            bin_acc = (probs[mask] < bins[i]).float().mean().item()
            ece += abs(bin_acc - bin_conf) * mask.float().mean().item()

    return float(ece)


def ace(y_pred, y_true, n_bins=10) -> float:
    """
    Adaptive Calibration Error (ACE).
    """
    if isinstance(n_bins, torch.Tensor):
        n_bins = int(n_bins.flatten()[0].item())

    confidences = torch.max(y_pred, dim=1).values
    predictions = torch.argmax(y_pred, dim=1)
    accuracies = predictions.eq(y_true)

    sorted_conf, sorted_idx = torch.sort(confidences)
    sorted_acc = accuracies[sorted_idx]
    conf_chunks = torch.chunk(sorted_conf, n_bins)
    acc_chunks = torch.chunk(sorted_acc, n_bins)

    ace = 0
    for b_conf, b_acc in zip(conf_chunks, acc_chunks):
        if b_conf.numel() > 0:
            acc = b_acc.float().mean()
            conf = b_conf.mean()
            ace += (acc - conf).abs()
    return (ace / n_bins).item()


def brier_score(y_pred, y_true) -> float:
    """
    Brier Score für probabilistische Klassifikation.
    """
    y_true = y_true.long()
    y_true_oh = tnf.one_hot(y_true, num_classes=y_pred.shape[1]).float()
    return torch.mean((y_pred - y_true_oh) ** 2).item()

def elbo(mu, v, alpha, beta, target, kl_div):
    # Sanitize inputs
    v = v.clamp(min=1e-6)
    alpha = alpha.clamp(min=1.01)
    beta = beta.clamp(min=1e-6)

    # Optional: debug print
    if torch.isnan(v).any() or torch.isnan(alpha).any() or torch.isnan(beta).any():
        print("⚠️ NaNs in ELBO input tensors")
        return float("nan")

    try:
        nll = evidential_loss(mu, v, alpha, beta, target)
        return (nll + kl_div.float().mean()).item()
    except Exception as e:
        print(f"❌ Exception in ELBO: {e}")
        return float("nan")

# def elbo(mu, v, alpha, beta, target, kl_div):
#     """
#     Evidence Lower Bound Loss (negativer ELBO).
#     """
#     nll = evidential_loss(mu, v, alpha, beta, target)
#     return (nll + kl_div.mean()).item()

# def elbo(y_pred, y_true, kl_div) -> float:
#     """
#     Evidence Lower Bound Loss (negativer ELBO).
#     """
#     nll = evidential_loss(y_pred, y_true)
#     return (nll + kl_div.mean()).item()


def evidence(y_pred) -> float:
    """
    Erwartete Evidenz: Summe der Alphas (Dirichlet-Parameter).
    """
    alpha = y_pred + 1  # Ensure alphas > 1
    return alpha.sum(dim=1).mean().item()


def marginal_likelihood(y_pred) -> float:
    """
    Marginale Likelihood-Schätzung.
    """
    alpha = y_pred + 1
    S = torch.sum(alpha, dim=1, keepdim=True)
    likelihood = torch.exp(torch.lgamma(S) - torch.lgamma(alpha).sum(dim=1, keepdim=True))
    return likelihood.mean().item()


def picp(y_lower, y_upper, y_true) -> float:
    """
    Prediction Interval Coverage Probability (PICP).
    """
    covered = ((y_true >= y_lower) & (y_true <= y_upper)).float()
    return covered.mean().item()


def mpiw(y_lower, y_upper) -> float:
    """
    Mean Prediction Interval Width (MPIW).
    """
    return (y_upper - y_lower).mean().item()


# Continuous Ranked Probability Score (CRPS)
def continuous_ranked_probability_score(y_pred_mean: np.ndarray, pred_std: np.ndarray, y_true: np.ndarray) -> float:
    """Compute CRPS for Gaussian predictions"""
    y_pred_mean = to_numpy(y_pred_mean)
    pred_std = to_numpy(pred_std)
    y_true = to_numpy(y_true)
    return np.mean(ps.crps_gaussian(y_true, mu=y_pred_mean, sig=pred_std))


# # KL-Divergenz
def kl_divergence_normal(y_pred_mean: np.ndarray, pred_std: np.ndarray, ref_mean: np.ndarray, ref_std: np.ndarray) -> float:
    """KL divergence between two Gaussians (prediction vs reference)"""
    y_pred_mean = to_numpy(y_pred_mean)
    pred_std = to_numpy(pred_std)
    ref_mean = to_numpy(ref_mean)
    ref_std = to_numpy(ref_std)
    var_ratio = (pred_std ** 2) / (ref_std ** 2)
    kl = np.log(ref_std / pred_std) + (var_ratio + (y_pred_mean - ref_mean) ** 2 / (ref_std ** 2) - 1) / 2
    return np.mean(kl)


# Mittlere Vorhergesagte Varianz
def mean_pred_variance(pred_std: np.ndarray) -> float:
    """Mean predictive variance (aleatoric + epistemic)"""
    pred_std = to_numpy(pred_std)
    return float(np.mean(pred_std ** 2))


# Entropie der Vorhersage (Klassifikation)
def predictive_entropy(pred_probs: np.ndarray) -> float:
    """Entropy of the predictive distribution (for classification)"""
    pred_probs = to_numpy(pred_probs)
    entropy = -np.sum(pred_probs * np.log(pred_probs + 1e-12), axis=1)
    return float(np.mean(entropy))


# Epistemic uncertainty
def mutual_information(pred_probs: np.ndarray) -> float:
    """Epistemic uncertainty estimate via Mutual Information"""
    pred_probs = to_numpy(pred_probs)
    mean_probs = np.mean(pred_probs, axis=0)
    entropy_mean = -np.sum(mean_probs * np.log(mean_probs + 1e-12))
    mean_entropy = np.mean([-np.sum(p * np.log(p + 1e-12)) for p in pred_probs])
    return float(entropy_mean - mean_entropy)


# Varianz-Dekomposition, Epistemische Varianz numpy
def epistemic_variance(mc_preds: np.ndarray) -> float:
    """Epistemic variance: Varianz über Modelle (MC-Samples)"""
    mc_preds = to_numpy(mc_preds)
    return float(np.mean(np.var(mc_preds, axis=0)))


# Varianz-Dekomposition, Epistemische Varianz torchmetrics
def epistemic_uncertainty(predictions: torch.Tensor) -> torch.Tensor:
    """
    Estimate epistemic uncertainty from a set of predictions (e.g., MC Dropout).
    Expects shape: (num_samples, batch_size)
    """
    return predictions.std(dim=0)


# Varianz-Dekomposition, Aleartorische Varianz
def aleatoric_variance(pred_std: np.ndarray) -> float:
    """Aleatoric variance: Modellinterne Unsicherheit"""
    pred_std = to_numpy(pred_std)
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
def calibration_error(y_true: torch.Tensor, y_pred: torch.Tensor, uncertainty: torch.Tensor, n_bins: int = 10) -> torch.Tensor:
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
    alphas: list[float] = [0.05, 0.1, 0.2]
) -> tuple[list[float], list[float]]:
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


def uda(pred_uncert_epistemic, pred_uncert_aleatoric, true_errors) -> float:
    """
    Compute Uncertainty Decomposition Accuracy (UDA).

    Args:
        pred_uncert_epistemic (np.ndarray): Predicted epistemic uncertainty per sample.
        pred_uncert_aleatoric (np.ndarray): Predicted aleatoric uncertainty per sample.
        true_errors (np.ndarray): True errors or error components per sample.

    Returns:
        float: UDA score (e.g., correlation or accuracy)
    """
    pred_uncert_epistemic = to_numpy(pred_uncert_epistemic)
    pred_uncert_aleatoric = to_numpy(pred_uncert_aleatoric)
    true_errors = to_numpy(true_errors)
    # For example, you might correlate epistemic uncertainty with error magnitude
    # This is a placeholder implementation — adjust according to your setup
    
    # Normalize uncertainties and errors
    epistemic_norm = (pred_uncert_epistemic - np.min(pred_uncert_epistemic)) / (np.ptp(pred_uncert_epistemic) + 1e-8)
    aleatoric_norm = (pred_uncert_aleatoric - np.min(pred_uncert_aleatoric)) / (np.ptp(pred_uncert_aleatoric) + 1e-8)
    errors_norm = (true_errors - np.min(true_errors)) / (np.ptp(true_errors) + 1e-8)
    
    epistemic_norm = epistemic_norm.flatten()
    aleatoric_norm = aleatoric_norm.flatten()
    errors_norm = errors_norm.flatten()

    # Compute correlation between epistemic uncertainty and error as proxy for UDA
    corr_epistemic = np.corrcoef(epistemic_norm, errors_norm)[0, 1]
    
    # Compute correlation between aleatoric uncertainty and error (or noise)
    corr_aleatoric = np.corrcoef(aleatoric_norm, errors_norm)[0, 1]
    
    # Average or weighted average of correlations as UDA
    uda_score = 0.5 * corr_epistemic + 0.5 * corr_aleatoric
    
    # Clamp score between 0 and 1 (optional)
    uda_score = max(0.0, min(1.0, uda_score))
    
    return uda_score


def ncg(confidence_scores, baseline_confidence_scores) -> float:
    """
    Compute Normalized Confidence Gain (NCG).

    Args:
        confidence_scores (np.ndarray): Confidence scores after calibration.
        baseline_confidence_scores (np.ndarray): Baseline confidence scores.

    Returns:
        float: NCG value between 0 and 1.
    """
    confidence_scores = to_numpy(confidence_scores)
    baseline_confidence_scores = to_numpy(baseline_confidence_scores)
    # For instance, use mean confidence improvement normalized by max possible gain
    gain = np.mean(confidence_scores) - np.mean(baseline_confidence_scores)
    
    # Normalize by max possible gain (e.g., 1 - baseline_mean_confidence)
    max_gain = 1.0 - np.mean(baseline_confidence_scores)
    
    if max_gain == 0:
        return 0.0
    
    ncg = gain / max_gain
    ncg = max(0.0, min(1.0, ncg))  # clamp between 0 and 1
    
    return ncg



def meta_metric__bnn_ednn(uda, meta_calibration_score, corr_err_epistemic, ncg) -> float:
    """
    Computes the meta-metric as a weighted average of four submetrics.
    Supports both scalar and batched inputs.
    """
    if isinstance(uda, torch.Tensor):
        uda = uda.mean().item()
    if isinstance(meta_calibration_score, torch.Tensor):
        meta_calibration_score = meta_calibration_score.mean().item()
    if isinstance(corr_err_epistemic, torch.Tensor):
        corr_err_epistemic = corr_err_epistemic.mean().item()
    if isinstance(ncg, torch.Tensor):
        ncg = ncg.mean().item()

    return (
        0.25 * uda +
        0.25 * meta_calibration_score +
        0.25 * corr_err_epistemic +
        0.25 * (1 - ncg)
    )


# def meta_metric__bnn_ednn(uda, meta_calibration_score, corr_err_epistemic, ncg) -> float:
#     """
#     Computes the meta-metric as a weighted average of four submetrics.

#     Args:
#         UDA (float): Uncertainty Decomposition Accuracy
#         meta_calibration_score (float): Calibration score (e.g. ECE, ACE)
#         corr_err_epistemic (float): Correlation between error and epistemic uncertainty
#         NCG (float): Normalized Confidence Gain

#     Returns:
#         float: Meta-metric value
#     """
#     return (
#         0.25 * uda +
#         0.25 * meta_calibration_score +
#         0.25 * corr_err_epistemic +
#         0.25 * (1 - ncg)
#     )

