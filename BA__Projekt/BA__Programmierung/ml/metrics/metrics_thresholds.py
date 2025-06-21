# BA__Projekt/BA__Programmierung/ml/metrics/metrics_treshholds.py

"""
This module defines the thresholds for various metrics used in the evaluation of machine learning models.

Each metric is assigned a set of thresholds, where:
- "positive": the threshold below which the metric is considered positive (lower is better for metrics like MSE, MAE).
- "negative": the threshold above which the metric is considered negative (higher is better for metrics like R²).

The thresholds can be used in the evaluation process to classify the performance of models based on the computed metric values.

Metric thresholds are defined for regression, uncertainty quantification, and other evaluation metrics.

Attributes
----------
metric_thresholds : dict
    A dictionary mapping metric names to their respective threshold values for classification ("positive" and "negative").
"""

# Schwellenwerte für die Metriken
metric_thresholds = {
    # Regression Metriken
    "mse": {"positive": 0.5, "negative": 2.0},  # Niedriger ist besser
    "rmse": {"positive": 0.7, "negative": 2.5},  # Niedriger ist besser
    "mae": {"positive": 0.5, "negative": 1.5},  # Niedriger ist besser
    "mape": {"positive": 10, "negative": 30},  # Niedriger ist besser
    "r2_score": {"positive": 0.8, "negative": 0.2},  # Hoch ist besser

    # Unsicherheitsmetriken (UQ)
    "ece": {"positive": 0.1, "negative": 0.2},  # Niedrig ist besser
    "ace": {"positive": 0.1, "negative": 0.2},  # Niedrig ist besser
    "elbo": {"positive": 1.0, "negative": 0.5},  # Hoch ist besser
    "nll_gaussian": {"positive": 1.0, "negative": 2.0},  # Niedrig ist besser
    "mean_pred_variance": {"positive": 0.1, "negative": 1.0},  # Niedrig ist besser
    "energy_score": {"positive": 1.0, "negative": 2.0},  # Niedrig ist besser
    "kl_divergence_normal": {"positive": 0.1, "negative": 0.5},  # Niedrig ist besser
    "mutual_information": {"positive": 0.5, "negative": 0.1},  # Hoch ist besser
    "calibration_error": {"positive": 0.1, "negative": 0.2},  # Niedrig ist besser
    "uda": {"positive": 0.5, "negative": 0.2},  # Hoch ist besser
    "ncg": {"positive": 0.9, "negative": 0.7},  # Hoch ist besser
    "meta_metric__bnn_ednn": {"positive": 0.5, "negative": 0.2},  # Hoch ist besser

    # Statistische Metriken
    "statistical__mod_pred__mean": {"positive": -5.0, "negative": 5.0},  # Werte um Null oder gering sind besser (modelliert den Mittelwert)
    "statistical__mod_pred__variance_band": {"positive": 0.05, "negative": 1.0},  # Niedrig ist besser
    "statistical__mod_pred__standard_error": {"positive": 0.01, "negative": 0.5},  # Niedrig ist besser
    "statistical__mod_pred__quantiles": {"positive": 0.1, "negative": 1.0},  # Niedrig ist besser
    "statistical__mod_pred__predictive_mean": {"positive": -5.0, "negative": 5.0},  # Werte um Null oder gering sind besser
    "statistical__mod_pred__plus_minus_sigma": {"positive": 0.05, "negative": 1.0},  # Niedrig ist besser
    "statistical__mod_pred__plus_minus_1_sigma": {"positive": 0.05, "negative": 1.0},  # Niedrig ist besser
    "statistical__mod_pred__plus_minus_2_sigma": {"positive": 0.1, "negative": 1.0},  # Niedrig ist besser
    "statistical__mod_pred__plus_minus_3_sigma": {"positive": 0.2, "negative": 2.0},  # Niedrig ist besser
    "statistical__mod_pred__plus_minus_4_sigma": {"positive": 0.3, "negative": 3.0},  # Niedrig ist besser
    "statistical__mod_pred__plus_minus_5_sigma": {"positive": 0.5, "negative": 5.0}   # Niedrig ist besser
}
