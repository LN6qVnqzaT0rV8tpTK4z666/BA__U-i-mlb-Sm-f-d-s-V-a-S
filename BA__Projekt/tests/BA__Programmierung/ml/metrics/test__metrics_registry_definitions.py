import unittest
import torch
import numpy as np
from BA__Programmierung.ml.metrics.metrics_registry_definitions import (
    accuracy, top_k_accuracy, mse, rmse, mae, mape, r2_score,
    nll_gaussian, energy_score, ece, regression_ece, ace, brier_score
)


class TestMetricsRegistryDefinitions(unittest.TestCase):

    def test_accuracy(self):
        y_pred = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
        y_true = torch.tensor([1, 0])
        result = accuracy(y_pred, y_true)
        self.assertEqual(result, 1.0)  # Correct prediction for both samples

    def test_top_k_accuracy(self):
        y_pred = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
        y_true = torch.tensor([1, 0])
        result = top_k_accuracy(y_pred, y_true, k=1)
        self.assertEqual(result, 1.0)  # Correct prediction for both samples

    def test_mse(self):
        y_true = torch.tensor([3.0, 5.0])
        y_pred = torch.tensor([2.5, 5.5])
        result = mse(y_true, y_pred)
        self.assertAlmostEqual(result, 0.25, places=2)

    def test_rmse(self):
        y_true = torch.tensor([3.0, 5.0])
        y_pred = torch.tensor([2.5, 5.5])
        result = rmse(y_pred, y_true)
        self.assertAlmostEqual(result, 0.5, places=2)

    def test_mae(self):
        y_true = torch.tensor([3.0, 5.0])
        y_pred = torch.tensor([2.5, 5.5])
        result = mae(y_pred, y_true)
        self.assertEqual(result, 0.5)

    def test_mape(self):
        y_true = torch.tensor([100.0, 200.0])
        y_pred = torch.tensor([90.0, 210.0])
        result = mape(y_pred, y_true)
        self.assertAlmostEqual(result, 5.0, places=1)  # 5% error

    def test_r2_score(self):
        y_true = torch.tensor([3.0, 5.0])
        y_pred = torch.tensor([2.5, 5.5])
        result = r2_score(y_pred, y_true)
        self.assertAlmostEqual(result, 0.99, places=2)

    def test_nll_gaussian(self):
        mean = torch.tensor([0.0, 1.0])
        logvar = torch.tensor([0.1, 0.2])
        target = torch.tensor([0.0, 1.0])
        result = nll_gaussian(mean, logvar, target)
        self.assertGreater(result, 0)

    def test_energy_score(self):
        y_samples = torch.tensor([[[0.1], [0.2]], [[0.3], [0.4]]])  # (B=2, S=2, D=1)
        y_true = torch.tensor([[0.15], [0.35]])  # (B=2, D=1)
        result = energy_score(y_samples, y_true)
        self.assertGreater(result, 0)

    def test_ece(self):
        y_pred = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
        y_true = torch.tensor([1, 0])
        result = ece(y_pred, y_true, n_bins=2)
        self.assertGreaterEqual(result, 0)

    def test_regression_ece(self):
        y_pred_mean = torch.tensor([2.0, 3.0])
        y_pred_std = torch.tensor([0.1, 0.2])
        y_true = torch.tensor([2.1, 3.0])
        result = regression_ece(y_pred_mean, y_pred_std, y_true, n_bins=2)
        self.assertGreaterEqual(result, 0)

    def test_ace(self):
        y_pred = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
        y_true = torch.tensor([1, 0])
        result = ace(y_pred, y_true, n_bins=2)
        self.assertGreaterEqual(result, 0)

    def test_brier_score(self):
        y_pred = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
        y_true = torch.tensor([1, 0])
        result = brier_score(y_pred, y_true)
        self.assertGreaterEqual(result, 0)


if __name__ == "__main__":
    unittest.main()
