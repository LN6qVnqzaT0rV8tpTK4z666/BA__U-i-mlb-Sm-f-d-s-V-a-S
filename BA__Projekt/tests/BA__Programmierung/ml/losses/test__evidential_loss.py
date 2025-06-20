import unittest
import torch
from evidential_loss import evidential_loss


class TestEvidentialLoss(unittest.TestCase):

    def setUp(self):
        # Set up mock data for testing
        torch.manual_seed(42)
        self.N = 16
        self.y = torch.randn(self.N)
        self.mu = self.y + 0.1 * torch.randn(self.N)
        self.v = torch.abs(torch.randn(self.N)) + 1e-6
        self.alpha = torch.abs(torch.randn(self.N)) + 1.0
        self.beta = torch.abs(torch.randn(self.N)) + 1.0

    def test_nll_loss(self):
        # Test NLL loss
        loss = evidential_loss(self.y, self.mu, self.v, self.alpha, self.beta, mode="nll")
        self.assertIsInstance(loss, torch.Tensor)
        self.assertFalse(loss.isnan().any(), "Loss contains NaN values")

    def test_abs_loss(self):
        # Test Abs loss
        loss = evidential_loss(self.y, self.mu, self.v, self.alpha, self.beta, mode="abs")
        self.assertIsInstance(loss, torch.Tensor)
        self.assertFalse(loss.isnan().any(), "Loss contains NaN values")

    def test_mse_loss(self):
        # Test MSE loss
        loss = evidential_loss(self.y, self.mu, self.v, self.alpha, self.beta, mode="mse")
        self.assertIsInstance(loss, torch.Tensor)
        self.assertFalse(loss.isnan().any(), "Loss contains NaN values")

    def test_kl_loss(self):
        # Test KL divergence loss
        loss = evidential_loss(self.y, self.mu, self.v, self.alpha, self.beta, mode="kl")
        self.assertIsInstance(loss, torch.Tensor)
        self.assertFalse(loss.isnan().any(), "Loss contains NaN values")

    def test_scaled_loss(self):
        # Test Scaled loss
        loss = evidential_loss(self.y, self.mu, self.v, self.alpha, self.beta, mode="scaled")
        self.assertIsInstance(loss, torch.Tensor)
        self.assertFalse(loss.isnan().any(), "Loss contains NaN values")

    def test_variational_loss(self):
        # Test Variational loss
        loss = evidential_loss(self.y, self.mu, self.v, self.alpha, self.beta, mode="variational")
        self.assertIsInstance(loss, torch.Tensor)
        self.assertFalse(loss.isnan().any(), "Loss contains NaN values")

    def test_full_loss(self):
        # Test Full loss
        loss = evidential_loss(self.y, self.mu, self.v, self.alpha, self.beta, mode="full")
        self.assertIsInstance(loss, torch.Tensor)
        self.assertFalse(loss.isnan().any(), "Loss contains NaN values")

    def test_invalid_mode(self):
        # Test invalid mode
        with self.assertRaises(ValueError):
            evidential_loss(self.y, self.mu, self.v, self.alpha, self.beta, mode="invalid_mode")

    def test_nan_input(self):
        # Test for NaN input
        self.y[0] = float("nan")
        loss = evidential_loss(self.y, self.mu, self.v, self.alpha, self.beta, mode="nll")
        self.assertTrue(torch.isnan(loss), "Loss should be NaN for NaN input")

    def test_sanitization(self):
        # Test input sanitization (clamping)
        v_original = self.v.clone()
        self.v[0] = -1.0  # Set to an invalid negative value
        loss = evidential_loss(self.y, self.mu, self.v, self.alpha, self.beta, mode="nll")
        self.assertTrue(torch.all(self.v >= 1e-6), "v should be clamped to >= 1e-6")
        self.assertFalse(loss.isnan().any(), "Loss contains NaN values after sanitization")

    def test_nan_output(self):
        # Test for NaN output when inputs contain NaNs
        self.mu[0] = float("nan")
        loss = evidential_loss(self.y, self.mu, self.v, self.alpha, self.beta, mode="nll")
        self.assertTrue(torch.isnan(loss), "Loss should be NaN when inputs contain NaN values")


if __name__ == "__main__":
    unittest.main()
