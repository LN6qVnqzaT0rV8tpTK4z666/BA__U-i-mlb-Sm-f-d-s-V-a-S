# BA__Projekt/tests/test__ednn_regression__nmavani_func1.py

import unittest

import torch
from torch.utils.data import DataLoader

from BA__Programmierung.config import (
    CSV_PATH__GENERATED__MAVANI__FUNC_1,
    DB_PATH__GENERATED__MAVANI__FUNC_1,
)
from BA__Programmierung.ml.datasets.dataset__torch__nmavani_func1 import (
    DatasetTorchDuckDBFunc1,
)
from BA__Programmierung.ml.losses.evidential_loss import evidential_loss
from models.model__ednn_deep import EvidentialNetDeep as EvidentialNet


class TestEvidentialRegressionNmavaniFunc1(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = "cpu"
        cls.dataset = DatasetTorchDuckDBFunc1(
            db_path=DB_PATH__GENERATED__MAVANI__FUNC_1,
            table_name=CSV_PATH__GENERATED__MAVANI__FUNC_1
        )
        cls.loader = DataLoader(cls.dataset, batch_size=32, shuffle=False)
        cls.model = EvidentialNet(input_dim=1).to(cls.device)  # Assuming single input feature
        cls.model.eval()

    def test_forward_output_shapes(self):
        for X, _ in self.loader:
            X = X.to(self.device)
            mu, v, alpha, beta = self.model(X)

            self.assertEqual(mu.shape, (X.shape[0], 1), "mu shape mismatch")
            self.assertEqual(v.shape, (X.shape[0], 1), "v shape mismatch")
            self.assertEqual(alpha.shape, (X.shape[0], 1), "alpha shape mismatch")
            self.assertEqual(beta.shape, (X.shape[0], 1), "beta shape mismatch")
            break

    def test_output_ranges(self):
        for X, _ in self.loader:
            X = X.to(self.device)
            _, v, alpha, beta = self.model(X)

            self.assertTrue(torch.all(v > 0), "v should be positive")
            self.assertTrue(torch.all(alpha > 1), "alpha should be > 1")
            self.assertTrue(torch.all(beta > 0), "beta should be positive")
            break

    def test_loss_computation(self):
        for X, y in self.loader:
            X, y = X.to(self.device), y.to(self.device)
            mu, v, alpha, beta = self.model(X)
            loss = evidential_loss(y, mu, v, alpha, beta)
            self.assertIsInstance(loss.item(), float)
            self.assertGreater(loss.item(), 0.0, "Loss should be positive")
            break


if __name__ == "__main__":
    unittest.main()
