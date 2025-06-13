# BA__Projekt/tests/test__ednn_regression__iris.py

import numpy as np
import torch
import unittest

from BA__Programmierung.ml.datasets.dataset__torch__duckdb_iris import DatasetTorchDuckDBIris
from BA__Programmierung.ml.losses.evidential_loss import evidential_loss
from models.model__ednn_deep import EvidentialNetDeep as EvidentialNet
from torch.utils.data import DataLoader


class TestEvidentialRegressionIris(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = "cpu"
        cls.dataset = DatasetTorchDuckDBIris(
            db_path="/root/BA__U-i-mlb-Sm-f-d-s-V-a-S/BA__Projekt/assets/dbs/dataset__iris__dataset.duckdb",
            table_name="iris__dataset_csv"
        )
        cls.loader = DataLoader(cls.dataset, batch_size=32, shuffle=False)
        cls.model = EvidentialNet(input_dim=4).to(cls.device)
        cls.model.eval()

    def test_forward_output_shapes(self):
        for X, _ in self.loader:
            X = X.to(self.device)
            mu, v, alpha, beta = self.model(X)

            self.assertEqual(mu.shape, (X.shape[0], 1), "mu shape mismatch")
            self.assertEqual(v.shape, (X.shape[0], 1), "v shape mismatch")
            self.assertEqual(alpha.shape, (X.shape[0], 1), "alpha shape mismatch")
            self.assertEqual(beta.shape, (X.shape[0], 1), "beta shape mismatch")
            break  # Only test one batch

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
