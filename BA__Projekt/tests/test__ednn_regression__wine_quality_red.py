# BA__Projekt/tests/test__ednn_regression__wine_quality_red.py

import unittest
import torch

from BA__Programmierung.ml.datasets.dataset__torch__wine_quality_red import WineQualityDataset
from BA__Programmierung.ml.losses.evidential_loss import evidential_loss
from models.model__ednn_deep import EvidentialNetDeep as EvidentialNet
from torch.utils.data import DataLoader


class TestEvidentialRegressionWineQualityRed(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = "cpu"
        cls.dataset = WineQualityDataset(
            db_path="/root/BA__Projekt/assets/dbs/dataset__wine_quality_red__dataset.duckdb",
            table_name="wine_quality_red__dataset_csv"
        )
        cls.loader = DataLoader(cls.dataset, batch_size=32, shuffle=False)
        cls.model = EvidentialNet(input_dim=11).to(cls.device)  # Wine Quality Red typically has 11 features
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
