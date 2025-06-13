# BA__Projekt/tests/test__ednn_classification__mnist.py

import unittest

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from BA__Programmierung.ml.losses.evidential_loss import evidential_loss
from models.model__ednn_deep import EvidentialNetDeep as EvidentialNet


class TestEvidentialRegressionMNIST(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = "cpu"

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        cls.dataset = datasets.MNIST(
            root="/root/BA__Projekt/assets/data/raw/",
            train=True,
            download=True,
            transform=transform
        )

        cls.loader = DataLoader(cls.dataset, batch_size=32, shuffle=False)
        cls.model = EvidentialNet(input_dim=28 * 28).to(cls.device)
        cls.model.eval()

    def test_forward_output_shapes(self):
        for X, y in self.loader:
            X, y = X.to(self.device), y.to(self.device).float().unsqueeze(1)
            X = X.view(X.size(0), -1)

            mu, v, alpha, beta = self.model(X)

            self.assertEqual(mu.shape, (X.shape[0], 1), "mu shape mismatch")
            self.assertEqual(v.shape, (X.shape[0], 1), "v shape mismatch")
            self.assertEqual(alpha.shape, (X.shape[0], 1), "alpha shape mismatch")
            self.assertEqual(beta.shape, (X.shape[0], 1), "beta shape mismatch")
            break

    def test_output_ranges(self):
        for X, _ in self.loader:
            X = X.to(self.device)
            X = X.view(X.size(0), -1)

            _, v, alpha, beta = self.model(X)

            self.assertTrue(torch.all(v > 0), "v should be positive")
            self.assertTrue(torch.all(alpha > 1), "alpha should be > 1")
            self.assertTrue(torch.all(beta > 0), "beta should be positive")
            break

    def test_loss_computation(self):
        for X, y in self.loader:
            X = X.to(self.device)
            y = y.to(self.device).float().unsqueeze(1)
            X = X.view(X.size(0), -1)

            mu, v, alpha, beta = self.model(X)
            loss = evidential_loss(y, mu, v, alpha, beta)
            self.assertIsInstance(loss.item(), float)
            self.assertGreater(loss.item(), 0.0, "Loss should be positive")
            break


if __name__ == "__main__":
    unittest.main()
