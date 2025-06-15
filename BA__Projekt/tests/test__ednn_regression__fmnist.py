# BA__Projekt/tests/test__ednn_regression__fmnist.py

import unittest
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from BA__Programmierung.ml.losses.evidential_loss import evidential_loss
from models.model__generic_ensemble import GenericEnsembleRegressor


class TestGenericEnsembleRegressionFMNIST(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        cls.dataset = datasets.FashionMNIST(
            root="/root/BA__Projekt/assets/data/raw/",
            train=True,
            download=True,
            transform=transform
        )

        cls.loader = DataLoader(cls.dataset, batch_size=32, shuffle=False)

        base_config = {
            "input_dim": 28 * 28,
            "hidden_dims": [128, 64],
            "output_type": "evidential",
            "use_dropout": False,
            "dropout_p": 0.1,
            "flatten_input": True,
            "use_batchnorm": False,
            "activation_name": "relu",
        }

        cls.model = GenericEnsembleRegressor(base_config=base_config, n_models=3).to(cls.device)
        cls.model.eval()

    def test_forward_output_shapes(self):
        for X, y in self.loader:
            X = X.to(self.device)
            y = y.to(self.device).float().unsqueeze(1)

            mu, v, alpha, beta = self.model(X)

            self.assertEqual(mu.shape, (3, X.shape[0], 1), "mu shape mismatch")
            self.assertEqual(v.shape, (3, X.shape[0], 1), "v shape mismatch")
            self.assertEqual(alpha.shape, (3, X.shape[0], 1), "alpha shape mismatch")
            self.assertEqual(beta.shape, (3, X.shape[0], 1), "beta shape mismatch")
            break

    def test_output_ranges(self):
        for X, _ in self.loader:
            X = X.to(self.device)

            _, v, alpha, beta = self.model(X)

            # Average across ensemble members
            v_mean = v.mean(dim=0)
            alpha_mean = alpha.mean(dim=0)
            beta_mean = beta.mean(dim=0)

            self.assertTrue(torch.all(v_mean > 0), "v should be positive")
            self.assertTrue(torch.all(alpha_mean > 1), "alpha should be > 1")
            self.assertTrue(torch.all(beta_mean > 0), "beta should be positive")
            break

    def test_loss_computation(self):
        for X, y in self.loader:
            X = X.to(self.device)
            y = y.to(self.device).float().unsqueeze(1)

            mu, v, alpha, beta = self.model(X)

            # Average ensemble predictions
            mu_mean = mu.mean(dim=0)
            v_mean = v.mean(dim=0)
            alpha_mean = alpha.mean(dim=0)
            beta_mean = beta.mean(dim=0)

            loss = evidential_loss(y, mu_mean, v_mean, alpha_mean, beta_mean)
            self.assertIsInstance(loss.item(), float)
            self.assertGreater(loss.item(), 0.0, "Loss should be positive")
            break


if __name__ == "__main__":
    unittest.main()
