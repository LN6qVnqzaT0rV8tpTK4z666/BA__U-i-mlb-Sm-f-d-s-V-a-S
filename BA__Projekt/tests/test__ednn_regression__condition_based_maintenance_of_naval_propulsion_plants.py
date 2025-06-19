# BA__Projekt/tests/test__ednn_regression__condition_based_maintenance_of_naval_propulsion_plants.py

import unittest
import torch
from torch.utils.data import DataLoader

from BA__Programmierung.ml.datasets.dataset__torch__condition_based_maintenance_of_naval_propulsion_plants import (
    NavalPropulsionDataset,
)
from BA__Programmierung.ml.losses.evidential_loss import evidential_loss
from models.model__generic_ensemble import GenericEnsembleRegressor


class TestGenericEnsembleEvidentialRegressionNavalPropulsion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # === Dataset & Dataloader ===
        cls.dataset = NavalPropulsionDataset(
            csv_path="/root/BA__U-i-mlb-Sm-f-d-s-V-a-S/BA__Projekt/assets/data/raw/dataset__condition-based-maintenance-of-naval-propulsion-plants/data.csv"
        )
        cls.loader = DataLoader(cls.dataset, batch_size=32, shuffle=False)

        # === Model Config ===
        cls.input_dim = 16  # Naval dataset has 16 input features
        cls.model_config = {
            "input_dim": cls.input_dim,
            "hidden_dims": [64, 64],
            "output_type": "evidential",
            "use_dropout": False,
            "dropout_p": 0.1,
            "flatten_input": False,
            "use_batchnorm": False,
            "activation_name": "relu",
        }

        # === Generic Ensemble ===
        cls.ensemble_size = 3
        cls.model = GenericEnsembleRegressor(
            base_config=cls.model_config,
            n_models=cls.ensemble_size
        ).to(cls.device)
        cls.model.eval()

    def test_forward_output_shapes(self):
        X, _ = next(iter(self.loader))
        X = X.to(self.device)

        mu, v, alpha, beta = self.model(X)

        self.assertEqual(mu.shape, (X.shape[0], 1), "mu shape mismatch")
        self.assertEqual(v.shape, (X.shape[0], 1), "v shape mismatch")
        self.assertEqual(alpha.shape, (X.shape[0], 1), "alpha shape mismatch")
        self.assertEqual(beta.shape, (X.shape[0], 1), "beta shape mismatch")

    def test_output_ranges(self):
        X, _ = next(iter(self.loader))
        X = X.to(self.device)
        _, v, alpha, beta = self.model(X)

        self.assertTrue(torch.all(v > 0), "v should be positive")
        self.assertTrue(torch.all(alpha > 1), "alpha should be > 1")
        self.assertTrue(torch.all(beta > 0), "beta should be positive")

    def test_loss_computation(self):
        X, y = next(iter(self.loader))
        X, y = X.to(self.device), y.to(self.device)

        mu, v, alpha, beta = self.model(X)
        loss = evidential_loss(y, mu, v, alpha, beta)

        self.assertIsInstance(loss.item(), float)
        self.assertGreater(loss.item(), 0.0, "Loss should be positive")


if __name__ == "__main__":
    unittest.main()

