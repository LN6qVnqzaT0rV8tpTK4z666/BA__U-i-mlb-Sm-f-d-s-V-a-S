# BA__Projekt/tests/test__ednn_regression__boston_housing.py
"""
Visualize predictions of a trained Evidential Deep Neural Network on the Boston Housing dataset.

This script:
- Loads the trained model
- Applies it to the entire dataset
- Visualizes true vs predicted values with a scatter plot
"""
import os
import unittest
import torch
from torch.utils.data import DataLoader

from BA__Programmierung.ml.datasets.dataset__torch__boston_housing import DatasetTorchBostonHousing
from BA__Programmierung.ml.losses.evidential_loss import evidential_loss
from models.model__generic_ensemble import GenericEnsembleRegressor


class TestEvidentialEnsembleRegressionBostonHousing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Dataset
        cls.dataset = DatasetTorchBostonHousing(
            csv_path="assets/data/raw/dataset__boston-housing/dataset__boston-housing.csv"
        )
        cls.loader = DataLoader(cls.dataset, batch_size=32, shuffle=False)

        # Ensemble setup
        model_dir = "/root/BA__U-i-mlb-Sm-f-d-s-V-a-S/BA__Projekt/assets/models/pth/ednn_regression__boston_housing"
        base_config = {"input_dim": 13, "hidden_dims": [64, 64], "output_type": "evidential"}

        model_files = sorted([f for f in os.listdir(model_dir) if f.startswith("model_") and f.endswith(".pth")])
        cls.ensemble = GenericEnsembleRegressor(base_config, n_models=len(model_files))

        for i, f in enumerate(model_files):
            path = os.path.join(model_dir, f)
            cls.ensemble.models[i].load_state_dict(torch.load(path, map_location=cls.device))
            cls.ensemble.models[i].to(cls.device)
            cls.ensemble.models[i].eval()

    def test_output_shapes_and_ranges(self):
        for X, _ in self.loader:
            X = X.to(self.device)
            mu, v, alpha, beta = self.ensemble(X)

            self.assertEqual(mu.shape, (X.shape[0], 1))  # Instead of (n_models, X.shape[0], 1)
            self.assertEqual(v.shape, mu.shape)
            self.assertEqual(alpha.shape, mu.shape)
            self.assertEqual(beta.shape, mu.shape)

            self.assertTrue(torch.all(v > 0), "v should be > 0")
            self.assertTrue(torch.all(alpha > 1), "alpha should be > 1")
            self.assertTrue(torch.all(beta > 0), "beta should be > 0")
            break

    def test_loss_computation_single_model(self):
        for X, y in self.loader:
            X, y = X.to(self.device), y.to(self.device)
            model = self.ensemble.models[0]
            mu, v, alpha, beta = model(X)
            loss = evidential_loss(y, mu, v, alpha, beta)
            self.assertIsInstance(loss.item(), float)
            self.assertGreater(loss.item(), 0.0, "Loss should be positive")
            break


if __name__ == "__main__":
    unittest.main()
