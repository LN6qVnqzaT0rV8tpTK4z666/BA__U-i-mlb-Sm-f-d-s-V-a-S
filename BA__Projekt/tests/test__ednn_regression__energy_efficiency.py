# BA__Projekt/tests/test__ednn_regression__energy_efficiency.py
import unittest
import os
import shutil
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

from BA__Programmierung.ml.datasets.dataset__torch__energy_efficiency import EnergyEfficiencyDataset
from models.model__generic_ensemble import GenericEnsembleRegressor
from BA__Programmierung.viz.viz__ednn_regression__energy_efficiency import evaluate_and_visualize_and_save


class TestEvidentialRegressionEnergyEfficiencyEnsemble(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load dataset and prepare dataloader
        dataset_path = "assets/data/raw/dataset__energy-efficiency/dataset__energy-efficiency.csv"
        full_dataset = EnergyEfficiencyDataset(dataset_path)
        train_size = int(0.8 * len(full_dataset))
        test_size = len(full_dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42)
        )

        def extract_xy(dataset):
            X = torch.stack([item[0] for item in dataset])
            Y = torch.stack([item[1] for item in dataset])
            return X.numpy(), Y.numpy()

        X_train, Y_train = extract_xy(train_dataset)
        X_test, Y_test = extract_xy(test_dataset)

        cls.scaler_x = StandardScaler().fit(X_train)
        cls.scaler_y = StandardScaler().fit(Y_train)

        X_test_scaled = torch.tensor(cls.scaler_x.transform(X_test), dtype=torch.float32)
        Y_test_scaled = torch.tensor(cls.scaler_y.transform(Y_test), dtype=torch.float32)

        cls.test_loader = DataLoader(torch.utils.data.TensorDataset(X_test_scaled, Y_test_scaled), batch_size=64, shuffle=False)
        cls.device = torch.device("cpu")
        input_dim = X_test_scaled.shape[1]

        base_config = {
            "input_dim": input_dim,
            "hidden_dims": [64, 64],
            "output_type": "evidential",
            "use_dropout": False,
            "dropout_p": 0.2,
            "flatten_input": False,
            "use_batchnorm": False,
            "activation_name": "relu",
        }

        cls.model = GenericEnsembleRegressor(base_config=base_config, n_models=5).to(cls.device)

        model_path = "assets/models/pth/ednn_regression__energy_efficiency_ensemble/ednn_regression__energy_efficiency_ensemble.pth"
        cls.model.load_state_dict(torch.load(model_path, map_location=cls.device))
        cls.model.eval()

        # Directory for saving plots during tests
        cls.test_save_dir = "test_viz_output"
        # Clean if exists from previous runs
        if os.path.exists(cls.test_save_dir):
            shutil.rmtree(cls.test_save_dir)

    def test_model_forward_and_shapes(self):
        # Test model output shapes from the ensemble
        for x, y in self.test_loader:
            x = x.to(self.device)
            outputs = self.model(x)
            # output is a tuple or list, where output[0] is predictions from first ensemble model
            self.assertTrue(len(outputs) > 0)
            preds = outputs[0]
            self.assertEqual(preds.shape[0], x.shape[0])
            break

    def test_evaluate_and_save_plots(self):
        # This calls the updated evaluation function that saves plots
        evaluate_and_visualize_and_save(
            self.model,
            self.test_loader,
            self.scaler_y,
            self.device,
            self.test_save_dir,
        )

        # Check that plot files exist
        expected_files = [
            "true_vs_pred.png",
            "residual_hist.png",
            "residuals_vs_true.png",
        ]
        for filename in expected_files:
            filepath = os.path.join(self.test_save_dir, filename)
            self.assertTrue(os.path.isfile(filepath), f"Plot file {filename} was not saved.")

    @classmethod
    def tearDownClass(cls):
        # Optionally clean up the saved plots directory after tests
        if os.path.exists(cls.test_save_dir):
            shutil.rmtree(cls.test_save_dir)


if __name__ == "__main__":
    unittest.main()
