# BA__Projekt/tests/BA__Programmierung/ml/utils/test__training_utils.py
import unittest
from unittest.mock import MagicMock, patch
import torch
from training_utils import train_one_epoch, evaluate, train_with_early_stopping


class TestTrainingUtils(unittest.TestCase):

    def setUp(self):
        """Set up mock model, dataloader, optimizer, and device for testing"""
        # Create a mock model
        self.model = MagicMock()
        self.model.return_value = (torch.tensor([0.5]), torch.tensor([0.1]), torch.tensor([2.0]), torch.tensor([0.3]))

        # Mock dataloader
        self.dataloader = MagicMock()
        self.dataloader.__iter__.return_value = iter([(torch.tensor([[1.0]]), torch.tensor([[1.0]]))])  # Single batch

        # Mock optimizer
        self.optimizer = MagicMock()
        self.optimizer.zero_grad = MagicMock()
        self.optimizer.step = MagicMock()

        # Mock device
        self.device = torch.device('cpu')

    def test_train_one_epoch(self):
        """Test the train_one_epoch function"""
        # Call the training function
        loss = train_one_epoch(self.model, self.dataloader, self.optimizer, self.device, loss_mode="nll")

        # Check that the model was called and optimizer was used
        self.optimizer.zero_grad.assert_called_once()
        self.optimizer.step.assert_called_once()
        self.assertEqual(len(self.dataloader), 1)

        # Ensure the loss returned is a float
        self.assertIsInstance(loss, float)

    def test_evaluate(self):
        """Test the evaluate function"""
        # Mock the metrics_registry to avoid actual metric computation
        with patch("BA__Programmierung.ml.metrics.metrics_registry.MetricsRegistry.report") as mock_report:
            loss = evaluate(self.model, self.dataloader, self.device, loss_mode="nll", metrics_token="regression")

            # Ensure the evaluate function runs through the entire loop and computes loss
            self.assertEqual(loss, 0.5)  # As expected loss from mock return value
            mock_report.assert_called_once_with("regression")

    @patch("torch.save")  # Mock torch.save to avoid actual saving
    def test_train_with_early_stopping(self):
        """Test the train_with_early_stopping function"""
        # Mock validation loader and other components
        val_loader = MagicMock()
        val_loader.__iter__.return_value = iter([(torch.tensor([[1.0]]), torch.tensor([[1.0]]))])  # Single batch

        model_path = "mock_model_path.pth"

        # Call train_with_early_stopping with early stopping set to patience=1 for quick testing
        train_with_early_stopping(
            model=self.model,
            train_loader=self.dataloader,
            val_loader=val_loader,
            optimizer=self.optimizer,
            model_path=model_path,
            device=self.device,
            epochs=2,  # Run for only 2 epochs
            patience=1,  # Early stopping patience
            loss_mode="nll",
            metrics_token="regression"
        )

        # Ensure the model was saved
        torch.save.assert_called_once_with(self.model.state_dict(), model_path)

    @patch("torch.save")  # Mock torch.save to avoid actual saving
    def test_early_stopping(self):
        """Test early stopping with no improvement after certain epochs"""
        # Mock validation loader to simulate the loss not improving
        val_loader = MagicMock()
        val_loader.__iter__.return_value = iter([(torch.tensor([[1.0]]), torch.tensor([[1.0]]))])  # Single batch

        model_path = "mock_model_path.pth"

        # Call train_with_early_stopping with patience=1 and simulate no improvement
        with patch("training_utils.evaluate", return_value=0.5):  # Simulate constant validation loss
            train_with_early_stopping(
                model=self.model,
                train_loader=self.dataloader,
                val_loader=val_loader,
                optimizer=self.optimizer,
                model_path=model_path,
                device=self.device,
                epochs=5,  # Run for 5 epochs
                patience=1,  # Early stopping patience
                loss_mode="nll",
                metrics_token="regression"
            )

        # Ensure early stopping happened after 2 epochs (due to patience=1)
        torch.save.assert_called_once_with(self.model.state_dict(), model_path)

    def test_invalid_loss_mode_in_train(self):
        """Test the train_one_epoch function with an invalid loss mode"""
        with self.assertRaises(ValueError):
            train_one_epoch(self.model, self.dataloader, self.optimizer, self.device, loss_mode="invalid_loss")

    def test_invalid_loss_mode_in_evaluate(self):
        """Test the evaluate function with an invalid loss mode"""
        with self.assertRaises(ValueError):
            evaluate(self.model, self.dataloader, self.device, loss_mode="invalid_loss", metrics_token="regression")


if __name__ == "__main__":
    unittest.main()

