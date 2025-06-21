# BA__Projekt/tests/BA__Programmierung/ml/metrics/test__metrics.py
import unittest
import torch
from metrics import Metric


class TestMetric(unittest.TestCase):

    def setUp(self):
        """Set up a new Metric instance before each test."""
        # A mock metric function (mean squared error)
        def mock_mse(y_pred, y_true):
            return ((y_pred - y_true) ** 2).mean()

        # Initialize the Metric with the mock function
        self.metric = Metric(
            fn=mock_mse,
            name="mse",
            accumulate=True,
            arg_names=["y_pred", "y_true"]
        )

    def test_metric_initialization(self):
        """Test the initialization of the Metric class."""
        self.assertEqual(self.metric.name, "mse")
        self.assertTrue(callable(self.metric.fn))
        self.assertTrue(self.metric.accumulate)
        self.assertEqual(self.metric.arg_names, ["y_pred", "y_true"])

    def test_invalid_arg_names(self):
        """Test that a ValueError is raised when arg_names don't match function args."""
        with self.assertRaises(ValueError):
            Metric(fn=self.mock_mse, name="mse", accumulate=True, arg_names=["y_pred"])

    def test_accumulate_method(self):
        """Test if predictions are accumulated correctly."""
        y_pred = torch.tensor([1.0, 2.0])
        y_true = torch.tensor([1.0, 3.0])

        # Call accumulate to add the batch
        self.metric.__call__(y_pred=y_pred, y_true=y_true)

        # Ensure that predictions are stored
        self.assertEqual(len(self.metric.preds), 1)
        self.assertEqual(self.metric.preds[0], (y_pred.detach().cpu(), y_true.detach().cpu()))

    def test_compute_method(self):
        """Test the compute method to ensure it computes the correct result."""
        y_pred1 = torch.tensor([1.0, 2.0])
        y_true1 = torch.tensor([1.0, 3.0])

        y_pred2 = torch.tensor([1.5, 2.5])
        y_true2 = torch.tensor([1.5, 3.5])

        # Add multiple batches
        self.metric.__call__(y_pred=y_pred1, y_true=y_true1)
        self.metric.__call__(y_pred=y_pred2, y_true=y_true2)

        # Compute the result
        result = self.metric.compute()

        # Verify that the result is the correct MSE (mean of squared differences)
        expected_result = ((y_pred1 - y_true1) ** 2).mean().item() + ((y_pred2 - y_true2) ** 2).mean().item()
        expected_result /= 2  # Since we have 2 batches

        self.assertAlmostEqual(result, expected_result, places=4)

    def test_reset_method(self):
        """Test the reset method to clear stored predictions."""
        y_pred = torch.tensor([1.0, 2.0])
        y_true = torch.tensor([1.0, 3.0])

        # Accumulate predictions
        self.metric.__call__(y_pred=y_pred, y_true=y_true)

        # Ensure that predictions are accumulated
        self.assertEqual(len(self.metric.preds), 1)

        # Reset the metric
        self.metric.reset()

        # Verify that the predictions are cleared
        self.assertEqual(len(self.metric.preds), 0)

    def test_call_method_with_missing_args(self):
        """Test that the __call__ method handles missing arguments gracefully."""
        # Call with missing argument
        y_pred = torch.tensor([1.0, 2.0])
        result = self.metric.__call__(y_pred=y_pred)  # Missing y_true

        # Verify that the metric was skipped due to missing input
        self.assertIsNone(result)  # The metric should not accumulate or compute anything

    def test_non_accumulating_metric(self):
        """Test that metrics which do not accumulate data skip the accumulation."""
        def mock_metric_no_accum(y_pred, y_true):
            return (y_pred == y_true).float().mean()

        metric = Metric(fn=mock_metric_no_accum, name="accuracy", accumulate=False, arg_names=["y_pred", "y_true"])

        y_pred = torch.tensor([1.0, 2.0])
        y_true = torch.tensor([1.0, 2.0])

        # Call the non-accumulating metric
        metric.__call__(y_pred=y_pred, y_true=y_true)

        # Ensure no accumulation
        self.assertEqual(len(metric.preds), 0)

        # Compute the result
        result = metric.compute()

        # Verify the result (direct computation without accumulation)
        expected_result = (y_pred == y_true).float().mean().item()
        self.assertEqual(result, expected_result)


if __name__ == "__main__":
    unittest.main()

