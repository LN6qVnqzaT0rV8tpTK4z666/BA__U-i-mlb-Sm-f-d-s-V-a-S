# BA__Projekt/tests/BA__Programmierung/ml/metrics/test__metrics_registry.py
import unittest
import torch
from metrics_registry import MetricsRegistry


class TestMetricsRegistry(unittest.TestCase):

    def setUp(self):
        """Set up a new instance of MetricsRegistry for each test."""
        self.registry = MetricsRegistry()

    def test_register_metric(self):
        """Test if we can register a metric correctly."""
        # Define a mock metric function
        def mock_metric(y_pred, y_true):
            return (y_pred == y_true).float().mean()

        # Register the metric
        self.registry.register(token="classification", fn=mock_metric, name="mock_accuracy", accumulate=True, arg_names=["y_pred", "y_true"])

        # Retrieve the registered metric
        metric = self.registry.get("classification", "mock_accuracy")["mock_accuracy"]
        
        # Verify the metric is registered correctly
        self.assertEqual(metric.name, "mock_accuracy")
        self.assertTrue(callable(metric.fn))

    def test_get_registered_metrics(self):
        """Test if we can retrieve registered metrics correctly."""
        # Define a mock metric function
        def mock_metric(y_pred, y_true):
            return (y_pred == y_true).float().mean()

        # Register the metric
        self.registry.register(token="classification", fn=mock_metric, name="mock_accuracy", accumulate=True, arg_names=["y_pred", "y_true"])

        # Retrieve the metric
        metrics = self.registry.get("classification", "mock_accuracy")

        # Verify retrieval
        self.assertTrue("mock_accuracy" in metrics)
        self.assertEqual(metrics["mock_accuracy"].name, "mock_accuracy")

    def test_add_batch(self):
        """Test adding batches of data to the registry."""
        y_pred = torch.tensor([1, 2, 3])
        y_true = torch.tensor([1, 2, 4])
        
        # Register the metric
        self.registry.register(token="classification", fn=lambda y_pred, y_true: (y_pred == y_true).float().mean(),
                               name="accuracy", accumulate=True, arg_names=["y_pred", "y_true"])

        # Add a batch to the registry
        self.registry.add_batch(token="classification", y_pred=y_pred, y_true=y_true)

        # Check if the batch was added correctly
        self.assertEqual(len(self.registry.storage["classification"]["y_pred"]), 1)
        self.assertEqual(len(self.registry.storage["classification"]["y_true"]), 1)

    def test_report_metrics(self):
        """Test if metrics can be computed and reported correctly."""
        # Define a mock metric function
        def mock_metric(y_pred, y_true):
            return (y_pred == y_true).float().mean()

        # Register the metric
        self.registry.register(token="classification", fn=mock_metric, name="mock_accuracy", accumulate=True, arg_names=["y_pred", "y_true"])

        # Add some batches
        y_pred1 = torch.tensor([1, 2, 3])
        y_true1 = torch.tensor([1, 2, 3])
        y_pred2 = torch.tensor([1, 2, 3])
        y_true2 = torch.tensor([1, 2, 4])
        
        self.registry.add_batch(token="classification", y_pred=y_pred1, y_true=y_true1)
        self.registry.add_batch(token="classification", y_pred=y_pred2, y_true=y_true2)

        # Generate the report
        report = self.registry.report("classification")

        # Verify that the metric is computed correctly
        self.assertIn("mock_accuracy", report)
        self.assertAlmostEqual(report["mock_accuracy"], 0.9167, places=4)  # Expected average accuracy

    def test_accumulation_of_metrics(self):
        """Test if accumulation across multiple batches works."""
        # Define a mock metric function
        def mock_metric(y_pred, y_true):
            return (y_pred == y_true).float().mean()

        # Register the metric
        self.registry.register(token="classification", fn=mock_metric, name="mock_accuracy", accumulate=True, arg_names=["y_pred", "y_true"])

        # Add multiple batches
        y_pred1 = torch.tensor([1, 2, 3])
        y_true1 = torch.tensor([1, 2, 3])
        y_pred2 = torch.tensor([1, 2, 3])
        y_true2 = torch.tensor([1, 2, 4])

        self.registry.add_batch(token="classification", y_pred=y_pred1, y_true=y_true1)
        self.registry.add_batch(token="classification", y_pred=y_pred2, y_true=y_true2)

        # Compute the accumulated result
        report = self.registry.report("classification")
        
        # Verify accumulated metric
        self.assertIn("mock_accuracy", report)
        self.assertAlmostEqual(report["mock_accuracy"], 0.9167, places=4)

    def test_multiple_metric_registration(self):
        """Test registering and retrieving multiple metrics."""
        # Register multiple metrics
        def mock_metric1(y_pred, y_true):
            return (y_pred == y_true).float().mean()

        def mock_metric2(y_pred, y_true):
            return torch.abs(y_pred - y_true).mean()

        self.registry.register(token="classification", fn=mock_metric1, name="accuracy", accumulate=True, arg_names=["y_pred", "y_true"])
        self.registry.register(token="classification", fn=mock_metric2, name="mae", accumulate=True, arg_names=["y_pred", "y_true"])

        # Retrieve both metrics
        metrics = self.registry.get("classification", ["accuracy", "mae"])

        # Verify retrieval
        self.assertEqual(len(metrics), 2)
        self.assertTrue("accuracy" in metrics)
        self.assertTrue("mae" in metrics)

    def test_empty_report(self):
        """Test reporting when no data is added."""
        report = self.registry.report("classification")
        self.assertEqual(report, {})


if __name__ == "__main__":
    unittest.main()

