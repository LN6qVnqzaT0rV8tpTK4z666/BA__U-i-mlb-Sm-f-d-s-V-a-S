import unittest

from metrics_treshholds import metric_thresholds


class TestMetricsThresholds(unittest.TestCase):

    def test_metric_thresholds(self):
        """
        Test the correctness of metric thresholds.
        Ensure each metric has defined 'positive' and 'negative' thresholds.
        """
        # List of metrics to check for thresholds
        metrics = [
            "mse", "rmse", "mae", "mape", "r2_score", "ece", "ace", "elbo", 
            "nll_gaussian", "mean_pred_variance", "energy_score", "kl_divergence_normal", 
            "mutual_information", "calibration_error", "uda", "ncg", "meta_metric__bnn_ednn"
        ]

        # Ensure each metric has defined thresholds for both positive and negative
        for metric in metrics:
            with self.subTest(metric=metric):
                self.assertIn(metric, metric_thresholds)
                self.assertIn("positive", metric_thresholds[metric])
                self.assertIn("negative", metric_thresholds[metric])

    def test_metric_threshold_values(self):
        """
        Test that the threshold values for each metric are correct.
        """
        # Sample expected threshold values for metrics
        expected_thresholds = {
            "mse": {"positive": 0.5, "negative": 2.0},
            "rmse": {"positive": 0.7, "negative": 2.5},
            "mae": {"positive": 0.5, "negative": 1.5},
            "mape": {"positive": 10, "negative": 30},
            "r2_score": {"positive": 0.8, "negative": 0.2},
            "ece": {"positive": 0.1, "negative": 0.2},
            "ace": {"positive": 0.1, "negative": 0.2},
            "elbo": {"positive": 1.0, "negative": 0.5},
            "nll_gaussian": {"positive": 1.0, "negative": 2.0},
            "mean_pred_variance": {"positive": 0.1, "negative": 1.0},
            "energy_score": {"positive": 1.0, "negative": 2.0},
            "kl_divergence_normal": {"positive": 0.1, "negative": 0.5},
            "mutual_information": {"positive": 0.5, "negative": 0.1},
            "calibration_error": {"positive": 0.1, "negative": 0.2},
            "uda": {"positive": 0.5, "negative": 0.2},
            "ncg": {"positive": 0.9, "negative": 0.7},
            "meta_metric__bnn_ednn": {"positive": 0.5, "negative": 0.2}
        }

        # Ensure each metric has the expected threshold values
        for metric, thresholds in expected_thresholds.items():
            with self.subTest(metric=metric):
                self.assertEqual(metric_thresholds.get(metric), thresholds)

    def test_missing_metrics(self):
        """
        Test that there are no missing metrics in the thresholds dictionary.
        """
        # List of all possible metrics expected to have thresholds
        expected_metrics = [
            "mse", "rmse", "mae", "mape", "r2_score", "ece", "ace", "elbo", 
            "nll_gaussian", "mean_pred_variance", "energy_score", "kl_divergence_normal", 
            "mutual_information", "calibration_error", "uda", "ncg", "meta_metric__bnn_ednn"
        ]

        # Check if any metric is missing from the threshold dictionary
        missing_metrics = [metric for metric in expected_metrics if metric not in metric_thresholds]
        self.assertEqual(missing_metrics, [])

if __name__ == "__main__":
    unittest.main()
