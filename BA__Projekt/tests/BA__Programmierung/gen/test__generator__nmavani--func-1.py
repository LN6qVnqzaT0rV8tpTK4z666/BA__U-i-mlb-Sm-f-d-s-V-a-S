# BA__Projekt/tests/BA__Programmierung/gen/test__generator__nmavani--func-1.py

import numpy as np
import pandas as pd
import unittest

from BA__Programmierung.gen.generator__nmavani__func_1 import generate_dataset, save_dataset
from pathlib import Path
from unittest.mock import patch


class TestGeneratorFunc1(unittest.TestCase):
    
    def test_generate_dataset(self):
        # Generate a dataset with default parameters
        df = generate_dataset(n_samples=10, x_min=-5, x_max=5, seed=42)

        # Check the shape of the dataframe (should be 10 samples, 2 columns)
        self.assertEqual(df.shape, (10, 2))

        # Check the column names
        self.assertListEqual(list(df.columns), ["x", "y"])

        # Check if the first few values match the expected generated values (based on the formula)
        expected_x_values = [-5.0, -4.444, -3.889, -3.333, -2.778]
        expected_y_values = [
            7 * np.sin(-5.0) + 3 * np.abs(-5.0 / 2) * np.random.randn(),
            7 * np.sin(-4.444) + 3 * np.abs(-4.444 / 2) * np.random.randn(),
            7 * np.sin(-3.889) + 3 * np.abs(-3.889 / 2) * np.random.randn(),
            7 * np.sin(-3.333) + 3 * np.abs(-3.333 / 2) * np.random.randn(),
            7 * np.sin(-2.778) + 3 * np.abs(-2.778 / 2) * np.random.randn(),
        ]

        # Check if generated values are close to expected values (accounting for epsilon randomness)
        self.assertAlmostEqual(df['x'][0], expected_x_values[0], places=2)
        self.assertAlmostEqual(df['x'][1], expected_x_values[1], places=2)

    @patch("os.makedirs")
    @patch("pandas.DataFrame.to_csv")
    def test_save_dataset(self, mock_to_csv, mock_makedirs):
        # Create a mock DataFrame
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})

        # Define output paths for the test
        output_paths = [
            Path("assets/data/source") / "test_dataset.csv",
            Path("assets/data/raw/dataset__generated__nmavani__func_1/") / "test_dataset.csv",
        ]

        # Call save_dataset function
        save_dataset(df, output_paths)

        # Check if os.makedirs was called to create the directories
        mock_makedirs.assert_any_call(Path("assets/data/source").parent, exist_ok=True)
        mock_makedirs.assert_any_call(Path("assets/data/raw/dataset__generated__nmavani__func_1/").parent, exist_ok=True)

        # Check if the DataFrame's to_csv function was called for both output paths
        mock_to_csv.assert_any_call(output_paths[0], index=False)
        mock_to_csv.assert_any_call(output_paths[1], index=False)

    @patch("builtins.print")  # Mock the print statement to capture output
    def test_save_dataset_print_output(self, mock_print):
        # Create a mock DataFrame
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})

        # Define output paths for the test
        output_paths = [
            Path("assets/data/source") / "test_dataset.csv",
            Path("assets/data/raw/dataset__generated__nmavani__func_1/") / "test_dataset.csv",
        ]

        # Call save_dataset function
        save_dataset(df, output_paths)

        # Check if the print statement was called with the correct path
        mock_print.assert_any_call(f"Datensatz gespeichert unter: {output_paths[0]}")
        mock_print.assert_any_call(f"Datensatz gespeichert unter: {output_paths[1]}")

    def test_generate_dataset_with_custom_seed(self):
        # Generate dataset with a custom seed
        df1 = generate_dataset(n_samples=10, x_min=-5, x_max=5, seed=123)
        df2 = generate_dataset(n_samples=10, x_min=-5, x_max=5, seed=123)

        # Ensure that datasets with the same seed are identical
        pd.testing.assert_frame_equal(df1, df2)

    @patch("os.makedirs")
    @patch("pandas.DataFrame.to_csv")
    def test_save_dataset_existing_directories(self, mock_to_csv, mock_makedirs):
        # Create a mock DataFrame
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})

        # Define existing output paths
        output_paths = [
            Path("assets/data/source") / "existing_dataset.csv",
            Path("assets/data/raw/dataset__generated__nmavani__func_1/") / "existing_dataset.csv",
        ]

        # Simulate existing directories
        mock_makedirs.return_value = None

        # Call save_dataset function
        save_dataset(df, output_paths)

        # Verify that os.makedirs was called with exist_ok=True to not raise errors
        mock_makedirs.assert_any_call(Path("assets/data/source").parent, exist_ok=True)
        mock_makedirs.assert_any_call(Path("assets/data/raw/dataset__generated__nmavani__func_1/").parent, exist_ok=True)

        # Verify that to_csv was called
        mock_to_csv.assert_any_call(output_paths[0], index=False)
        mock_to_csv.assert_any_call(output_paths[1], index=False)


if __name__ == '__main__':
    unittest.main()

