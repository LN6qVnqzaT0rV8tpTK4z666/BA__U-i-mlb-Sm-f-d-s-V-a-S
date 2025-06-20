import unittest
from unittest.mock import patch, MagicMock
import duckdb
import pandas as pd
import torch
from BA__Programmierung.ml.datasets.dataset__torch__duckdb_iris import DatasetTorchDuckDBIris


class TestDatasetTorchDuckDBIris(unittest.TestCase):

    @patch("duckdb.connect")
    def test_initialize_dataset(self, mock_duckdb_connect):
        # Mock DuckDB connection and query execution
        mock_con = MagicMock()
        mock_duckdb_connect.return_value = mock_con

        mock_con.execute.return_value.fetchdf.return_value = pd.DataFrame({
            "SepalLengthCm": [5.1, 4.9],
            "SepalWidthCm": [3.5, 3.0],
            "PetalLengthCm": [1.4, 1.4],
            "PetalWidthCm": [0.2, 0.2],
            "Species": ["Iris-setosa", "Iris-versicolor"]
        })

        # Initialize the dataset
        dataset = DatasetTorchDuckDBIris(db_path="fake_path.duckdb", table_name="iris__dataset_csv", normalize=True)

        # Verify that the connection was made to the correct database
        mock_duckdb_connect.assert_called_with("fake_path.duckdb")

        # Verify that the dataframe was fetched from the correct table
        mock_con.execute.assert_called_with("SELECT * FROM iris__dataset_csv")

        # Ensure that the columns were renamed correctly
        self.assertTrue(all(col in dataset.input_cols for col in ["sepal_length", "sepal_width", "petal_length", "petal_width"]))
        self.assertEqual(dataset.target_col, "target")

        # Check that species were mapped to numeric values and filtered correctly
        self.assertTrue((dataset.dataframe["target"] == [0, 1]).all())  # Only 0 and 1 after filtering

    @patch("duckdb.connect")
    def test_species_mapping(self, mock_duckdb_connect):
        # Mock DuckDB connection and query execution
        mock_con = MagicMock()
        mock_duckdb_connect.return_value = mock_con

        mock_con.execute.return_value.fetchdf.return_value = pd.DataFrame({
            "SepalLengthCm": [5.1, 4.9, 6.3],
            "SepalWidthCm": [3.5, 3.0, 2.9],
            "PetalLengthCm": [1.4, 1.4, 5.6],
            "PetalWidthCm": [0.2, 0.2, 2.5],
            "Species": ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
        })

        # Initialize the dataset
        dataset = DatasetTorchDuckDBIris(db_path="fake_path.duckdb", table_name="iris__dataset_csv", normalize=True)

        # Verify that species were correctly mapped to numeric labels
        target = dataset.dataframe["target"].values
        self.assertEqual(list(target), [0, 1, 2])

    @patch("duckdb.connect")
    def test_class_filtering(self, mock_duckdb_connect):
        # Mock DuckDB connection and query execution
        mock_con = MagicMock()
        mock_duckdb_connect.return_value = mock_con

        mock_con.execute.return_value.fetchdf.return_value = pd.DataFrame({
            "SepalLengthCm": [5.1, 4.9, 6.3],
            "SepalWidthCm": [3.5, 3.0, 2.9],
            "PetalLengthCm": [1.4, 1.4, 5.6],
            "PetalWidthCm": [0.2, 0.2, 2.5],
            "Species": ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
        })

        # Initialize the dataset with class filtering
        dataset = DatasetTorchDuckDBIris(db_path="fake_path.duckdb", table_name="iris__dataset_csv", normalize=True)

        # Verify that only classes 0 and 1 are included (filtering Iris-virginica)
        target = dataset.dataframe["target"].values
        self.assertTrue((target == 0) | (target == 1))

    @patch("duckdb.connect")
    def test_normalization(self, mock_duckdb_connect):
        # Mock DuckDB connection and query execution
        mock_con = MagicMock()
        mock_duckdb_connect.return_value = mock_con

        mock_con.execute.return_value.fetchdf.return_value = pd.DataFrame({
            "SepalLengthCm": [5.1, 4.9],
            "SepalWidthCm": [3.5, 3.0],
            "PetalLengthCm": [1.4, 1.4],
            "PetalWidthCm": [0.2, 0.2],
            "Species": ["Iris-setosa", "Iris-versicolor"]
        })

        # Initialize the dataset with normalization
        dataset = DatasetTorchDuckDBIris(db_path="fake_path.duckdb", table_name="iris__dataset_csv", normalize=True)

        # Check if normalization is applied (values should be between 0 and 1)
        self.assertTrue(torch.all(dataset.X >= 0) and torch.all(dataset.X <= 1))

    @patch("duckdb.connect")
    def test_invalid_column_count(self, mock_duckdb_connect):
        # Mock DuckDB connection and query execution with incorrect column count
        mock_con = MagicMock()
        mock_duckdb_connect.return_value = mock_con

        mock_con.execute.return_value.fetchdf.return_value = pd.DataFrame({
            "SepalLengthCm": [5.1, 4.9],
            "SepalWidthCm": [3.5, 3.0],
            "Species": ["Iris-setosa", "Iris-versicolor"]
        })

        # Initialize the dataset and check for ValueError
        with self.assertRaises(ValueError):
            dataset = DatasetTorchDuckDBIris(db_path="fake_path.duckdb", table_name="iris__dataset_csv", normalize=True)


if __name__ == "__main__":
    unittest.main()
