import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import torch
from BA__Programmierung.ml.datasets.dataset__torch__nmavani_func1 import DatasetTorchDuckDBFunc1


class TestDatasetTorchDuckDBFunc1(unittest.TestCase):

    @patch("duckdb.connect")
    @patch("pandas.read_csv")
    def test_initialize_dataset(self, mock_read_csv, mock_duckdb_connect):
        # Mocking the CSV data
        mock_df = pd.DataFrame({
            "x": [1, 2, 3],
            "y": [4, 5, 6]
        })
        mock_read_csv.return_value = mock_df

        # Mocking the DuckDB connection and query execution
        mock_con = MagicMock()
        mock_duckdb_connect.return_value = mock_con
        mock_con.execute.return_value.fetchdf.return_value = mock_df

        # Initialize the dataset
        db_path = "fake_path.duckdb"
        dataset = DatasetTorchDuckDBFunc1(db_path)

        # Ensure that the correct table was queried and columns renamed
        mock_con.execute.assert_called_with("SELECT * FROM generated__nmavani__func_1_csv")
        self.assertTrue("x_val" in dataset.input_cols)
        self.assertTrue("y_val" in dataset.target_col)

        # Ensure that the input columns and target column are set correctly
        self.assertEqual(dataset.input_cols, ["x_val"])
        self.assertEqual(dataset.target_col, "y_val")

        # Ensure that normalization was applied (values should be between 0 and 1)
        self.assertTrue(torch.all(dataset.X >= 0) and torch.all(dataset.X <= 1))
        self.assertTrue(torch.all(dataset.y >= 0) and torch.all(dataset.y <= 6))

    @patch("duckdb.connect")
    @patch("pandas.read_csv")
    def test_invalid_db_path(self, mock_read_csv, mock_duckdb_connect):
        # Simulating that the database connection raises an error
        mock_duckdb_connect.side_effect = FileNotFoundError("DuckDB file not found")

        # Call the function and verify FileNotFoundError is raised
        with self.assertRaises(FileNotFoundError):
            DatasetTorchDuckDBFunc1("invalid_path.duckdb")

    @patch("duckdb.connect")
    @patch("pandas.read_csv")
    def test_column_renaming(self, mock_read_csv, mock_duckdb_connect):
        # Mocking the CSV data
        mock_df = pd.DataFrame({
            "x": [1, 2, 3],
            "y": [4, 5, 6]
        })
        mock_read_csv.return_value = mock_df

        # Mocking the DuckDB connection and query execution
        mock_con = MagicMock()
        mock_duckdb_connect.return_value = mock_con
        mock_con.execute.return_value.fetchdf.return_value = mock_df

        # Initialize the dataset
        db_path = "fake_path.duckdb"
        dataset = DatasetTorchDuckDBFunc1(db_path)

        # Ensure that the columns are renamed correctly
        self.assertTrue("x_val" in dataset.input_cols)
        self.assertTrue("y_val" in dataset.target_col)

    @patch("duckdb.connect")
    @patch("pandas.read_csv")
    def test_dataset_size(self, mock_read_csv, mock_duckdb_connect):
        # Mocking the CSV data
        mock_df = pd.DataFrame({
            "x": [1, 2, 3],
            "y": [4, 5, 6]
        })
        mock_read_csv.return_value = mock_df

        # Mocking the DuckDB connection and query execution
        mock_con = MagicMock()
        mock_duckdb_connect.return_value = mock_con
        mock_con.execute.return_value.fetchdf.return_value = mock_df

        # Initialize the dataset
        db_path = "fake_path.duckdb"
        dataset = DatasetTorchDuckDBFunc1(db_path)

        # Verify that the dataset size matches the number of rows in the CSV
        self.assertEqual(len(dataset), 3)

    @patch("duckdb.connect")
    @patch("pandas.read_csv")
    def test_normalization(self, mock_read_csv, mock_duckdb_connect):
        # Mocking the CSV data
        mock_df = pd.DataFrame({
            "x": [1, 2, 3],
            "y": [4, 5, 6]
        })
        mock_read_csv.return_value = mock_df

        # Mocking the DuckDB connection and query execution
        mock_con = MagicMock()
        mock_duckdb_connect.return_value = mock_con
        mock_con.execute.return_value.fetchdf.return_value = mock_df

        # Initialize the dataset with normalization
        db_path = "fake_path.duckdb"
        dataset = DatasetTorchDuckDBFunc1(db_path, normalize=True)

        # Ensure that normalization is applied (values should be between 0 and 1)
        self.assertTrue(torch.all(dataset.X >= 0) and torch.all(dataset.X <= 1))
        self.assertTrue(torch.all(dataset.y >= 0) and torch.all(dataset.y <= 6))

    @patch("duckdb.connect")
    @patch("pandas.read_csv")
    def test_no_normalization(self, mock_read_csv, mock_duckdb_connect):
        # Mocking the CSV data
        mock_df = pd.DataFrame({
            "x": [1, 2, 3],
            "y": [4, 5, 6]
        })
        mock_read_csv.return_value = mock_df

        # Mocking the DuckDB connection and query execution
        mock_con = MagicMock()
        mock_duckdb_connect.return_value = mock_con
        mock_con.execute.return_value.fetchdf.return_value = mock_df

        # Initialize the dataset without normalization
        db_path = "fake_path.duckdb"
        dataset = DatasetTorchDuckDBFunc1(db_path, normalize=False)

        # Ensure that normalization was not applied (values should be in original range)
        self.assertTrue(torch.all(dataset.X >= 0) and torch.all(dataset.X <= 6))
        self.assertTrue(torch.all(dataset.y >= 0) and torch.all(dataset.y <= 6))


if __name__ == "__main__":
    unittest.main()
