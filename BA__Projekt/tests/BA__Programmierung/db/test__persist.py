import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import duckdb
from io import StringIO
from BA__Projekt. BA__Programmierung.db.persist import find_csv_files, find_sqlite_for_dataset, derive_dataset_name, db__persist


class TestPersist(unittest.TestCase):

    @patch("pathlib.Path.glob")
    def test_find_csv_files(self, mock_glob):
        # Prepare mock return value
        mock_glob.return_value = [
            Path("dataset__boston_housing/data.csv"),
            Path("dataset__new_dataset/data.csv"),
        ]

        base_path = Path("/raw_data")

        # Test the find_csv_files function
        result = find_csv_files(base_path)

        # Verify if the glob method was called correctly
        mock_glob.assert_called_once_with("dataset__*/**/*.csv")

        # Verify if the correct CSV files were found
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], Path("dataset__boston_housing/data.csv"))
        self.assertEqual(result[1], Path("dataset__new_dataset/data.csv"))

    @patch("pathlib.Path.glob")
    def test_find_sqlite_for_dataset(self, mock_glob):
        # Prepare mock return value
        mock_glob.return_value = [Path("dataset__boston_housing/dataset.sqlite")]

        dataset_dir = Path("/raw_data/dataset__boston_housing")

        # Test the find_sqlite_for_dataset function
        result = find_sqlite_for_dataset(dataset_dir)

        # Verify if the glob method was called correctly
        mock_glob.assert_any_call("*.sqlite")
        mock_glob.assert_any_call("*.db")

        # Verify if the correct SQLite file was found
        self.assertEqual(result, Path("dataset__boston_housing/dataset.sqlite"))

    @patch("pathlib.Path.glob")
    def test_find_sqlite_for_dataset_no_sqlite(self, mock_glob):
        # Simulate no SQLite files being found
        mock_glob.return_value = []

        dataset_dir = Path("/raw_data/dataset__boston_housing")

        # Test the find_sqlite_for_dataset function
        result = find_sqlite_for_dataset(dataset_dir)

        # Verify if no SQLite files were found
        self.assertIsNone(result)

    def test_derive_dataset_name(self):
        # Test that the dataset name is derived correctly
        csv_path = Path("/raw_data/dataset__boston_housing/data.csv")
        result = derive_dataset_name(csv_path)

        # Verify that the dataset name is derived correctly from the path
        self.assertEqual(result, "dataset__boston_housing")

    @patch("duckdb.connect")
    @patch("BA__Projekt. BA__Programmierung.db.persist.find_csv_files")
    @patch("BA__Projekt. BA__Programmierung.db.persist.find_sqlite_for_dataset")
    @patch("rich.console.Console.log")
    def test_db__persist(self, mock_log, mock_find_sqlite, mock_find_csv, mock_duckdb_connect):
        # Mock CSV files and SQLite database finding
        mock_find_csv.return_value = [Path("dataset__boston_housing/data.csv")]
        mock_find_sqlite.return_value = Path("dataset__boston_housing/dataset.sqlite")

        # Mock DuckDB connection and execution
        mock_con = MagicMock()
        mock_duckdb_connect.return_value = mock_con

        # Mock execution results
        mock_con.execute.return_value = None
        mock_con.execute.return_value.fetchall.return_value = [("table1",), ("table2",)]

        # Mock Console log method
        mock_log.return_value = None

        # Call db__persist function
        db__persist()

        # Verify that DuckDB was connected
        mock_duckdb_connect.assert_called_with(Path("path_to_db/dataset__boston_housing.duckdb"))

        # Verify that DuckDB table creation was executed
        mock_con.execute.assert_any_call("""
            CREATE TABLE dataset_boston_housing_csv AS
            SELECT * FROM read_csv_auto('dataset__boston_housing/data.csv')
        """)

        # Verify that SQLite database was attached
        mock_con.execute.assert_any_call("""
            ATTACH DATABASE 'dataset__boston_housing/dataset.sqlite' AS sqlite_db (TYPE SQLITE)
        """)

        # Verify that tables were imported from the SQLite database
        mock_con.execute.assert_any_call("""
            CREATE TABLE dataset_boston_housing__table1 AS
            SELECT * FROM sqlite_db.table1
        """)

        mock_con.execute.assert_any_call("""
            CREATE TABLE dataset_boston_housing__table2 AS
            SELECT * FROM sqlite_db.table2
        """)

        # Verify that the persistence log message was shown
        mock_log.assert_any_call("[green]✔ Persistiert:[/] dataset__boston_housing.duckdb")

    @patch("rich.console.Console.log")
    @patch("BA__Projekt. BA__Programmierung.db.persist.find_csv_files")
    def test_db__persist_no_csv_files(self, mock_find_csv, mock_log):
        # Simulate no CSV files found
        mock_find_csv.return_value = []

        # Call db__persist function
        db__persist()

        # Verify that the process did not attempt to persist anything
        mock_log.assert_any_call("Starte Persistierung aller Rohdaten...")
        mock_log.assert_any_call("[bold green]Alle nicht-persistierten Datensätze wurden gespeichert.[/]")

if __name__ == "__main__":
    unittest.main()
