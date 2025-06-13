# BA__Projekt/BA__Programmierung/ml/datasets/dataset__torch_duckdb_iris.py

import duckdb

from BA__Programmierung.ml.datasets.dataset__torch__base_tabular import (
    BaseTabularDataset,
)


class DatasetTorchDuckDBIris(BaseTabularDataset):
    """
    PyTorch dataset class for the Iris dataset loaded from a DuckDB database.

    Loads data from a DuckDB database table, renames columns, maps species to class labels,
    optionally filters to two classes, and initializes BaseTabularDataset for classification.

    Parameters
    ----------
    db_path : str
        Path to the DuckDB database file.
    table_name : str, default="iris__dataset_csv"
        Name of the table containing the Iris dataset.
    normalize : bool, default=True
        Whether to normalize the feature columns.

    Attributes
    ----------
    input_cols : list of str
        Feature column names: sepal_length, sepal_width, petal_length, petal_width.
    target_col : str
        Target column name: target (mapped species labels).

    Examples
    --------
    >>> dataset = DatasetTorchDuckDBIris(db_path="iris.duckdb")
    """

    def __init__(self, db_path, table_name="iris__dataset_csv", normalize=True):
        """
        Initialize the Iris dataset from DuckDB.

        Parameters
        ----------
        db_path : str
            Path to the DuckDB database file.
        table_name : str, optional
            Table name in the database (default is "iris__dataset_csv").
        normalize : bool, default=True
            Whether to normalize features.
        """
        con = duckdb.connect(db_path)
        df = con.execute(f"SELECT * FROM {table_name}").fetchdf()
        con.close()

        df = df.rename(
            columns={
                "SepalLengthCm": "sepal_length",
                "SepalWidthCm": "sepal_width",
                "PetalLengthCm": "petal_length",
                "PetalWidthCm": "petal_width",
                "Species": "species",
            }
        )

        # Map species names to numeric class labels
        df["target"] = df["species"].map(
            {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
        )

        # Optional: filter to classes 0 and 1 only
        df = df[df["target"] < 2]

        input_cols = [
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width",
        ]
        target_col = "target"

        # classification=True converts labels to LongTensor and disables target scaling
        super().__init__(
            dataframe=df,
            input_cols=input_cols,
            target_cols=target_col,
            normalize=normalize,
            classification=True,
        )
