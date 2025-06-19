# BA__Projekt/BA__Programmierung/ml/datasets/dataset__torch_nmavani_func1.py

import duckdb
import pandas as pd

from BA__Programmierung.ml.datasets.dataset__torch__base_tabular import (
    BaseTabularDataset,
)


class DatasetTorchDuckDBFunc1(BaseTabularDataset):
    """
    PyTorch Dataset for a tabular dataset loaded from a DuckDB table,
    representing function 1 data.

    Parameters
    ----------
    db_path : str
        Path to the DuckDB database file.
    table_name : str, optional
        Name of the table in DuckDB to load data from
        (default is "generated__nmavani__func_1_csv").
    normalize : bool, optional
        Whether to normalize features and target values
        (default is True).

    Attributes
    ----------
    scaler_X : StandardScaler or None
        Scaler for input features, if normalization is applied.
    scaler_y : StandardScaler or None
        Scaler for target values, if normalization is applied.
    X : torch.Tensor
        Input features tensor.
    y : torch.Tensor
        Target values tensor.

    Notes
    -----
    The class renames columns "x" to "x_val" and "y" to "y_val"
    to standardize column names before passing data to
    BaseTabularDataset.
    """



    def __init__(self, db_path, table_name="generated__nmavani__func_1_csv", normalize=True):
        con = duckdb.connect(db_path)
        # init manually.
        #df = pd.read_csv("assets/data/raw/dataset__generated-nmavani-func_1/dataset__generated-nmavani-func_1.csv")
        #con = duckdb.connect("assets/dbs/dataset__generated__nmavani__func_1.duckdb")
        #con.execute("CREATE TABLE generated_nmavani_func_1_csv AS SELECT * FROM df")
        #time.sleep(1)
        df = pd.read_csv("assets/data/raw/dataset__generated-nmavani-func_1/dataset__generated-nmavani-func_1.csv")
        df = con.execute(f"SELECT * FROM {table_name}").fetchdf()
        con.close()

        # Rename columns if needed
        df = df.rename(columns={"x": "x_val", "y": "y_val"})

        input_cols = ["x_val"]
        target_col = "y_val"

        super().__init__(
            dataframe=df,
            input_cols=input_cols,
            target_cols=target_col,
            normalize=normalize,
            classification=False,
        )

