# BA__Projekt/BA__Programmierung/ml/datasets/dataset__torch__concrete-compressive-strength.py

import pandas as pd

from BA__Programmierung.ml.datasets.dataset__torch__base_tabular import (
    BaseTabularDataset,
)


class DatasetTorchConcreteCompressiveStrength(BaseTabularDataset):
    """
    PyTorch dataset class for the Concrete Compressive Strength dataset.

    Loads data from a CSV file, cleans column names, separates features and target,
    and initializes the BaseTabularDataset with optional normalization.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file containing the dataset.
    normalize : bool, default=True
        Whether to normalize features and target values.

    Attributes
    ----------
    input_cols : list of str
        List of feature column names.
    target_col : str
        Name of the target column.

    Examples
    --------
    >>> dataset = DatasetTorchConcreteCompressiveStrength(csv_path="data/concrete.csv")
    """

    def __init__(self, csv_path, normalize=True):
        """
        Initialize the Concrete Compressive Strength dataset.

        Parameters
        ----------
        csv_path : str
            Path to the CSV data file.
        normalize : bool, default=True
            Whether to apply normalization to features and target.
        """
        df = pd.read_csv(csv_path)
        # Remove whitespace from column names
        df.columns = [col.strip() for col in df.columns]

        input_cols = df.columns[:-1].tolist()
        target_col = df.columns[-1]

        super().__init__(
            dataframe=df,
            input_cols=input_cols,
            target_cols=target_col,
            normalize=normalize,
            classification=False,
        )

