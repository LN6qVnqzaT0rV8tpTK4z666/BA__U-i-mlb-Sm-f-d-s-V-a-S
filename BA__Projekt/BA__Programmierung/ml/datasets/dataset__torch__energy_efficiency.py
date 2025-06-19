# BA__Projekt/BA__Programmierung/ml/datasets/dataset__torch__energy_efficiency.py

import pandas as pd

from BA__Programmierung.ml.datasets.dataset__torch__base_tabular import (
    BaseTabularDataset,
)


class EnergyEfficiencyDataset(BaseTabularDataset):
    """
    PyTorch dataset class for the Energy Efficiency dataset.

    Loads data from a CSV file, drops rows with missing values,
    allows specifying a target column (default is the last column),
    and initializes the BaseTabularDataset with optional normalization.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file containing the dataset.
    target_col : str or None, optional
        Name of the target column. If None, the last column is used as target.
    normalize : bool, default=True
        Whether to normalize features and target values.

    Attributes
    ----------
    input_cols : list of str
        List of feature column names.
    target_col : str
        Target column name.

    Examples
    --------
    >>> dataset = EnergyEfficiencyDataset(csv_path="data/energy_efficiency.csv")
    """

    def __init__(self, csv_path, target_col=None, normalize=True):
        """
        Initialize the Energy Efficiency dataset.

        Parameters
        ----------
        csv_path : str
            Path to the CSV data file.
        target_col : str or None, optional
            Target column name; defaults to last column if None.
        normalize : bool, default=True
            Whether to apply normalization to features and target.
        """
        df = pd.read_csv(csv_path).dropna()

        if target_col is None:
            # Default: last column as target
            target_col = df.columns[-1]

        input_cols = [col for col in df.columns if col != target_col]

        super().__init__(
            dataframe=df,
            input_cols=input_cols,
            target_cols=target_col,
            normalize=normalize,
            classification=False,
        )

