# BA__Projekt/BA__Programmierung/ml/datasets/dataset__torch__condition-based-maintenance-of-naval-propulsion-plants.py

import pandas as pd

from BA__Programmierung.ml.datasets.dataset__torch__base_tabular import (
    BaseTabularDataset,
)


class NavalPropulsionDataset(BaseTabularDataset):
    """
    PyTorch dataset class for the Condition Based Maintenance of Naval Propulsion Plants dataset.

    Loads data from a CSV file without headers, checks column count,
    assigns string column names (as BaseTabularDataset expects string column names),
    and initializes the BaseTabularDataset with optional normalization.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file containing the dataset.
    normalize : bool, default=True
        Whether to normalize features and targets.

    Raises
    ------
    ValueError
        If the CSV file does not have exactly 18 columns.

    Attributes
    ----------
    input_cols : list of str
        List of feature column names as strings representing indices.
    target_cols : list of str
        List of target column names as strings representing indices.

    Examples
    --------
    >>> dataset = NavalPropulsionDataset(csv_path="data/naval_propulsion.csv")
    """

    def __init__(self, csv_path, normalize=True):
        """
        Initialize the Naval Propulsion dataset.

        Parameters
        ----------
        csv_path : str
            Path to the CSV data file.
        normalize : bool, default=True
            Whether to apply normalization to features and targets.
        """
        df = pd.read_csv(csv_path, header=None)

        if df.shape[1] != 18:
            raise ValueError(f"Expected 18 columns, got {df.shape[1]} in {csv_path}")

        # Feature columns are all except last two, targets are last two columns
        input_cols = list(range(df.shape[1] - 2))
        target_cols = list(range(df.shape[1] - 2, df.shape[1]))

        # Convert column indices to strings for BaseTabularDataset compatibility
        input_cols = [str(i) for i in input_cols]
        target_cols = [str(i) for i in target_cols]

        # Rename dataframe columns to string indices
        df.columns = [str(i) for i in range(df.shape[1])]

        super().__init__(
            dataframe=df,
            input_cols=input_cols,
            target_cols=target_cols,
            normalize=normalize,
            classification=False,
        )

