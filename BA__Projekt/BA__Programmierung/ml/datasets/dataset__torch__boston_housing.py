# BA__Projekt/BA__Programmierung/ml/datasets/dataset__torch__boston-housing.py

import pandas as pd

from BA__Programmierung.ml.datasets.dataset__torch__base_tabular import (
    BaseTabularDataset,
)


class DatasetTorchBostonHousing(BaseTabularDataset):
    """
    PyTorch dataset class for the Boston Housing dataset.

    This dataset loads data from a whitespace-delimited CSV file without headers,
    assigns appropriate column names, and uses the BaseTabularDataset class
    for tabular data handling with normalization enabled.

    Parameters
    ----------
    csv_path : str
        Path to the Boston Housing CSV file.

    Attributes
    ----------
    input_cols : list of str
        Names of the feature columns.
    target_col : str
        Name of the target column ('MEDV').

    Examples
    --------
    >>> dataset = DatasetTorchBostonHousing(csv_path='data/boston_housing.csv')
    """

    def __init__(self, csv_path):
        """
        Initialize the Boston Housing dataset.

        Loads the CSV file with predefined column names, splits features and target,
        and initializes the BaseTabularDataset with normalization.

        Parameters
        ----------
        csv_path : str
            Path to the Boston Housing dataset CSV file.
        """
        column_names = [
            "CRIM",
            "ZN",
            "INDUS",
            "CHAS",
            "NOX",
            "RM",
            "AGE",
            "DIS",
            "RAD",
            "TAX",
            "PTRATIO",
            "B",
            "LSTAT",
            "MEDV",
        ]

        # Load CSV with whitespace delimiter, no header, assign column names
        df = pd.read_csv(csv_path, header=None, delimiter=r"\s+", names=column_names)

        input_cols = column_names[:-1]
        target_col = column_names[-1]

        # Call the BaseTabularDataset constructor with normalization enabled
        super().__init__(df, input_cols, target_col, normalize=True, classification=False)

