# BA__Programmierung/ml/datasets/dataset__torch__combined-cycle-power-plant.py

import pandas as pd

from BA__Programmierung.ml.datasets.dataset__torch__base_tabular import (
    BaseTabularDataset,
)


class DatasetTorchCombinedCyclePowerPlant(BaseTabularDataset):
    """
    PyTorch dataset class for the Combined Cycle Power Plant dataset.

    Loads data from a CSV file, selects relevant feature columns and target column,
    and initializes the BaseTabularDataset with optional normalization.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file containing the dataset.
    normalize : bool, default=True
        Whether to apply feature and target normalization.

    Attributes
    ----------
    input_cols : list of str
        List of feature column names: ["AT", "V", "AP", "RH"].
    target_col : list of str
        List containing the target column name: ["PE"].

    Examples
    --------
    >>> dataset = DatasetTorchCombinedCyclePowerPlant(csv_path="data/ccpp.csv")
    """

    def __init__(self, csv_path, normalize=True):
        """
        Initialize the Combined Cycle Power Plant dataset.

        Parameters
        ----------
        csv_path : str
            Path to the CSV data file.
        normalize : bool, default=True
            Whether to normalize features and targets.
        """
        df = pd.read_csv(csv_path)
        input_cols = ["AT", "V", "AP", "RH"]
        target_col = ["PE"]
        super().__init__(
            dataframe=df,
            input_cols=input_cols,
            target_cols=target_col,
            normalize=normalize,
            classification=False,
        )
