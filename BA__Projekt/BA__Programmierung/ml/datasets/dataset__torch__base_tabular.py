# BA__Projekt/BA__Programmierung/ml/datasets/dataset__torch.py

import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


class BaseTabularDataset(Dataset):
    """
    Generic base class for tabular PyTorch datasets.

    This class supports optional feature normalization and target normalization
    (for regression tasks), and handles classification labels appropriately.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input data as a pandas DataFrame.
    input_cols : list of str
        List of column names used as features.
    target_cols : str or list of str
        Name or list of names of target columns.
    normalize : bool, default=True
        Whether to scale features (and target values for regression) using StandardScaler.
    classification : bool, default=False
        Whether the dataset is for classification tasks (labels as LongTensor).

    Attributes
    ----------
    scaler_X : sklearn.preprocessing.StandardScaler or None
        Scaler applied to features if normalization enabled.
    scaler_y : sklearn.preprocessing.StandardScaler or None
        Scaler applied to targets if regression and normalization enabled.
    X : torch.Tensor
        Feature tensor.
    y : torch.Tensor
        Target tensor (LongTensor for classification, FloatTensor for regression).

    Methods
    -------
    inverse_transform_y(y_pred)
        Inverse transform regression predictions to original scale.
    features
        Property to access feature tensor.
    labels
        Property to access label tensor.
    """

    def __init__(self, dataframe, input_cols, target_cols, normalize=True, classification=False):
        """
        Initialize the dataset with feature and target data.

        Parameters
        ----------
        dataframe : pd.DataFrame
            DataFrame containing the data.
        input_cols : list of str
            Feature column names.
        target_cols : str or list of str
            Target column name(s).
        normalize : bool, default=True
            Whether to normalize features (and target values if regression).
        classification : bool, default=False
            Whether dataset is for classification.
        """
        self.input_cols = input_cols
        self.target_cols = (
            target_cols if isinstance(target_cols, list) else [target_cols]
        )
        self.classification = classification

        X = dataframe[self.input_cols].values
        y = dataframe[self.target_cols].values

        if normalize:
            self.scaler_X = StandardScaler()
            X = self.scaler_X.fit_transform(X)
        else:
            self.scaler_X = None

        if classification:
            # Labels as 1D LongTensor for classification
            self.y = torch.tensor(y.squeeze(), dtype=torch.long)
            self.scaler_y = None
        else:
            if normalize:
                self.scaler_y = StandardScaler()
                y = self.scaler_y.fit_transform(y)
            else:
                self.scaler_y = None
            self.y = torch.tensor(y, dtype=torch.float32)

        self.X = torch.tensor(X, dtype=torch.float32)

    def __len__(self):
        """
        Returns
        -------
        int
            Number of samples in the dataset.
        """
        return len(self.X)

    def __getitem__(self, idx):
        """
        Retrieve feature and label tensors at the specified index.

        Parameters
        ----------
        idx : int
            Index of the sample to retrieve.

        Returns
        -------
        tuple of (torch.Tensor, torch.Tensor)
            Feature tensor and corresponding label tensor.
        """
        return self.X[idx], self.y[idx]

    def inverse_transform_y(self, y_pred):
        """
        Inverse transform target predictions to original scale if normalization was applied.
        For classification, returns the input as-is.

        Parameters
        ----------
        y_pred : torch.Tensor
            Predicted target values.

        Returns
        -------
        torch.Tensor
            Target values in the original scale.
        """
        if self.scaler_y is not None:
            return torch.tensor(
                self.scaler_y.inverse_transform(y_pred.detach().cpu().numpy()),
                dtype=torch.float32,
            )
        return y_pred

    @property
    def features(self):
        """
        Returns
        -------
        torch.Tensor
            Tensor containing feature data.
        """
        return self.X

    @property
    def labels(self):
        """
        Returns
        -------
        torch.Tensor
            Tensor containing label data.
        """
        return self.y
