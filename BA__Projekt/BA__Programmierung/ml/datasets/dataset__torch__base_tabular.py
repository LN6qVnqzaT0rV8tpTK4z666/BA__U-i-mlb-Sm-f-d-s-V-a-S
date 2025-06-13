# BA__Projekt/BA__Programmierung/ml/datasets/dataset__torch.py

import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


class BaseTabularDataset(Dataset):
    def __init__(self, dataframe, input_cols, target_cols, normalize=True, classification=False):
        """
        Generische Basis-Klasse für tabellarische PyTorch-Datasets.

        Args:
            dataframe (pd.DataFrame): Eingelesener DataFrame mit Daten.
            input_cols (list of str): Liste der Feature-Spaltennamen.
            target_cols (str or list of str): Name oder Liste der Zielspalten.
            normalize (bool): Ob Features (und bei Regression auch Zielwerte) skaliert werden.
            classification (bool): Ob es sich um Klassifikation handelt (Labels als Long Tensor).

        Funktionen:
            - Skaliert Features bei Regression und Klassifikation optional.
            - Skaliert Zielwerte nur bei Regression.
            - Bietet inverse_transform_y für Regressionsvorhersagen.
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
            # Labels als 1D LongTensor (Klassenlabels)
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
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def inverse_transform_y(self, y_pred):
        """
        Falls Skala auf Zielwerte angewendet wurde, wird hier zurücktransformiert.
        Bei Klassifikation einfach Originalwerte zurückgeben.
        """
        if self.scaler_y is not None:
            return torch.tensor(
                self.scaler_y.inverse_transform(y_pred.detach().cpu().numpy()),
                dtype=torch.float32,
            )
        return y_pred
    
    @property
    def features(self):
        return self.X

    @property
    def labels(self):
        return self.y
