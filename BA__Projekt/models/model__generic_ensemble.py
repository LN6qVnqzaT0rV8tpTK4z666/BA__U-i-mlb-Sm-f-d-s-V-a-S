# BA__Projekt/models/model__generic_ensemble.py

"""
This module defines the GenericEnsembleRegressor class, which ensembles
multiple instances of a base `GenericRegressor` model to improve predictive
performance and uncertainty estimation.

Classes
-------
GenericEnsembleRegressor
    An ensemble of configurable regressors supporting standard and evidential outputs.
"""

import torch
import torch.nn as nn

from models.model__generic import GenericRegressor


class GenericEnsembleRegressor(nn.Module):
    """
    Ensemble aus mehreren GenericRegressor-Modellen.
    Unterstützt evidential Regression mit Ensemble-Mittelung.

    Args:
        base_config (dict): Config für jeden Base-Regressor.
        n_models (int): Anzahl Modelle im Ensemble.
        seed (int|None): Optionaler Seed für reproduzierbare Initialisierung.

    Attribute:
        models (nn.ModuleList): Liste der Base-Regressoren.
    """

    def __init__(self, base_config: dict, n_models: int = 5, seed: int = None):
        super().__init__()
        self.models = nn.ModuleList()
        for i in range(n_models):
            if seed is not None:
                torch.manual_seed(seed + i)
            self.models.append(GenericRegressor(**base_config))

    def forward(self, x):
        """
        Forward-Pass über das Ensemble.

        Args:
            x (torch.Tensor): Input-Tensor (batch_size, input_dim).

        Returns:
            torch.Tensor oder Tuple von torch.Tensor:
                Ensemble-aggregierte Vorhersagen. Für evidential Regression
                (mu, v, alpha, beta), jeweils gemittelt über alle Modelle.
        """
        outputs = [model(x) for model in self.models]
        if isinstance(outputs[0], tuple):
            mus, vs, alphas, betas = zip(*outputs)
            mu = torch.stack(mus).mean(dim=0)
            v = torch.stack(vs).mean(dim=0)
            alpha = torch.stack(alphas).mean(dim=0)
            beta = torch.stack(betas).mean(dim=0)
            return mu, v, alpha, beta
        else:
            return torch.stack(outputs).mean(dim=0)

    def get_individual_outputs(self, x):
        """
        Gibt alle einzelnen Modelloutputs als Liste zurück,
        z.B. für diversity loss oder Analyse.

        Args:
            x (torch.Tensor): Input-Tensor.

        Returns:
            List von torch.Tensor oder Tuple[torch.Tensor]:
                Einzelvorhersagen der einzelnen Ensemble-Modelle.
        """
        return [model(x) for model in self.models]

