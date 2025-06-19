# BA__Projekt/models/generic_model.py

"""
This module provides the `GenericRegressor` class for building flexible
neural network regressors, including support for evidential uncertainty modeling.

Functions
---------
get_activation(name)
    Returns a PyTorch activation layer by name.

Classes
-------
GenericRegressor
    A modular MLP supporting both standard and evidential regression.
"""

import torch.nn as nn
import torch.nn.functional as tnf


def get_activation(name):
    """
    Returns the activation function given its name.

    Parameters
    ----------
    name : str
        Name of the activation function. One of ['relu', 'leaky_relu', 'gelu', 'elu'].

    Returns
    -------
    nn.Module
        PyTorch activation function module.
    """
    activations = {
        'relu': nn.ReLU(),
        'leaky_relu': nn.LeakyReLU(0.01),
        'gelu': nn.GELU(),
        'elu': nn.ELU(),
    }
    return activations.get(name, nn.ReLU())


class GenericRegressor(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dims=[64, 64],
        output_type="regression",
        output_dim=1,  # <-- Add this
        use_dropout=False,
        dropout_p=0.2,
        flatten_input=False,
        use_batchnorm=False,
        activation_name="relu"
    ):
        super().__init__()
        self.output_type = output_type
        self.output_dim = output_dim  # <-- Store output_dim
        self.flatten_input = flatten_input

        activation = get_activation(activation_name)
        layers = []
        prev_dim = input_dim

        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(activation)
            if use_dropout:
                layers.append(nn.Dropout(dropout_p))
            prev_dim = h

        self.hidden = nn.Sequential(*layers)

        if output_type == "regression":
            self.output = nn.Linear(prev_dim, output_dim)

        elif output_type == "evidential":
            self.out_mu = nn.Linear(prev_dim, output_dim)
            self.out_log_v = nn.Linear(prev_dim, output_dim)
            self.out_log_alpha = nn.Linear(prev_dim, output_dim)
            self.out_log_beta = nn.Linear(prev_dim, output_dim)

        else:
            raise ValueError(f"Unknown output_type: {output_type}")

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if self.flatten_input:
            x = x.view(x.size(0), -1)
        x = self.hidden(x)

        if self.output_type == "regression":
            return self.output(x)

        elif self.output_type == "evidential":
            mu = self.out_mu(x)
            v = tnf.softplus(self.out_log_v(x)) + 1e-6
            alpha = tnf.softplus(self.out_log_alpha(x)) + 1.0
            beta = tnf.softplus(self.out_log_beta(x)) + 1e-6
            return mu, v, alpha, beta
