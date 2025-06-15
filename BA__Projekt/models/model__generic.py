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

import torch
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
    """
    A configurable neural network model supporting standard and evidential regression.

    Parameters
    ----------
    input_dim : int
        Number of input features.
    hidden_dims : list of int, optional
        Sizes of hidden layers, by default [64, 64].
    output_type : str, optional
        Type of output layer: either 'regression' or 'evidential', by default 'regression'.
    use_dropout : bool, optional
        Whether to apply dropout after each hidden layer, by default False.
    dropout_p : float, optional
        Dropout probability if dropout is used, by default 0.2.
    flatten_input : bool, optional
        Whether to flatten the input tensor before processing, by default False.
    use_batchnorm : bool, optional
        Whether to apply Batch Normalization between layers, by default False.
    activation_name : str, optional
        Activation function name ('relu', 'leaky_relu', 'gelu', 'elu'), by default 'relu'.
    """

    def __init__(
        self,
        input_dim,
        hidden_dims=[64, 64],
        output_type="regression",
        use_dropout=False,
        dropout_p=0.2,
        flatten_input=False,
        use_batchnorm=False,
        activation_name="relu"
    ):
        super().__init__()
        self.output_type = output_type
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
            self.output = nn.Linear(prev_dim, 1)
        elif output_type == "evidential":
            self.out_mu = nn.Linear(prev_dim, 1)
            self.out_log_v = nn.Linear(prev_dim, 1)
            self.out_log_alpha = nn.Linear(prev_dim, 1)
            self.out_log_beta = nn.Linear(prev_dim, 1)
        else:
            raise ValueError(f"Unknown output_type: {output_type}")

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dim).

        Returns
        -------
        torch.Tensor or tuple
            Output tensor(s). For evidential regression, returns a tuple:
            (mu, v, alpha, beta).
        """
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

    def predict(self, x):
        """
        Convenience method for inference with no gradient computation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor or tuple
            Model output.
        """
        self.eval()
        with torch.no_grad():
            return self.forward(x)

    def forward_regression(self, x):
        """
        Direct forward method for standard regression mode.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Regression output.
        """
        assert self.output_type == "regression"
        return self.output(self.hidden(x))

    def forward_evidential(self, x):
        """
        Direct forward method for evidential regression mode.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        tuple
            (mu, v, alpha, beta) outputs for evidential modeling.
        """
        assert self.output_type == "evidential"
        x = self.hidden(x)
        mu = self.out_mu(x)
        v = tnf.softplus(self.out_log_v(x)) + 1e-6
        alpha = tnf.softplus(self.out_log_alpha(x)) + 1.0
        beta = tnf.softplus(self.out_log_beta(x)) + 1e-6
        return mu, v, alpha, beta
