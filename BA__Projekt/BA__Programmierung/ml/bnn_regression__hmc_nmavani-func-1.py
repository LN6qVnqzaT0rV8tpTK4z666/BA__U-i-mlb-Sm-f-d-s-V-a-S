# BA__Projekt/BA__Programmierung/ml/bnn_regression_hmc__nmavani-func-1.py

import os
import numpy as np
import pyro
import pyro.distributions as dist
import torch
import torch.nn as nn

from pyro.infer import MCMC, NUTS
from pyro.nn import PyroModule, PyroSample


# Set random seed
np.random.seed(42)
pyro.set_rng_seed(42)

# Directory for saving model
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_SAVE_PATH = os.path.join(SCRIPT_DIR, "bnn_model.pth")

# Generate data
means = [-4, 0, 4]
stds = [np.sqrt(2/5), np.sqrt(0.9), np.sqrt(2/5)]
weights = [1/3, 1/3, 1/3]
n_samples = 750
components = np.random.choice([0, 1, 2], size=n_samples, p=weights)
x_obs = np.array([np.random.normal(loc=means[c], scale=stds[c]) for c in components])
epsilon = np.random.normal(0, 1, size=n_samples)
y_obs = 7 * np.sin(x_obs) + 3 * np.abs(np.cos(x_obs / 2)) * epsilon

# BNN definition
class DynamicBNN(PyroModule):
    def __init__(self, input_dim=1, output_dim=1, hidden_dims=[10, 10], prior_scale=[10.0]):
        super().__init__()
        self.activation = nn.Tanh()
        self.hidden_layers = nn.ModuleList()
        self.prior_scale = prior_scale

        self.hidden_layers.append(PyroModule[nn.Linear](input_dim, hidden_dims[0]))
        self.hidden_layers[0].weight = PyroSample(dist.Normal(0., self._scale(0)).expand([hidden_dims[0], input_dim]).to_event(2))
        self.hidden_layers[0].bias = PyroSample(dist.Normal(0., self._scale(0)).expand([hidden_dims[0]]).to_event(1))

        for i in range(1, len(hidden_dims)):
            layer = PyroModule[nn.Linear](hidden_dims[i-1], hidden_dims[i])
            layer.weight = PyroSample(dist.Normal(0., self._scale(i)).expand([hidden_dims[i], hidden_dims[i-1]]).to_event(2))
            layer.bias = PyroSample(dist.Normal(0., self._scale(i)).expand([hidden_dims[i]]).to_event(1))
            self.hidden_layers.append(layer)

        self.output_layer = PyroModule[nn.Linear](hidden_dims[-1], output_dim)
        self.output_layer.weight = PyroSample(dist.Normal(0., self._scale(-1)).expand([output_dim, hidden_dims[-1]]).to_event(2))
        self.output_layer.bias = PyroSample(dist.Normal(0., self._scale(-1)).expand([output_dim]).to_event(1))

    def _scale(self, i):
        return self.prior_scale[i] if isinstance(self.prior_scale, list) and i < len(self.prior_scale) else self.prior_scale[-1]

    def forward(self, x, y=None):
        x = x.reshape(-1, 1)
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        mu = self.output_layer(x).squeeze()
        sigma = pyro.sample("sigma", dist.Gamma(0.5, 1.0))
        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Normal(mu, sigma * sigma), obs=y)
        return mu

# Train BNN with MCMC
model_bnn = DynamicBNN(input_dim=1, output_dim=1, hidden_dims=[5], prior_scale=[5.0, 5.0])
nuts_kernel = NUTS(model_bnn, jit_compile=False)
mcmc = MCMC(nuts_kernel, num_samples=1000, warmup_steps=500)
x_train = torch.from_numpy(x_obs).float()
y_train = torch.from_numpy(y_obs).float()
mcmc.run(x_train, y_train)

# Save the posterior samples
posterior_samples = mcmc.get_samples()
torch.save(posterior_samples, MODEL_SAVE_PATH)

print(f"✅ Model saved to {MODEL_SAVE_PATH}")
