# BA__Projekt/BA__Programmierung/ml/metrics/metrics.py

from collections.abc import Callable
from typing import Optional, Any

import torch


class Metric:
    """
    Wraps a metric function and provides optional accumulation for batch-wise evaluation.

    Parameters
    ----------
    fn : Callable
        The metric function. Must accept predicted and true values (or other model outputs).
    name : str, optional
        Optional custom name for the metric. Defaults to the function's __name__.
    accumulate : bool, default=False
        Whether the metric should accumulate predictions across multiple batches.
    """

    def __init__(self, fn: Callable, name: Optional[str] = None, accumulate: bool = False):
        self.fn = fn
        self.name = name or fn.__name__
        self.accumulate = accumulate
        self.reset()

    def __call__(self, *args, **kwargs) -> Optional[Any]:
        """
        Invoke the metric function or store values for accumulation.

        Parameters
        ----------
        *args : Any
            Typically includes predicted and true values or model outputs.
        **kwargs : Any
            Additional keyword arguments passed to the metric function.

        Returns
        -------
        Any or None
            Returns the result of the metric function if `accumulate=False`,
            otherwise stores inputs and returns None.
        """
        if self.accumulate:
            self._accumulate(*args)
        else:
            return self.fn(*args, **kwargs)

    def _accumulate(self, y_pred, y_true):
        """
        Store batch predictions and targets for later evaluation.

        Parameters
        ----------
        y_pred : torch.Tensor
            Predicted outputs.
        y_true : torch.Tensor
            Ground-truth labels.
        """
        self.preds.append(y_pred.detach().cpu())
        self.trues.append(y_true.detach().cpu())

    def compute(self) -> Any:
        """
        Compute the metric using accumulated values.

        Returns
        -------
        Any
            The result of the metric function using concatenated predictions and targets.

        Raises
        ------
        ValueError
            If accumulation is not enabled.
        """
        if not self.accumulate:
            raise ValueError(f"Metric '{self.name}' does not support accumulation.")
        preds = torch.cat(self.preds)
        trues = torch.cat(self.trues)
        return self.fn(preds, trues)

    def reset(self):
        """
        Reset the internal buffers for a new accumulation cycle.
        """
        self.preds = []
        self.trues = []
