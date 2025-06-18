# BA__Projekt/BA__Programmierung/ml/metrics/metrics.py

from typing import Any
import torch
import inspect


class Metric:
    """
    Wraps a metric function and provides optional accumulation for batch-wise evaluation.

    Parameters
    ----------
    fn : Callable
        The metric function. Must accept predicted and true values (or other model outputs).
    name : str, optional
        Optional custom name for the metric. Defaults to the function's __name__.
    accumulate : bool, default=True
        Whether the metric should accumulate predictions across multiple batches.
    arg_names : list[str], optional
        Names of the arguments the metric expects.
    """

    def __init__(self, fn, name: str, accumulate: bool = True, arg_names: list[str] = None):

        fn_sig = inspect.signature(fn)
        if arg_names and len(arg_names) != len(fn_sig.parameters):
            raise ValueError(f"Mismatch in arg_names vs. function args for {name}")

        self.fn = fn
        self.name = name or fn.__name__
        self.accumulate = accumulate
        self.arg_names = arg_names or []

        self.preds = []

        # Optional: precompute the signature for better error messages / performance
        self.signature = inspect.signature(fn)

    def __call__(self, **kwargs):
        if not self.accumulate:
            print(f"{self.name:<25}: (batch-only metric, skipping accumulation)")
            return

        missing = [k for k in self.arg_names if k not in kwargs]
        if missing:
            print(f"{self.name:<25}: ⏭ Skipped - Missing inputs: {missing}")
            return

        self._accumulate(**kwargs)

    def _accumulate(self, **kwargs) -> None:
        try:
            data_tuple = tuple(
                v.detach().cpu() if isinstance(v, torch.Tensor) else v
                for k, v in kwargs.items() if k in self.arg_names
            )
            self.preds.append(data_tuple)
        except Exception as e:
            print(f"❌ {self.name}: Accumulate failed: {e}")

    def compute(self) -> Any:
        if not self.accumulate:
            raise ValueError(f"Metric '{self.name}' does not support accumulation.")

        if isinstance(self.preds[0], tuple):
            pred_components = list(zip(*self.preds))
            pred_components = [torch.cat(p) for p in pred_components]
        else:
            pred_components = [torch.cat(self.preds)]

        # Unpack tuples if multiple args were stored
        if isinstance(self.preds[0], tuple):
            pred_components = list(zip(*self.preds))  # transpose
            pred_components = [torch.cat(p) for p in pred_components]
        else:
            pred_components = [torch.cat(self.preds)]

        if len(pred_components) != len(self.arg_names):
            raise ValueError(
                f"Mismatch in number of inputs for '{self.name}': "
                f"expected {len(self.arg_names)}, got {len(pred_components)}"
            )

        input_dict = dict(zip(self.arg_names, pred_components))
        print(f"🧪 {self.name:<25} | Inputs: {list(input_dict.keys())}")
        return self.fn(**input_dict)

    def reset(self) -> None:
        """Clear stored predictions and targets."""
        self.preds.clear()
