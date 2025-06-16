from typing import Optional, Callable

import torch
import torch.nn as nn
from torch import Tensor

from models.base_models import ShapleyNetwork


class ViaShapModel(nn.Module):
    """
    Wrapper for a ShapleyNetwork that computes predictions by summing its output.

    Args:
        shapley_network: ShapleyNetwork instance returning values of shape
            (batch_size, n_features, d_out).
        link_fn: Optional callable (nn.Module or function) to transform raw predictions.
        add_trainable_bias: If True, includes a learnable bias of shape (d_out,).
    """

    def __init__(
        self,
        shapley_network: ShapleyNetwork,
        link_fn: Optional[Callable[[Tensor], Tensor]] = None,
        add_trainable_bias: bool = False,
    ):
        super().__init__()
        self.shapley_network = shapley_network
        self.d_out = shapley_network.d_out
        self.link_fn = link_fn

        if add_trainable_bias:
            self.bias = nn.Parameter(torch.zeros(self.d_out))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: Tensor) -> Tensor:
        """
        Compute predictions by summing shapley values, then applying bias and link_fn.

        Args:
            x: Tensor of shape (batch_size, n_features).

        Returns:
            Tensor of shape (batch_size, d_out).
        """
        values = self.shapley_network(x)
        preds = values.sum(dim=1)

        if hasattr(self, 'bias') and self.bias is not None:
            preds = preds + self.bias

        if self.link_fn is not None:
            preds = self.link_fn(preds)

        return preds

    def predict(self, x: Tensor) -> Tensor:
        """
        Inference without gradient tracking.

        Args:
            x: Tensor of shape (batch_size, n_features).

        Returns:
            Tensor of shape (batch_size, d_out).
        """
        with torch.no_grad():
            return self.forward(x)

    def get_shapley_values(self, x: Tensor) -> Tensor:
        """
        Retrieve raw shapley values of shape (batch_size, n_features, d_out).
        """
        return self.shapley_network(x)

    def get_local_importance(self, x: Tensor) -> Tensor:
        """
        Flatten shapley values to (batch_size, n_features * d_out).
        """
        values = self.get_shapley_values(x)
        return values.reshape(x.size(0), -1)
