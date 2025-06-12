from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Callable


class ShapleyNetwork(ABC, nn.Module):
    """
    Abstract base class for any network that, given x in R^n, returns phi ∈ R^{n*d}.
    
    Users should subclass this and define `forward(x) -> torch.Tensor` where the output
    shape is (batch_size, n_features, d_out).
    """
    def __init__(self, n_features: int, d_out: int = 1):
        super().__init__()
        self.n_features = n_features
        self.d_out = d_out
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
          x: (batch_size, n_features)
        Returns:
          phi: (batch_size, n_features, d_out)
        """
        raise NotImplementedError
    