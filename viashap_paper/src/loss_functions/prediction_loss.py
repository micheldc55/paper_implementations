import torch.nn as nn
from torch import Tensor


class PredictionLoss(nn.Module):
    """
    Wraps any nn.Module loss (MSE, CE).

    Args:
        loss_fn: torch.nn loss (e.g. nn.CrossEntropyLoss)
    """

    def __init__(self, loss_fn: nn.Module):
        super().__init__()
        self.loss_fn = loss_fn

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        return self.loss_fn(y_pred, y_true)
    

### Provide a few common prediction losses for convenience ###


class MSEPredictionLoss(PredictionLoss):
    def __init__(self):
        super().__init__(nn.MSELoss())


class BCEPredictionLoss(PredictionLoss):
    def __init__(self):
        super().__init__(nn.BCELoss())


class CrossEntropyPredictionLoss(PredictionLoss):
    def __init__(self):
        super().__init__(nn.CrossEntropyLoss())