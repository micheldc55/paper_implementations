import torch
import torch.nn as nn
from torch import Tensor
from typing import Callable
from samplers.base_sampler import FeatureSampler
from via_shap.via_shap import ViaShapModel


class ShapleyRegressionLoss(nn.Module):
    """
    Weighted regression of Shapley values via coalition sampling (or any sampling scheme provided).
    
    Implements the Shapley Regression loss, uses Coalition Sampling or any sampling method you select 
    in order to learn the Shapley values. This loss function attempts to do so by sampling the values 
    from the input matrix and applying the 1^T transformation to a masked phi(x) and forcing it to 
    be closer to the ViaShap(x_S).
    
    In the original ViaShap regression paper, this is implemented as Eq (6). 

    Args:
        value_fn: function (model, x_masked, mask) -> value estimate
        sampler: FeatureSampler instance providing (masked_x, masks)
        beta: scaling factor for shapley loss
        relaxed: if True, learns bias delta (Appendix G in original paper)
    """

    def __init__(
        self,
        value_fn: Callable[[ViaShapModel, Tensor], Tensor],
        sampler: FeatureSampler,
        beta: float = 1.0,
    ):
        super().__init__()
        self.value_fn = value_fn
        self.sampler = sampler
        self.beta = beta

    def forward(self, model: ViaShapModel, x: Tensor, y: Tensor) -> Tensor:
        batch_size = x.shape[0]
        d_out = y.shape[-1]

        masked_x, masks = self.sampler.sample(x, n_coalitions=batch_size)
        n_coalitions = masked_x.shape[0] // batch_size  # sampler returns (batch_size * n_coalitions, n_features)

        value_est = self.value_fn(model, masked_x) # viashap(x_S)
        shapley_values = model.get_shapley_values(x)  # phi_via(x) --> pure x / no sampling

        masks = masks.view(batch_size, n_coalitions, -1)

        shapley_values_exp = shapley_values.unsqueeze(1).expand(-1, n_coalitions, -1, -1)
        shapley_sum = torch.einsum("bkn, bknf -> bkf", masks, shapley_values_exp)  # 1^T * phi_via(x)

        pred_baseline = model.predict(torch.zeros_like(x))  # viashap(empty_set) / shape: (batch_size, d_out)

        value_est = value_est.view(batch_size, n_coalitions, d_out)

        loss_shapley = (
            (value_est - pred_baseline.unsqueeze(1) - shapley_sum).pow(2)
        ).mean()

        return self.beta * loss_shapley