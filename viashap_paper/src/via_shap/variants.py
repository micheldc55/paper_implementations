import torch.nn as nn

from models.base_models import ShapleyNetwork
from via_shap.via_shap import ViaShapModel


class ViaShapSigmoid(ViaShapModel):
    """
    ViaSHAP model with a sigmoid link function (suitable for binary classification/regression to [0,1]).

    Args:
        shapley_network: Underlying ShapleyNetwork instance.
        add_trainable_bias: If True, includes a learnable bias term.
    """

    def __init__(self, shapley_network: ShapleyNetwork, add_trainable_bias: bool = False):
        super().__init__(
            shapley_network=shapley_network,
            link_fn=nn.Sigmoid(),
            add_trainable_bias=add_trainable_bias,
        )


class ViaShapSoftmax(ViaShapModel):
    """
    ViaSHAP model with softmax over output dimension (suitable for multi-class classification).

    Args:
        shapley_network: Underlying ShapleyNetwork instance with d_out = num_classes.
        add_trainable_bias: If True, includes a learnable bias term per class.
    """

    def __init__(self, shapley_network: ShapleyNetwork, add_trainable_bias: bool = False):
        super().__init__(
            shapley_network=shapley_network,
            link_fn=nn.Softmax(dim=1),  # apply softmax over class dimension
            add_trainable_bias=add_trainable_bias,
        )
