from typing import Callable, Sequence

import torch.nn as nn
from models.base_models import ShapleyNetwork
from torch import Tensor


class MLP(nn.Module):
    """
    A generic fully-connected MLP, simplified to be easily integrated into ViaShap
    without having to worry about the netwrok details.

    Args:
      input_dim: size of the last dimension of the incoming tensor
      hidden_dims: list of hidden-layer widths (e.g. [64, 128, 64])
      output_dim: final layer width
      activation: a class or factory (e.g. nn.ReLU)
      use_batchnorm: if True, insert BatchNorm1d between each hidden Linear and activation
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        output_dim: int,
        activation: Callable[[], nn.Module] = nn.ReLU,
        use_batchnorm: bool = False,
    ):
        super().__init__()
        dims = [input_dim, *hidden_dims, output_dim]  # unpack the hidden dims
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))

            if i < len(hidden_dims):
                if use_batchnorm:
                    layers.append(nn.BatchNorm1d(dims[i + 1]))
                layers.append(activation())
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        # x: (batch, input_dim) --> out: (batch, output_dim)
        return self.net(x)


class MLPShapleyNetwork(ShapleyNetwork):
    """
    A simple MLP-based implementation of a ShapleyNetwork.

    This class uses a standard MLP to compute Shapley values for each feature. The MLP takes
    the input features and produces attributions that sum to the model's prediction.

    Args:
        n_features: Number of input features
        d_out: Dimension of output for each feature attribution (default: 1)
        hidden_dims: Sequence of hidden layer dimensions (default: (64, 128, 64))
        activation: Activation function to use between layers (default: nn.ReLU)
        use_batchnorm: Whether to use batch normalization (default: True)

    Shape:
        - Input: (batch_size, n_features)
        - Output: (batch_size, n_features, d_out)
          Where each feature's d_out values sum across features to give the final prediction

    Example:
        >>> net = MLPShapleyNetwork(n_features=10, d_out=1)
        >>> x = torch.randn(32, 10)  # batch of 32 samples
        >>> attributions = net(x)  # shape: (32, 10, 1)
        >>> predictions = attributions.sum(dim=1)  # shape: (32, 1)
    """

    def __init__(
        self,
        n_features: int,
        d_out: int = 1,
        hidden_dims: Sequence[int] = (64, 128, 64),
        activation: Callable[[], nn.Module] = nn.ReLU,
        use_batchnorm: bool = True,
    ):
        super().__init__(n_features=n_features, d_out=d_out)
        self.mlp = MLP(
            input_dim=n_features,
            hidden_dims=hidden_dims,
            output_dim=n_features * d_out,
            activation=activation,
            use_batchnorm=use_batchnorm,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (batch, n_features)
        returns phi: (batch, n_features, d_out)
        """
        batch_size = x.shape[0]
        out = self.mlp(x)  # (batch, n_features*d_out)
        return out.view(batch_size, self.n_features, self.d_out)
