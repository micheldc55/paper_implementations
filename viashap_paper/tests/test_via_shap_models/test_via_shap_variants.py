import pytest
import torch
import torch.nn as nn
from torch import Tensor

from via_shap.variants import ViaShapSigmoid, ViaShapSoftmax
from models.base_models import ShapleyNetwork


class DummyShapleyNetwork(ShapleyNetwork):
    """
    Simple ShapleyNetwork stub: returns x unsqueezed across d_out dimension.
    """
    def __init__(self, n_features: int, d_out: int = 1):
        super().__init__(n_features, d_out)

    def forward(self, x: Tensor) -> Tensor:
        # x: (batch, n_features) -> (batch, n_features, d_out)
        return x.unsqueeze(2).expand(-1, -1, self.d_out)
    

@pytest.fixture
def input_tensor():
    # batch_size=2, n_features=3
    return torch.tensor([[1.0, 2.0, 3.0], [2.0, 1.0, 4.0]])


@pytest.fixture
def dummy_net(input_tensor):
    # n_features inferred from input, d_out=1
    _, n = input_tensor.shape
    return DummyShapleyNetwork(n_features=n, d_out=1)
    

def test_sigmoid_variant(dummy_net, input_tensor):
    model = ViaShapSigmoid(shapley_network=dummy_net)
    # sum = 6, sigmoid(6)
    out = model(input_tensor)
    assert torch.allclose(out, torch.sigmoid(torch.tensor([[6.0], [7.0]])))

    # test bias behavior
    model = ViaShapSigmoid(shapley_network=dummy_net, add_trainable_bias=True)
    model.bias.data.fill_(1.0)
    out2 = model(input_tensor)
    assert torch.allclose(out2, torch.sigmoid(torch.tensor([[7.0], [8.0]])))


def test_softmax_variant():
    # n_features=2, d_out=3
    dummy = DummyShapleyNetwork(n_features=2, d_out=3)
    model = ViaShapSoftmax(shapley_network=dummy)
    x = torch.tensor([[1.0, 2.0]])  # shape (1,2)

    # shapley values: [[[1,1,1],[2,2,2]]] -> sums [3,3,3]
    out = model(x)
    # softmax across d_out dim=1 of preds shape (1,3)
    expected = nn.functional.softmax(torch.tensor([[3.0, 3.0, 3.0]]), dim=1)
    assert torch.allclose(out, expected)

    # test with bias
    model = ViaShapSoftmax(shapley_network=dummy, add_trainable_bias=True)
    model.bias.data.copy_(torch.tensor([0.0, 1.0, 2.0]))
    out2 = model(x)
    raw = torch.tensor([[3.0, 4.0, 5.0]])
    expected2 = nn.functional.softmax(raw, dim=1)
    assert torch.allclose(out2, expected2)

    importance = model.get_local_importance(x)
    # shape (1, 2*3=6)
    assert importance.shape == (1, 6)
    # flattened values = [1,1,1,2,2,2]
    assert torch.allclose(importance, torch.tensor([[1,1,1,2,2,2]], dtype=torch.float))
