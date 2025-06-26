import pytest
import torch
import torch.nn as nn
from torch import Tensor

from via_shap.via_shap import ViaShapModel
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


def test_forward_no_bias_no_link(dummy_net, input_tensor):
    model = ViaShapModel(shapley_network=dummy_net)
    out = model(input_tensor)
    # sum of [1,2,3] = 6 / sum of [2,1,4] = 7
    assert out.shape == (2, 1)
    assert torch.allclose(out, torch.tensor([[6.0], [7.0]]))


def test_forward_with_bias(dummy_net, input_tensor):
    model = ViaShapModel(shapley_network=dummy_net, add_trainable_bias=True)
    # default bias is zero
    out = model(input_tensor)
    assert torch.allclose(out, torch.tensor([[6.0], [7.0]]))

    # set bias to 2.5
    model.bias.data.fill_(2.5)
    out2 = model(input_tensor)
    assert torch.allclose(out2, torch.tensor([[8.5], [9.5]])) # 6.0 + 2.5 / 7.0 + 2.5


def test_forward_with_link(dummy_net, input_tensor):
    sigmoid = nn.Sigmoid()
    model = ViaShapModel(shapley_network=dummy_net, link_fn=sigmoid)
    out = model(input_tensor)
    expected = torch.sigmoid(torch.tensor([[6.0], [7.0]]))
    assert torch.allclose(out, expected)


def test_predict_no_grad(dummy_net, input_tensor):
    model = ViaShapModel(shapley_network=dummy_net)
    pred = model.predict(input_tensor)
    assert not pred.requires_grad
    # matches forward
    assert torch.allclose(pred, model(input_tensor))


def test_get_shapley_and_importance(dummy_net, input_tensor):
    model = ViaShapModel(shapley_network=dummy_net)
    shap_vals = model.get_shapley_values(input_tensor)
    # shape (2,3,1), values [[1,2,3]]
    assert shap_vals.shape == (2, 3, 1)
    assert torch.allclose(shap_vals, input_tensor.unsqueeze(2))

    importance = model.get_local_importance(input_tensor)
    # flattened to (1,3)
    assert importance.shape == (2, 3)
    assert torch.allclose(importance, input_tensor)