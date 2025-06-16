# tests/test_mlp_and_shapleynetwork.py
import pytest
import torch
import torch.nn as nn

from models.mlp_models import MLP, MLPShapleyNetwork
from models.base_models import ShapleyNetwork


def test_shapley_network_is_abstract():
    # ShapleyNetwork cannot be instantiated
    with pytest.raises(TypeError):
        ShapleyNetwork(n_features=3, d_out=1)


@pytest.mark.parametrize("hidden_dims,use_batchnorm", [
    ([], False),
    ([10, 20], False),
    ([5, 5, 5], True),
])
def test_mlp_shapes_and_modules(hidden_dims, use_batchnorm):
    batch, in_dim, out_dim = 4, 3, 7
    mlp = MLP(
        input_dim=in_dim,
        hidden_dims=hidden_dims,
        output_dim=out_dim,
        activation=nn.LeakyReLU,
        use_batchnorm=use_batchnorm,
    )
    x = torch.randn(batch, in_dim)
    y = mlp(x)
    assert y.shape == (batch, out_dim)

    has_bn = any(isinstance(m, nn.BatchNorm1d) for m in mlp.net)
    assert has_bn is use_batchnorm

    num_acts_expected = len(hidden_dims)
    num_acts = sum(isinstance(m, nn.LeakyReLU) for m in mlp.net)
    assert num_acts == num_acts_expected


def test_mlp_determinism(random_seed):
    torch.manual_seed(random_seed)
    m1 = MLP(4, [8, 2], 2)
    o1 = m1(torch.randn(2, 4))
    torch.manual_seed(random_seed)
    m2 = MLP(4, [8, 2], 2)
    o2 = m2(torch.randn(2, 4))
    assert torch.allclose(o1, o2)


@pytest.mark.parametrize("d_out", [1, 3, 5])
@pytest.mark.parametrize("hidden_dims", [[], [16, 8]])
def test_mlpshapley_network_shapes_and_forward(hidden_dims, d_out):
    batch, n_features = 6, 4
    net = MLPShapleyNetwork(
        n_features=n_features,
        d_out=d_out,
        hidden_dims=hidden_dims,
        activation=nn.ReLU,
        use_batchnorm=False,
    )

    x = torch.randn(batch, n_features)
    phi = net(x)
    assert phi.shape == (batch, n_features, d_out)

    flat = net.mlp(x)  # (batch, n_features * d_out)
    reconstructed = flat.view(batch, n_features, d_out)
    assert torch.allclose(phi, reconstructed)


def test_mlpshapley_network_sum_prediction():
    batch, n_features, d_out = 3, 5, 2
    net = MLPShapleyNetwork(n_features, d_out, hidden_dims=[20], use_batchnorm=False)
    x = torch.randn(batch, n_features)
    phi = net(x)
    
    pred_via_phi = phi.sum(dim=1)   # (batch, d_out)
    
    flat = net.mlp(x).view(batch, n_features, d_out)
    assert torch.allclose(pred_via_phi, flat.sum(dim=1))


def test_invalid_forward_signature():
    net = MLPShapleyNetwork(n_features=2, d_out=1, hidden_dims=[])
    with pytest.raises(RuntimeError):
        net(torch.randn(2, 3))