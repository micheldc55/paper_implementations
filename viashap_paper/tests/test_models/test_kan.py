import pytest
import torch

from models.kan_models import KANLinear, KAN, KANShapleyNetwork


def test_KANLinear_forward_shape_and_dtype():
    in_f, out_f = 4, 3
    layer = KANLinear(in_features=in_f, out_features=out_f, grid_size=5, spline_order=2)
    x = torch.randn(2, in_f)
    y = layer(x)
    assert isinstance(y, torch.Tensor), "Output should be a tensor"
    assert y.shape == (2, out_f), f"Expected shape (2, {out_f}), but got {y.shape}"


def test_KANLinear_b_spline_basis_shape():
    in_f, grid_size, spline_order = 3, 4, 2
    layer = KANLinear(in_features=in_f, out_features=2, grid_size=grid_size, spline_order=spline_order)
    x = torch.randn(5, in_f)
    bases = layer.b_spline_basis(x)
    expected_dim = grid_size + spline_order
    assert bases.shape == (5, in_f, expected_dim), (
        f"Expected basis shape (5, {in_f}, {expected_dim}), but got {bases.shape}"
    )
    # Ensure basis values are between 0 and 1
    assert bases.min() >= 0 and bases.max() <= 1, "Basis values should lie in [0,1]"


def test_KANLinear_backward_gradient_flow():
    layer = KANLinear(in_features=3, out_features=2)
    x = torch.randn(4, 3, requires_grad=True)
    y = layer(x)
    loss = y.sum()
    loss.backward()
    assert layer.base_weight.grad is not None, "base_weight should receive a gradient"
    assert layer.spline_weight.grad is not None, "spline_weight should receive a gradient"


def test_KAN_multilayer_shape():
    layers_hidden = [4, 6, 3]
    kan = KAN(layers_hidden=layers_hidden, grid_size=5, spline_order=3)
    x = torch.randn(2, 4)
    y = kan(x)
    assert y.shape == (2, 3), f"Expected shape (2, 3), got {y.shape}"


def test_KANShapleyNetwork_shape_and_equivalence():
    n_features, d_out = 3, 1
    # No hidden layers => direct mapping
    shap_net = KANShapleyNetwork(
        n_features=n_features,
        d_out=d_out,
        hidden_dims=(),
        grid_size=5,
        spline_order=3,
    )
    x = torch.randn(2, n_features)
    phi = shap_net(x)
    assert phi.shape == (2, n_features, d_out), (
        f"Expected Shapley output shape (2, {n_features}, {d_out}), got {phi.shape}"
    )
    # Flatten contributions should match underlying KAN output
    flat = phi.view(2, n_features * d_out)
    kan = shap_net.kan  # underlying KAN model
    direct = kan(x)
    assert torch.allclose(flat, direct, atol=1e-6), "Flattened Shapley output should equal KAN output"
