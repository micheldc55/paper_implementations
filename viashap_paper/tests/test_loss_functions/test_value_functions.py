import torch
import pytest

from via_shap.via_shap import ViaShapModel
from models.mlp_models import MLPShapleyNetwork
from loss_functions.value_functions import (
    baseline_removal_value_fn,
    marginal_expectations_value_fn,
)

@pytest.fixture
def simple_model() -> ViaShapModel:
    torch.manual_seed(0)
    net = MLPShapleyNetwork(n_features=4, hidden_dims=[8], d_out=1)
    model = ViaShapModel(net)
    model.baseline_value = 0.0
    return model

@pytest.fixture
def model_with_bias() -> ViaShapModel:
    torch.manual_seed(0)
    net = MLPShapleyNetwork(n_features=4, hidden_dims=[8], d_out=1)
    model = ViaShapModel(net, add_trainable_bias=True)
    model.baseline_value = 0.0
    return model

@pytest.fixture
def input_tensor() -> torch.Tensor:
    return torch.tensor([
        [1.0, 2.0, 3.0, 4.0],
        [4.0, 3.0, 2.0, 1.0]
    ])

@pytest.fixture
def masked_tensor(input_tensor) -> torch.Tensor:
    masked = input_tensor.clone()
    masked[:, [1, 3]] = 0.0
    return masked

@pytest.fixture
def background_data() -> torch.Tensor:
    return torch.tensor([
        [0.1, 0.1, 0.1, 0.1],
        [0.2, 0.2, 0.2, 0.2],
        [0.3, 0.3, 0.3, 0.3],
        [0.4, 0.4, 0.4, 0.4],
    ])

@pytest.fixture
def image_tensor() -> torch.Tensor:
    return torch.tensor([
        [[[1.0, 0.0], [2.0, 0.0]]],
        [[[0.0, 1.0], [0.0, 2.0]]]
    ])  # shape (2, 1, 2, 2)

@pytest.fixture
def image_background() -> torch.Tensor:
    return torch.stack([
        torch.ones(1, 2, 2) * 0.1,
        torch.ones(1, 2, 2) * 0.2,
        torch.ones(1, 2, 2) * 0.3,
        torch.ones(1, 2, 2) * 0.4,
    ])  # shape (4, 1, 2, 2)

def test_baseline_removal_value_fn(simple_model, masked_tensor):
    out = baseline_removal_value_fn(simple_model, masked_tensor)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (2, 1)
    assert out.dtype == torch.float32

def test_marginal_expectations_value_fn_shape_and_determinism(simple_model, masked_tensor, background_data, random_seed):
    torch.manual_seed(random_seed)
    out1 = marginal_expectations_value_fn(simple_model, masked_tensor, background_data, n_mc_samples=8)

    torch.manual_seed(random_seed)
    out2 = marginal_expectations_value_fn(simple_model, masked_tensor, background_data, n_mc_samples=8)

    assert isinstance(out1, torch.Tensor)
    assert out1.shape == (2, 1)
    assert torch.allclose(out1, out2, atol=1e-6)

def test_marginal_expectation_deviation_from_baseline(simple_model, masked_tensor, background_data, random_seed):
    torch.manual_seed(random_seed)
    baseline_out = baseline_removal_value_fn(simple_model, masked_tensor)

    torch.manual_seed(random_seed)
    marginal_out = marginal_expectations_value_fn(simple_model, masked_tensor, background_data, n_mc_samples=16)

    diffs = (marginal_out - baseline_out).abs()
    assert torch.any(diffs > 1e-3)

def test_model_with_trainable_bias(model_with_bias, masked_tensor, background_data):
    out = marginal_expectations_value_fn(model_with_bias, masked_tensor, background_data, n_mc_samples=8)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (2, 1)
    assert not torch.isnan(out).any()

def test_image_input_supported(image_tensor, image_background):
    class IdentityModel(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x
        
        @property
        def d_out(self) -> int:
            return 1
        
    
    class DummyImageModel(ViaShapModel):
        def __init__(self):
            super().__init__(IdentityModel())
            self.baseline_value = 0.0

        def predict(self, x: torch.Tensor) -> torch.Tensor:
            return x.mean(dim=[1, 2, 3], keepdim=True)

    model = DummyImageModel()
    out = marginal_expectations_value_fn(model, image_tensor, image_background, n_mc_samples=4)
    assert out.shape == (2, 1)
    assert not torch.isnan(out).any()