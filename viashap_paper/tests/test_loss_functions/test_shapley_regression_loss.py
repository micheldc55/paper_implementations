import pytest

import torch
import numpy as np

from models.mlp_models import MLPShapleyNetwork
from loss_functions.shapley_regression_loss import ShapleyRegressionLoss
from loss_functions.value_functions import baseline_removal_value_fn
from samplers.uniform_sampler import UniformFeatureSampler
from utils.pytorch_generators import init_torch_generator_from_seed
from via_shap.via_shap import ViaShapModel


@pytest.fixture
def uniform_sampler():
    sampler = UniformFeatureSampler(baseline=10.0)
    sampler.set_seed(101)
    return sampler

@pytest.fixture
def shapley_regression_loss(uniform_sampler):
    return ShapleyRegressionLoss(sampler=uniform_sampler, value_fn=baseline_removal_value_fn, beta=10.0)

@pytest.fixture
def input_tensor_ones():
    return torch.ones(32, 10)

@pytest.fixture
def input_tensor_zeros():
    return torch.zeros(32, 10)

@pytest.fixture
def input_tensor_pseudo_random(random_seed):
    gen = init_torch_generator_from_seed(random_seed, device='cpu')
    return torch.randn(32, 10, generator=gen)

@pytest.fixture
def input_target_tensor():
    gen = init_torch_generator_from_seed(101, device='cpu')
    return torch.randn(32, 1, generator=gen)

@pytest.fixture
def shapley_network(random_seed):
    torch.manual_seed(random_seed)
    return MLPShapleyNetwork(n_features=10, d_out=1, hidden_dims=[64, 64])

@pytest.fixture
def shapley_regression_loss(uniform_sampler):
    return ShapleyRegressionLoss(sampler=uniform_sampler, value_fn=baseline_removal_value_fn, beta=10.0)

@pytest.fixture
def shapley_regression_loss_10x_beta(uniform_sampler):
    return ShapleyRegressionLoss(sampler=uniform_sampler, value_fn=baseline_removal_value_fn, beta=100.0)

@pytest.fixture
def via_shap_model_no_bias(shapley_network):
    return ViaShapModel(shapley_network, add_trainable_bias=False)

@pytest.fixture
def via_shap_model_with_bias(shapley_network):
    return ViaShapModel(shapley_network, add_trainable_bias=True)

@pytest.fixture
def via_shap_model_with_link_fn(shapley_network):
    return ViaShapModel(shapley_network, add_trainable_bias=False, link_fn=torch.sigmoid)


def test_via_shap_model_with_bias(
        via_shap_model_with_bias, 
        shapley_regression_loss, 
        input_tensor_ones,
        input_target_tensor
    ):
    via_shap_model_with_bias.eval()  # remove sources of randomness like dropout or batchnorm

    x = input_tensor_ones
    y = input_target_tensor
    
    loss = shapley_regression_loss(via_shap_model_with_bias, x, y)
    assert loss.shape == torch.Size([])
    assert np.allclose(loss.item(), 93.952392578125)


def test_via_shap_model_no_bias(
        via_shap_model_no_bias, 
        shapley_regression_loss, 
        input_tensor_ones,
        input_target_tensor
    ):
    via_shap_model_no_bias.eval()  # remove sources of randomness like dropout or batchnorm

    x = input_tensor_ones
    y = input_target_tensor
    
    loss = shapley_regression_loss(via_shap_model_no_bias, x, y)
    assert loss.shape == torch.Size([])
    assert np.allclose(loss.item(), 93.952392578125)


def test_via_shap_model_with_link_fn(
        via_shap_model_with_link_fn, 
        shapley_regression_loss, 
        input_tensor_ones,
        input_target_tensor
    ):
    via_shap_model_with_link_fn.eval()  # remove sources of randomness like dropout or batchnorm

    x = input_tensor_ones
    y = input_target_tensor
    
    loss = shapley_regression_loss(via_shap_model_with_link_fn, x, y)
    assert np.allclose(loss.item(), 2.664985179901123)


def test_shapley_regression_loss_shape(
        via_shap_model_with_link_fn,
        shapley_regression_loss,
        input_tensor_ones,
        input_target_tensor
    ):
    x = input_tensor_ones
    y = input_target_tensor

    loss = shapley_regression_loss(via_shap_model_with_link_fn, x, y)
    assert loss.shape == torch.Size([])

def test_beta_impact(
        via_shap_model_no_bias,
        shapley_regression_loss,
        shapley_regression_loss_10x_beta,
        input_tensor_ones,
        input_target_tensor
    ):
    via_shap_model_no_bias.eval()  # remove sources of randomness like dropout or batchnorm

    x = input_tensor_ones
    y = input_target_tensor

    loss_10x_beta = shapley_regression_loss_10x_beta(via_shap_model_no_bias, x, y)
    loss = shapley_regression_loss(via_shap_model_no_bias, x, y)

    assert np.allclose(loss_10x_beta.item(), 10 * loss.item())
