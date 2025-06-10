import pytest
import torch

from samplers.kernel_shap_sampler import KernelShapSampler


@pytest.fixture
def input_tensor():
    return torch.tensor([
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0]
    ])


@pytest.fixture
def kernel_shap_sampler(input_tensor):
    n_features = input_tensor.shape[1]
    return KernelShapSampler(n_features=n_features, baseline=10.0)


def test_sample_method_shape(input_tensor, kernel_shap_sampler, random_seed):
    n_coalitions = 1000
    x_s, masks = kernel_shap_sampler.sample(input_tensor, n_coalitions, random_seed=random_seed)

    # Check shapes match and masks are binary
    batch_size, n_features = input_tensor.shape
    assert x_s.shape == (batch_size * n_coalitions, n_features)
    assert masks.shape == (batch_size * n_coalitions, n_features)
    assert torch.all((masks == 0) | (masks == 1))


def test_sample_method_shape(input_tensor, kernel_shap_sampler, random_seed):
    n_coalitions = 5

    x1, m1 = kernel_shap_sampler.sample(input_tensor, n_coalitions=n_coalitions, random_seed=random_seed)

    x1_expected = torch.tensor([
        [ 1.,  2., 10., 10.],
        [ 1.,  2.,  3., 10.],
        [10., 10., 10.,  4.],
        [10., 10.,  3., 10.],
        [ 1., 10.,  3., 10.],
        [10.,  6.,  7.,  8.],
        [ 5.,  6., 10.,  8.],
        [ 5., 10., 10., 10.],
        [ 5.,  6.,  7., 10.],
        [10.,  6.,  7.,  8.]
    ])
    m1_expected = torch.tensor([
        [1., 1., 0., 0.],
        [1., 1., 1., 0.],
        [0., 0., 0., 1.],
        [0., 0., 1., 0.],
        [1., 0., 1., 0.],
        [0., 1., 1., 1.],
        [1., 1., 0., 1.],
        [1., 0., 0., 0.],
        [1., 1., 1., 0.],
        [0., 1., 1., 1.]
    ])

    assert torch.allclose(x1, x1_expected)
    assert torch.allclose(m1, m1_expected)


def test_sample_method_repeated_n_times(input_tensor, kernel_shap_sampler, random_seed):
    n_coalitions = 3

    results = []
    for _ in range(10):
        x_s, masks = kernel_shap_sampler.sample(input_tensor, n_coalitions, random_seed=random_seed)
        results.append((x_s, masks))

    # Check all results are equal
    first_x_s, first_masks = results[0]
    for x_s, masks in results[1:]:
        assert torch.equal(x_s, first_x_s)
        assert torch.equal(masks, first_masks)