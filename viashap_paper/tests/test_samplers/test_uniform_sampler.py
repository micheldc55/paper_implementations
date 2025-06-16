import pytest
import torch

from samplers.uniform_sampler import UniformFeatureSampler


@pytest.fixture
def input_tensor():
    return torch.tensor([
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0]
    ])  # batch_size=2, n_features=4


@pytest.fixture
def uniform_sampler():
    return UniformFeatureSampler(baseline=10.0)


def test_sample_method_shape(input_tensor, uniform_sampler, random_seed):
    n_coalitions = 1000
    x_s, masks = uniform_sampler.sample(input_tensor, n_coalitions, random_seed=random_seed)

    # Check shapes match and masks are binary
    batch_size, n_features = input_tensor.shape
    assert x_s.shape == (batch_size * n_coalitions, n_features)
    assert masks.shape == (batch_size * n_coalitions, n_features)
    assert torch.all((masks == 0) | (masks == 1))


def test_sample_method_shape(input_tensor, uniform_sampler, random_seed):
    n_coalitions = 3
    x_s, masks = uniform_sampler.sample(input_tensor, n_coalitions, random_seed=random_seed)

    expected_masks = torch.tensor([
        [1., 1., 1., 0.],  # coal. 1
        [1., 1., 1., 1.],  # coal. 2 
        [1., 1., 1., 0.]   # coal. 3
    ])
    
    expected_values = torch.tensor([
        [1., 2., 3., 10.],  # coal. 1
        [1., 2., 3., 4.],  # coal. 2
        [1., 2., 3., 10.]   # coal. 3
    ])

    actual_masks = masks[:3].cpu().numpy()
    actual_values = x_s[:3].cpu().numpy()

    assert torch.allclose(torch.tensor(actual_masks), expected_masks)
    assert torch.allclose(torch.tensor(actual_values), expected_values)


def test_sample_method_repeated_n_times(input_tensor, uniform_sampler, random_seed):
    n_coalitions = 3

    results = []
    for _ in range(10):
        x_s, masks = uniform_sampler.sample(input_tensor, n_coalitions, random_seed=random_seed)
        results.append((x_s, masks))

    # Check all results are equal
    first_x_s, first_masks = results[0]
    for x_s, masks in results[1:]:
        assert torch.equal(x_s, first_x_s)
        assert torch.equal(masks, first_masks)