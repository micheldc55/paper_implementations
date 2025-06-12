from abc import ABC, abstractmethod
from typing import Tuple, Optional

import torch
from torch import Tensor

from samplers.base_sampler import FeatureSampler
from utils.pytorch_generators import init_torch_generator_from_seed


class UniformFeatureSampler(FeatureSampler):
    """
    Uniformly sample S in {1, ..., n_features} by picking each subset size k with
    probability 1/(2^n), or equivalently, each mask vector uniformly at random.
    (i.e. each feature is included with p=0.5 independently).

    Args:
        baseline: Baseline value of the coalition (occurs when sampling a 0-mask).
    """
    def __init__(self, baseline: float = 0.0):
        super().__init__(baseline=baseline)

    def sample(
        self,
        x: Tensor,
        n_coalitions: int,
        random_seed: Optional[int] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Sample features using a Uniform distribution where each feature in the batch is 
        sampled using p(x_S) = 0.5

        Args:
            x (Tensor): Tensor containing the features you want to sample.
            n_coalitions (int): Number of coalitions to draw. Equivalent to number of 
                experiments, the higher the numebr of coalitions the more robust the 
                results.
            random_seed (Optional[int], optional): Whether you want to set up a random 
                seed for reproducibility. Defaults to None.

        Returns:
            Tuple[Tensor, Tensor]: Returns two tensors with shape 
                `batch_size` * `n_coalitions` * `n_features`

                Tensor 1: Sampled features. If the features have been selected, their 
                    original values will appear. Else, the baseline will appear.

                Tensor 2: Masked feature matrix which will indicate with a `1` if the 
                    feature has been selected. Else `0`.
        """
        device = x.device
        batch_size, n_features = x.shape

        # build a local RNG if a seed is provided
        generator = init_torch_generator_from_seed(random_seed, device=device)

        masks = torch.randint(
            low=0, high=2,
            size=(batch_size, n_coalitions, n_features),
            device=device, dtype=x.dtype,
            generator=generator
        )

        x_expanded = x.unsqueeze(1).expand(-1, n_coalitions, -1).clone()  # (batch_size, n_coalitions, n_features)
        
        baseline_tensor = torch.ones_like(x_expanded) * self.baseline
        
        x_s = torch.where(masks.bool(), x_expanded, baseline_tensor)  # (batch_size, n_coalitions, n_features)
        
        x_s_flat = x_s.view(batch_size * n_coalitions, n_features)
        masks_flat = masks.view(batch_size * n_coalitions, n_features)
        return x_s_flat, masks_flat
    