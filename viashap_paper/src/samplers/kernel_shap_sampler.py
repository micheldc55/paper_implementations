from typing import Tuple, Optional

import torch
from torch import Tensor

from samplers.base_sampler import FeatureSampler
from utils.pytorch_generators import init_torch_generator_from_seed
    

class KernelShapSampler(FeatureSampler):
    """
    Implements approximate KernelSHAP sampling (vectorized grouping implementation):
    - Weights for |S|=k are proportional to 1/[C(n,k)*k*(n-k)] for k=1..n-1.
    - Sample k via multinomial, then group rows by k to vectorize subset selection.

    Args:
    n_features: total number of features n
    baseline:   float or (n_features,) tensor for masked-out positions
    """
    def __init__(self, n_features: int, baseline: float = 0.0):
        super().__init__(baseline=baseline)
        self.n = n_features
        # compute unnormalized weights in one vectorized pass
        k = torch.arange(1, self.n, dtype=torch.float32)
        log_c = (
            torch.lgamma(torch.tensor(self.n + 1.0))
            - torch.lgamma(k + 1.0)
            - torch.lgamma((self.n - k) + 1.0)
        )
        weights = torch.exp(-log_c) / (k * (self.n - k))
        self.k_weights_cpu = (weights / weights.sum()).to(torch.float32)

    def sample(
        self,
        x: Tensor,
        n_coalitions: int,
        random_seed: Optional[int] = None
    ) -> Tuple[Tensor, Tensor]:
        device = x.device
        batch_size, n_features = x.shape
        assert n_features == self.n, f"Expected {self.n} features, got {n_features}"
        
        # set up generator for reproducibility
        generator = init_torch_generator_from_seed(random_seed, device=device)

        # move weights to device and sample k indices
        k_weights = self.k_weights_cpu.to(device)
        total = batch_size * n_coalitions
        k_idx = torch.multinomial(k_weights, total, replacement=True, generator=generator) + 1
        k_flat = k_idx.view(-1)  # shape = (total,)

        # generate a random matrix for selecting top-k
        random_tensor = torch.rand((total, n_features), device=device, generator=generator)
        mask_flat = torch.zeros((total, n_features), device=device)

        # group rows by k and vectorize topk selection
        for k in torch.unique(k_flat):
            k_val = int(k.item())
            if k_val <= 0 or k_val >= n_features:
                continue
            rows = (k_flat == k).nonzero(as_tuple=True)[0]
            _, topk_idx = random_tensor[rows].topk(k_val, dim=1)
            mask_flat[rows.unsqueeze(1), topk_idx] = 1.0

        masks = mask_flat.view(batch_size, n_coalitions, n_features)
        x_exp = x.unsqueeze(1).expand(-1, n_coalitions, -1)

        if isinstance(self.baseline, torch.Tensor):
            base = self.baseline.to(device).view(1,1,n_features).expand(batch_size, n_coalitions, n_features)
        else:
            base = torch.ones_like(x_exp) * float(self.baseline)
        x_s = torch.where(masks.bool(), x_exp, base)

        return x_s.view(batch_size * n_coalitions, n_features), masks.view(batch_size * n_coalitions, n_features)
