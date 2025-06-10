from typing import Tuple, Optional

import torch
from torch import Tensor

from samplers.base_sampler import FeatureSampler
    

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
        B, N = x.shape
        assert N == self.n, f"Expected {self.n} features, got {N}"
        
        # set up generator for reproducibility
        if random_seed is not None:
            gen = torch.Generator(device=device)
            gen.manual_seed(random_seed)
        else:
            gen = None

        # move weights to device and sample k indices
        k_weights = self.k_weights_cpu.to(device)
        total = B * n_coalitions
        k_idx = torch.multinomial(k_weights, total, replacement=True, generator=gen) + 1
        k_flat = k_idx.view(-1)  # shape = (total,)

        # generate a random matrix for selecting top-k
        R = torch.rand((total, N), device=device, generator=gen)
        mask_flat = torch.zeros((total, N), device=device)

        # group rows by k and vectorize topk selection
        for k in torch.unique(k_flat):
            k_val = int(k.item())
            if k_val <= 0 or k_val >= N:
                continue
            rows = (k_flat == k).nonzero(as_tuple=True)[0]
            # select top k_val indices along each row
            topk_vals, topk_idx = R[rows].topk(k_val, dim=1)
            # scatter 1s into mask_flat
            mask_flat[rows.unsqueeze(1), topk_idx] = 1.0

        # reshape masks and build masked inputs
        masks = mask_flat.view(B, n_coalitions, N)
        x_exp = x.unsqueeze(1).expand(-1, n_coalitions, -1)
        if isinstance(self.baseline, torch.Tensor):
            base = self.baseline.to(device).view(1,1,N).expand(B, n_coalitions, N)
        else:
            base = torch.ones_like(x_exp) * float(self.baseline)
        x_s = torch.where(masks.bool(), x_exp, base)

        # flatten and return
        return x_s.view(B * n_coalitions, N), masks.view(B * n_coalitions, N)

# --- Test to verify equivalence with original implementation ---
if __name__ == '__main__':
    seed = 42

    n, C = 8, 5
    x = torch.arange(0, n, dtype=torch.float32).unsqueeze(0)
    seed = 42

    orig = KernelShapSampler(n_features=n, baseline=0.0)
    vectorized = NewKernelShapSampler(n_features=n, baseline=0.0)

    x1, m1 = orig.sample(x, C, random_seed=seed)
    x2, m2 = vectorized.sample(x, C, random_seed=seed)

    print(m1)
    print(m2)

    assert torch.equal(m1, m2), 'Masks differ between implementations'
    assert torch.equal(x1, x2), 'Masked inputs differ between implementations'
    print('Vectorized sampler matches original exactly!')