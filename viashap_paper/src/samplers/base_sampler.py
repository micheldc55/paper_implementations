from abc import ABC, abstractmethod
from typing import Tuple, Optional

from torch import Tensor


class FeatureSampler(ABC):
    """
    Abstract base class for sampling feature subsets (coalitions) S.
    A sampler must return:
      - x_S: Tensor of same shape as x, where features outside S are replaced by baseline
      - m_S: Binary mask tensor (0/1), shape (batch_size, n_features)
    
    Users can override .sample() to define arbitrary p(S).
    """
    def __init__(self, baseline: float = 0.0):
        """
        baseline: the value to use for features that are 'masked out'. 
                  By default, we zero them, but you could also pass a vector 
                  of means, etc.
        """
        self.baseline = baseline
    
    @abstractmethod
    def sample(
        self, 
        x: Tensor, 
        n_coalitions: int,
        random_seed: Optional[int] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Sample a set of coalitions of features for each element in the batch.
        
        Args:
          x: (batch_size, n_features) original inputs.
          n_coalitions: how many subsets S to sample per example.
        
        Returns:
          x_S: (batch_size * n_coalitions, n_features) where features outside S are replaced by baseline
          m_S: (batch_size * n_coalitions, n_features) binary mask (1 if feature is present, 0 otherwise)
          
        Note: The order is interleaved: for each example i in [0..batch_size-1], 
              we produce n_coalitions masked versions. So the first n_coalitions rows 
              correspond to x[0], next n_coalitions to x[1], etc.
        """
        raise NotImplementedError