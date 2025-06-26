import torch
from torch import Tensor
from via_shap.via_shap import ViaShapModel


def baseline_removal_value_fn(model: ViaShapModel, masked_x: Tensor) -> Tensor:
    """
    Implements baseline removal: v_x(S) = f(x_S, E[X_N\S]).
    Since masked_x is constructed already with masked values, we pass through.
    We are just predicting on the masked `x`, so it depends on the model on how
    it handles the masked values.

    A ViaShapModel will handle masked values by using some baseline value it was
    initialized with.
    """
    return model.predict(masked_x)


def marginal_expectations_value_fn(
    model: ViaShapModel,
    masked_x: Tensor,
    background_data: Tensor,
    n_mc_samples: int = 10,
) -> Tensor:
    """
    Approximates marginal expectation:
    E_xS[ f(x_S, X_{N\S}) ] using background data.

    Args:
        model: ViaShapModel
        masked_x: [batch, n_features], already masked
        background_data: [n_background, n_features]
        n_mc_samples: number of MC samples per input

    Returns:
        Tensor [batch, output_dim]
    """
    batch_size = masked_x.shape[0]
    # full_shape = masked_x.shape[1:]
    # flat_shape = (-1, *full_shape)

    masked_x_repeat = masked_x.repeat_interleave(n_mc_samples, dim=0)

    n_background = background_data.shape[0]
    bg_indices = torch.randint(0, n_background, size=(batch_size * n_mc_samples,), device=masked_x.device)
    bg_samples = background_data[bg_indices]

    mask = (masked_x != model.baseline_value).repeat_interleave(n_mc_samples, dim=0)

    mixed_x = torch.where(mask, masked_x_repeat, bg_samples)

    preds = model.predict(mixed_x).view(batch_size, n_mc_samples, -1)
    return preds.mean(dim=1)
