import torch


def init_torch_generator_from_seed(random_seed: int | None, device: str) -> torch.Generator:
    """Initialize a PyTorch Generator with an optional random seed.

    Args:
        random_seed (int | None): Random seed to initialize the generator. If None,
            returns None to use PyTorch's global RNG.
        device (str): Device to create the generator on ('cpu' or 'cuda').

    Returns:
        torch.Generator: Initialized random number generator on specified device,
            or None if no seed was provided.
    """
    generator = torch.Generator(device=device)

    if random_seed is not None:
        generator.manual_seed(random_seed)
    else:
        generator = None

    return generator