import numpy as np


def _is_float(param):
    return np.issubdtype(param.dtype, np.floating)


def apply_weight_spiking(parameters: list[np.ndarray], magnitude: float = 100.0,
                         spike_fraction: float = 0.1, seed: int = 42) -> list[np.ndarray]:
    """Multiply a random subset of model weights by a large magnitude factor."""
    rng = np.random.default_rng(seed)
    spiked = []
    for param in parameters:
        if _is_float(param):
            p = param.copy()
            mask = rng.random(p.shape) < spike_fraction
            p[mask] *= magnitude
            spiked.append(p)
        else:
            spiked.append(param.copy())
    return spiked


def apply_gradient_scaling(parameters: list[np.ndarray], scale_factor: float = 10.0,
                           global_parameters: list[np.ndarray] | None = None) -> list[np.ndarray]:
    """Scale model update deltas by a constant factor.

    If global_parameters is provided, computes: global + scale_factor * (local - global).
    This is the correct gradient scaling attack that amplifies the update direction.
    Falls back to uniform parameter scaling if global_parameters is not available.
    """
    if global_parameters is None:
        raise ValueError(
            "apply_gradient_scaling requires global_parameters to compute update "
            "deltas. Without it, the attack cannot amplify gradients correctly."
        )
    if len(parameters) != len(global_parameters):
        raise ValueError(
            f"Parameter list length mismatch: local has {len(parameters)}, "
            f"global has {len(global_parameters)}"
        )
    scaled = []
    for local, glob in zip(parameters, global_parameters):
        if _is_float(local):
            delta = local - glob
            scaled.append((glob + scale_factor * delta).astype(local.dtype))
        else:
            scaled.append(local.copy())
    return scaled


def apply_byzantine_perturbation(parameters: list[np.ndarray], noise_std: float = 1.0,
                                  seed: int = 42) -> list[np.ndarray]:
    """Replace model parameters with random noise scaled to each layer's magnitude."""
    rng = np.random.default_rng(seed)
    perturbed = []
    for param in parameters:
        if _is_float(param):
            layer_scale = max(np.std(param), np.abs(param).mean(), 1e-6)
            noise = rng.normal(0, layer_scale * noise_std, size=param.shape)
            perturbed.append(noise.astype(param.dtype))
        else:
            perturbed.append(param.copy())
    return perturbed
