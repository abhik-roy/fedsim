"""Shared utilities for aggregation strategies."""
import numpy as np


def filter_nan_clients(
    weights: list[list[np.ndarray]],
    cids: list[int],
) -> tuple[list[list[np.ndarray]], list[int]]:
    """Remove clients whose parameters contain NaN or Inf.

    Returns (clean_weights, clean_cids) with bad clients filtered out.
    """
    clean_weights = []
    clean_cids = []
    for w, cid in zip(weights, cids):
        has_bad = False
        for layer in w:
            if not np.all(np.isfinite(layer)):
                has_bad = True
                break
        if not has_bad:
            clean_weights.append(w)
            clean_cids.append(cid)
    return clean_weights, clean_cids
