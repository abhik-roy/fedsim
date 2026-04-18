import json
from typing import Any

import numpy as np
from fl_core import FedAvg, FitRes, NDArrays
from strategies.utils import filter_nan_clients


class RFAStrategy(FedAvg):
    """Robust Federated Aggregation via geometric median (Pillutla et al., 2022).

    Computes the geometric median of client updates using Weiszfeld's iterative
    algorithm. The geometric median minimizes the sum of Euclidean distances to
    all input points, providing a theoretical breakdown point of up to 50%
    (tolerates fewer than half Byzantine clients).

    Unlike coordinate-wise median, RFA operates on full parameter vectors,
    preserving cross-coordinate structure of the updates. All clients contribute
    to the geometric median (no exclusion). The returned client_scores reflect
    inverse distance to the median (higher = closer to consensus).

    Attributes:
        max_iter: Maximum number of Weiszfeld iterations.
        tol: Convergence tolerance; iteration stops when the shift in the
            geometric median estimate falls below this value.
    """

    def __init__(self, max_iter: int = 100, tol: float = 1e-6, **kwargs):
        super().__init__(**kwargs)
        self.max_iter = max_iter
        self.tol = tol

    def _geometric_median(self, points: np.ndarray) -> np.ndarray:
        """Compute geometric median using Weiszfeld's algorithm.

        Args:
            points: Array of shape (n_clients, n_params) — flattened parameter vectors.

        Returns:
            Geometric median vector of shape (n_params,).
        """
        # Initialize with coordinate-wise median (robust starting point)
        median = np.median(points, axis=0)

        for _ in range(self.max_iter):
            distances = np.linalg.norm(points - median, axis=1, keepdims=True)
            # Avoid division by zero for points at/near the current estimate.
            # Using 1e-4 rather than 1e-7 prevents extreme weight imbalance
            # (1/1e-7 = 1e7) when a point coincides with the median estimate.
            distances = np.maximum(distances, 1e-4)
            weights = 1.0 / distances
            total_weight = weights.sum()
            new_median = (weights * points).sum(axis=0) / total_weight

            shift = np.linalg.norm(new_median - median)
            median = new_median
            if shift < self.tol:
                break

        return median

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[int, FitRes]],
        failures: list,
    ) -> tuple[NDArrays | None, dict[str, Any]]:
        if not results:
            return None, {}

        weights = [fit_res.parameters for _, fit_res in results]
        cids = [cid for cid, _ in results]
        weights, cids = filter_nan_clients(weights, cids)
        num_clients = len(weights)
        if num_clients == 0:
            return None, {}

        # Flatten all client parameters into vectors
        flat_params = np.array([
            np.concatenate([layer.flatten() for layer in w]) for w in weights
        ])

        # Compute geometric median
        gm = self._geometric_median(flat_params)

        # Compute per-client scores based on distance to geometric median
        final_distances = np.linalg.norm(flat_params - gm, axis=1)
        _max_d = np.max(final_distances)
        max_dist = _max_d if _max_d > 1e-10 else 1.0
        client_scores_dict = {str(cids[i]): float(1.0 - final_distances[i] / max_dist) for i in range(len(weights))}

        # Unflatten back into layer structure
        aggregated = []
        offset = 0
        for layer in weights[0]:
            size = layer.size
            aggregated.append(gm[offset:offset + size].reshape(layer.shape).astype(layer.dtype))
            offset += size

        metrics = {
            "client_scores": json.dumps(client_scores_dict),
            "included_clients": json.dumps(sorted(cids)),
            "excluded_clients": json.dumps([]),
        }
        return aggregated, metrics
