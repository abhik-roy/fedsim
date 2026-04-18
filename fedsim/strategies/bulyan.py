import json
from typing import Any

import numpy as np
from fl_core import FedAvg, FitRes, NDArrays
from strategies.utils import filter_nan_clients


class BulyanStrategy(FedAvg):
    """Bulyan aggregation strategy (El Mhamdi et al., 2018).

    Two-stage meta-aggregation that combines Krum selection with trimmed mean:
    1. Selection: Iteratively apply Krum to select n - 2f candidate updates.
    2. Aggregation: Apply coordinate-wise trimmed mean to the selected candidates.

    Requires n >= 4f + 3 for full Byzantine resilience, providing stronger
    theoretical guarantees than Krum alone at the cost of higher n requirements.

    Attributes:
        num_malicious: Number of expected Byzantine (malicious) clients f.
    """

    def __init__(self, num_malicious: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.num_malicious = num_malicious

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
        f = self.num_malicious

        # Validate Bulyan invariant: n >= 4f + 3
        if f > 0 and num_clients < 4 * f + 3:
            raise ValueError(
                f"Bulyan requires n >= 4f + 3 (have n={num_clients}, f={f}). "
                f"Either increase num_clients to {4 * f + 3} or reduce "
                f"malicious_fraction, or use a different strategy."
            )

        # Stage 1: Krum-based iterative selection
        # Select theta = n - 2f candidates (or as many as safely possible)
        theta = max(1, num_clients - 2 * f)
        theta = min(theta, num_clients)  # safety clamp

        flat_weights = np.array([
            np.concatenate([layer.flatten() for layer in w]) for w in weights
        ])

        remaining = list(range(num_clients))
        selected_indices = []

        for _ in range(theta):
            if len(remaining) <= 1:
                selected_indices.extend(remaining)
                break

            # Compute pairwise distances among remaining clients
            rem_flat = flat_weights[remaining]
            sq_norms = np.sum(rem_flat ** 2, axis=1)
            distances = sq_norms[:, None] + sq_norms[None, :] - 2 * rem_flat @ rem_flat.T
            np.maximum(distances, 0, out=distances)

            # Krum score: sum of distances to nearest (n_rem - f - 2) neighbors
            n_rem = len(remaining)
            neighbors = max(1, n_rem - f - 2)
            # Set diagonal to inf so self-distance is never among the smallest
            np.fill_diagonal(distances, np.inf)
            k = min(neighbors, n_rem - 1)
            partitioned = np.partition(distances, k, axis=1)[:, :k]
            scores = partitioned.sum(axis=1)

            # Select best (lowest Krum score)
            best_local = int(np.argmin(scores))
            selected_indices.append(remaining[best_local])
            remaining.pop(best_local)

        # Stage 2: Coordinate-wise trimmed mean on selected candidates
        selected_weights = [weights[i] for i in selected_indices]
        n_sel = len(selected_weights)
        # Trim f values from each end (or as many as possible)
        # When f=0 (no malicious clients), trim=0 is correct (plain average)
        trim = min(f, (n_sel - 1) // 2)

        aggregated = []
        for layer_idx in range(len(selected_weights[0])):
            layer_stack = np.stack([w[layer_idx] for w in selected_weights], axis=0)
            if trim > 0 and n_sel > 2 * trim:
                sorted_stack = np.sort(layer_stack, axis=0)
                trimmed = sorted_stack[trim:n_sel - trim]
            else:
                trimmed = layer_stack
            aggregated.append(np.mean(trimmed, axis=0).astype(selected_weights[0][layer_idx].dtype))

        all_cids = set(cids)
        included_cids = {cids[idx] for idx in selected_indices}
        excluded_cids = all_cids - included_cids

        # Compute client scores for monitoring (distance to aggregated result)
        _flat_all = np.array([np.concatenate([l.flatten() for l in w]) for w in weights])
        _flat_agg = np.concatenate([a.flatten() for a in aggregated])
        _dists = [float(np.linalg.norm(_flat_all[i] - _flat_agg)) for i in range(len(weights))]
        _max_d = max(_dists) if _dists and max(_dists) > 1e-10 else 1.0
        client_scores_dict = {str(cids[i]): float(1.0 - _dists[i] / _max_d) for i in range(len(weights))}

        metrics = {
            "included_clients": json.dumps(sorted(included_cids)),
            "excluded_clients": json.dumps(sorted(excluded_cids)),
            "client_scores": json.dumps(client_scores_dict),
        }
        return aggregated, metrics
