import json
from typing import Any

import numpy as np
from fl_core import FedAvg, FitRes, NDArrays
from strategies.utils import filter_nan_clients


class KrumStrategy(FedAvg):
    """Byzantine-robust aggregation strategy using the Krum algorithm (Blanchard et al., 2017).

    Krum selects the client update that is closest to its nearest neighbors,
    providing robustness against up to f Byzantine clients when n >= 2f + 3.
    Multi-Krum averages the top-m scoring clients rather than selecting just one,
    improving accuracy while retaining Byzantine resilience.

    Attributes:
        num_malicious: Number of expected Byzantine (malicious) clients f.
        multi_krum: If True, use Multi-Krum (average top n-f clients).
            If False, use single Krum (select the single best client).
    """

    def __init__(self, num_malicious: int = 0, multi_krum: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.num_malicious = num_malicious
        self.multi_krum = multi_krum

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

        # Validate Krum invariant: n >= 2f + 3
        if num_clients < 2 * self.num_malicious + 3:
            import warnings
            warnings.warn(
                f"Krum requires n >= 2f + 3 (have n={num_clients}, f={self.num_malicious}). "
                f"Falling back to selecting based on available neighbors.",
                stacklevel=2,
            )

        # Flatten each client's weights into a single vector
        flat_weights = np.array([
            np.concatenate([layer.flatten() for layer in w]) for w in weights
        ])

        # Compute pairwise squared distances via broadcasting
        num_to_select = max(1, num_clients - self.num_malicious - 2)

        sq_norms = np.sum(flat_weights ** 2, axis=1)
        distances = sq_norms[:, None] + sq_norms[None, :] - 2 * flat_weights @ flat_weights.T
        np.maximum(distances, 0, out=distances)  # clamp numerical noise

        # Krum score: sum of distances to nearest num_to_select neighbors
        # Set diagonal to inf so self-distance is never among the smallest
        np.fill_diagonal(distances, np.inf)
        k = min(num_to_select, num_clients - 1)
        partitioned = np.partition(distances, k, axis=1)[:, :k]
        scores = partitioned.sum(axis=1).tolist()

        if self.multi_krum:
            # Multi-Krum: average the top-m clients (m = n - 2f per Blanchard et al.)
            m = max(1, num_clients - 2 * self.num_malicious)
            selected = np.argsort(scores)[:m]
        else:
            # Single Krum: select the client with minimum score
            selected = [np.argmin(scores)]

        # Average the selected clients
        aggregated = []
        for layer_idx in range(len(weights[0])):
            layer_stack = np.stack([weights[s][layer_idx] for s in selected], axis=0)
            aggregated.append(np.mean(layer_stack, axis=0).astype(weights[0][layer_idx].dtype))

        all_cids = set(cids)
        included_cids = {cids[int(s)] for s in selected}
        excluded_cids = all_cids - included_cids
        # Normalize: higher = more trusted (consistent with all other strategies)
        max_score = max(scores) if scores else 1.0
        max_score = max_score if max_score > 1e-10 else 1.0
        client_scores_dict = {str(cids[i]): float(1.0 - scores[i] / max_score) for i in range(num_clients)}

        metrics = {
            "included_clients": json.dumps(sorted(included_cids)),
            "excluded_clients": json.dumps(sorted(excluded_cids)),
            "client_scores": json.dumps(client_scores_dict),
        }
        return aggregated, metrics
