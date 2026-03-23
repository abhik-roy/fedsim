import json
from typing import Any

import numpy as np
from fl_core import FedAvg, FitRes, NDArrays


class TrimmedMean(FedAvg):
    def __init__(self, beta: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.beta = beta

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[int, FitRes]],
        failures: list,
    ) -> tuple[NDArrays | None, dict[str, Any]]:
        if not results:
            return None, {}

        # Extract weights from results
        weights = [fit_res.parameters for _, fit_res in results]
        num_clients = len(weights)
        trim_count = max(1, int(num_clients * self.beta))
        # Ensure at least 1 client survives after trimming both ends
        max_trim = (num_clients - 1) // 2
        trim_count = min(trim_count, max_trim)

        # Coordinate-wise trimmed mean
        aggregated = []
        for layer_idx in range(len(weights[0])):
            layer_stack = np.stack([w[layer_idx] for w in weights], axis=0)
            sorted_stack = np.sort(layer_stack, axis=0)
            trimmed = sorted_stack[trim_count : num_clients - trim_count]
            if trimmed.shape[0] == 0:
                trimmed = sorted_stack
            aggregated.append(np.mean(trimmed, axis=0))

        # Compute per-client scores: L2 distance from aggregated result
        cids = [cid for cid, _ in results]
        flat_agg = np.concatenate([a.flatten() for a in aggregated])
        client_scores_dict = {}
        client_distances = []
        for i in range(num_clients):
            flat_client = np.concatenate([weights[i][layer_idx].flatten() for layer_idx in range(len(weights[i]))])
            dist = float(np.linalg.norm(flat_client - flat_agg))
            client_distances.append(dist)
        max_dist = max(client_distances) if max(client_distances) > 1e-10 else 1.0
        for i in range(num_clients):
            client_scores_dict[str(cids[i])] = float(1.0 - client_distances[i] / max_dist)

        metrics = {
            "trim_count": trim_count,
            "aggregation_method": "trimmed_mean",
            "included_clients": json.dumps(sorted(cids)),
            "excluded_clients": json.dumps([]),
            "client_scores": json.dumps(client_scores_dict),
        }
        return aggregated, metrics
