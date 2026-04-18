import json
from typing import Any

import numpy as np
from fl_core import FedAvg, FitRes, NDArrays
from strategies.utils import filter_nan_clients


class MedianStrategy(FedAvg):
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

        aggregated = []
        for layer_idx in range(len(weights[0])):
            layer_stack = np.stack([w[layer_idx] for w in weights], axis=0)
            med = np.median(layer_stack, axis=0)
            aggregated.append(med.astype(weights[0][layer_idx].dtype))

        flat_agg = np.concatenate([a.flatten() for a in aggregated])
        client_scores_dict = {}
        for i in range(num_clients):
            flat_client = np.concatenate([weights[i][j].flatten()
                                          for j in range(len(weights[i]))])
            dist = float(np.linalg.norm(flat_client - flat_agg))
            client_scores_dict[str(cids[i])] = dist
        max_dist = max(client_scores_dict.values()) if client_scores_dict else 1.0
        max_dist = max_dist if max_dist > 1e-10 else 1.0
        client_scores_dict = {k: float(1.0 - v / max_dist) for k, v in client_scores_dict.items()}
        metrics = {
            "aggregation_method": "coordinate_median",
            "included_clients": json.dumps(sorted(cids)),
            "excluded_clients": json.dumps([]),
            "client_scores": json.dumps(client_scores_dict),
        }
        return aggregated, metrics
