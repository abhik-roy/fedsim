import json
from typing import Any

import numpy as np
from fl_core import FedAvg, FitRes, NDArrays


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

        aggregated = []
        for layer_idx in range(len(weights[0])):
            layer_stack = np.stack([w[layer_idx] for w in weights], axis=0)
            aggregated.append(np.median(layer_stack, axis=0))

        cids = [cid for cid, _ in results]
        metrics = {
            "aggregation_method": "coordinate_median",
            "included_clients": json.dumps(sorted(cids)),
            "excluded_clients": json.dumps([]),
        }
        return aggregated, metrics
