"""Reputation-based client selection strategy plugin.

Maintains per-client reputation scores across rounds using asymmetric
temporal updates (Chuprov et al.). Selects top-k clients by reputation
for FedAvg aggregation. Clients with low truth values (far from centroid)
have their reputation decay exponentially, while honest clients grow linearly.

Based on: Chuprov et al. "Reputation and Trust Approach for Security and
Safety Assurance" and Roy's capstone implementation.
"""
import json
from typing import Any

import numpy as np
from fl_core import FedAvg, FitRes, NDArrays

NAME = "Reputation"
DESCRIPTION = "Trust-based client selection with asymmetric reputation updates (Chuprov et al.)"

PARAMS = {
    "truth_threshold": {
        "type": "float", "default": 0.7, "min": 0.1, "max": 1.0, "step": 0.05,
        "label": "Truth Threshold",
    },
    "selection_fraction": {
        "type": "float", "default": 0.6, "min": 0.1, "max": 1.0, "step": 0.05,
        "label": "Selection Fraction",
    },
    "initial_reputation": {
        "type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
        "label": "Initial Reputation",
    },
}


class ReputationStrategy(FedAvg):
    """Reputation-based client selection strategy.

    Maintains per-client reputation scores across rounds. Each round:
    1. Computes a 'truth' value per client based on how close the client's
       update is to the group centroid.
    2. Updates reputation asymmetrically: linear growth when truth >= threshold,
       exponential decay otherwise.
    3. Selects top-k clients by reputation for FedAvg aggregation.
    """

    def __init__(
        self,
        num_clients: int,
        selection_fraction: float = 0.6,
        truth_threshold: float = 0.7,
        initial_reputation: float = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_clients = num_clients
        self.selection_fraction = selection_fraction
        self.truth_threshold = truth_threshold
        self.initial_reputation = initial_reputation

        self.reputations: dict[int, float] = {
            i: initial_reputation for i in range(num_clients)
        }
        self.current_round = 0
        self.global_params: np.ndarray | None = None

    def _compute_truth_values(
        self, all_weights: dict[int, list[np.ndarray]]
    ) -> dict[int, float]:
        if not all_weights:
            return {}

        cids = list(all_weights.keys())
        flat_params = []
        for cid in cids:
            client_flat = np.concatenate([p.flatten() for p in all_weights[cid]])
            if self.global_params is not None:
                delta = client_flat - self.global_params
            else:
                delta = client_flat
            flat_params.append(delta)
        flat_params = np.array(flat_params)

        centroid = np.median(flat_params, axis=0)

        sq_distances = np.array([
            np.sum((fp - centroid) ** 2) for fp in flat_params
        ])
        max_sq_dist = np.max(sq_distances)
        if max_sq_dist < 1e-10:
            return {cid: 1.0 for cid in cids}

        truth = {}
        for i, cid in enumerate(cids):
            normalized_dist = sq_distances[i] / max_sq_dist
            truth[cid] = float(max(0.0, 1.0 - normalized_dist))
        return truth

    def _update_reputations(self, truth_values: dict[int, float],
                            server_round: int | None = None) -> None:
        rnd = server_round if server_round is not None else self.current_round + 1
        self.current_round = rnd

        for cid, truth in truth_values.items():
            prev_rep = self.reputations.get(cid, self.initial_reputation)

            if rnd == 1:
                self.reputations[cid] = truth
            else:
                decay_term = prev_rep / rnd
                if truth >= self.truth_threshold:
                    new_rep = prev_rep + truth - decay_term
                else:
                    new_rep = prev_rep + truth - np.exp(1 - truth) * decay_term

                self.reputations[cid] = float(np.clip(new_rep, 0.0, 1.0))

    def _select_clients(
        self, results: list[tuple[int, FitRes]]
    ) -> list[tuple[int, FitRes]]:
        cid_to_idx = {}
        for idx, (cid, _) in enumerate(results):
            cid_to_idx[cid] = idx

        num_participating = len(cid_to_idx)
        num_to_select = max(1, int(num_participating * self.selection_fraction))

        # Rank all participating clients by reputation and take the top fraction.
        # No hard eligibility cutoff — selection_fraction alone controls how many
        # clients are included, and ranking ensures the best-reputed are chosen.
        ranked = sorted(
            cid_to_idx.keys(),
            key=lambda c: self.reputations.get(c, 0.0),
            reverse=True,
        )
        selected_cids = ranked[:num_to_select]

        return [results[cid_to_idx[cid]] for cid in selected_cids]

    def get_reputations(self) -> dict[int, float]:
        return dict(self.reputations)

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[int, FitRes]],
        failures: list,
    ) -> tuple[NDArrays | None, dict[str, Any]]:
        if not results:
            return None, {}

        all_weights = {}
        for cid, fit_res in results:
            all_weights[cid] = fit_res.parameters

        truth_values = self._compute_truth_values(all_weights)
        self._update_reputations(truth_values, server_round=server_round)
        selected = self._select_clients(results)

        if not selected:
            return None, {}

        weights_list = [fr.parameters for _, fr in selected]
        total_examples = sum(fr.num_examples for _, fr in selected)

        if total_examples == 0:
            return None, {}

        aggregated = []
        for layer_idx in range(len(weights_list[0])):
            layer_sum = np.zeros_like(weights_list[0][layer_idx], dtype=np.float64)
            for w_idx, w in enumerate(weights_list):
                num_ex = selected[w_idx][1].num_examples
                layer_sum += w[layer_idx] * num_ex
            aggregated.append(
                (layer_sum / total_examples).astype(weights_list[0][layer_idx].dtype)
            )

        # Update global_params so truth values are computed as deltas next round
        self.global_params = np.concatenate([a.flatten() for a in aggregated])

        selected_cids = {cid for cid, _ in selected}
        all_cids = set(all_weights.keys())
        excluded_cids = all_cids - selected_cids
        metrics = {
            "included_clients": json.dumps(sorted(selected_cids)),
            "excluded_clients": json.dumps(sorted(excluded_cids)),
            "client_scores": json.dumps({str(k): float(v) for k, v in self.reputations.items()}),
        }
        return aggregated, metrics


def build(initial_parameters, num_clients, num_malicious=0, **kwargs):
    """Build a ReputationStrategy from plugin params."""
    truth_threshold = kwargs.pop("truth_threshold", PARAMS["truth_threshold"]["default"])
    selection_fraction = kwargs.pop("selection_fraction", PARAMS["selection_fraction"]["default"])
    initial_reputation = kwargs.pop("initial_reputation", PARAMS["initial_reputation"]["default"])
    return ReputationStrategy(
        num_clients=num_clients,
        selection_fraction=selection_fraction,
        truth_threshold=truth_threshold,
        initial_reputation=initial_reputation,
        initial_parameters=initial_parameters,
        **kwargs,
    )
