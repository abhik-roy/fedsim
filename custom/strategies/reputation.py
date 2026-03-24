"""Reputation and Trust-based client selection strategy plugin.

Implements the Reputation and Trust framework from Patel, Chuprov et al.
(ASIA'24) for detecting and excluding adversarial clients in federated
learning.  Each round the strategy:

1. Computes a per-client *truth value* ``d`` — how close the client's
   parameter update is to the group centroid (median).
2. Updates *reputation* ``R`` asymmetrically: linear growth when
   ``d >= α``, exponential decay otherwise (Eq. 1-3).
3. Derives a *trust indicator* from ``R`` and ``d`` with exponential
   smoothing across rounds (Eq. 4-6).
4. After configurable warm-up rounds, excludes any client whose trust
   falls below a threshold ``θ`` from the FedAvg aggregation.

References
----------
Patel H., Chuprov S., Korobeinikov D., Zatsarenko R., Reznik L.
"Improving Federated Learning Security with Trust Evaluation to Detect
Adversarial Attacks", 19th Annual Symposium on Information Assurance
(ASIA'24), 2024.

Chuprov S. et al. "Reputation and Trust Approach for Security and Safety
Assurance in Intersection Management System", Energies 12(23), 2019.
"""
import json
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from fl_core import FedAvg, FitRes, NDArrays

NAME = "Reputation"
DESCRIPTION = (
    "Trust-based client exclusion with asymmetric reputation updates "
    "and exponential trust smoothing (Chuprov et al.)"
)

PARAMS = {
    "truth_threshold": {
        "type": "float", "default": 0.5, "min": 0.1, "max": 1.0, "step": 0.05,
        "label": "Truth Threshold (α)",
    },
    "trust_exclusion_threshold": {
        "type": "float", "default": 0.15, "min": 0.0, "max": 1.0, "step": 0.05,
        "label": "Trust Exclusion Threshold (θ)",
    },
    "initial_reputation": {
        "type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
        "label": "Initial Reputation",
    },
    "warmup_rounds": {
        "type": "int", "default": 3, "min": 0, "max": 10, "step": 1,
        "label": "Warm-up Rounds",
    },
    "smoothing_beta": {
        "type": "float", "default": 0.85, "min": 0.0, "max": 1.0, "step": 0.05,
        "label": "Trust Smoothing (β)",
    },
}


class ReputationStrategy(FedAvg):
    """Reputation and Trust-based client exclusion strategy.

    Maintains per-client reputation and trust scores across rounds.  Each
    round:
    1. Computes truth ``d`` per client (closeness to centroid).
    2. Updates reputation ``R`` asymmetrically (Eq. 1-3).
    3. Computes trust ``T`` from ``R`` and ``d`` with smoothing (Eq. 4-6).
    4. After warm-up, excludes clients with ``T < θ`` from aggregation.
    """

    def __init__(
        self,
        num_clients: int,
        truth_threshold: float = 0.5,
        trust_exclusion_threshold: float = 0.15,
        initial_reputation: float = 0.5,
        warmup_rounds: int = 3,
        smoothing_beta: float = 0.85,
        # Legacy parameter — ignored, kept for backward compatibility
        selection_fraction: float | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_clients = num_clients
        self.truth_threshold = truth_threshold          # α in the paper
        self.trust_exclusion_threshold = trust_exclusion_threshold  # θ
        self.initial_reputation = initial_reputation
        self.warmup_rounds = warmup_rounds
        self.smoothing_beta = smoothing_beta            # β in the paper

        self.reputations: dict[int, float] = {
            i: initial_reputation for i in range(num_clients)
        }
        self.trust_scores: dict[int, float] = {
            i: 0.0 for i in range(num_clients)
        }
        self.current_round = 0
        self.global_params: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Truth value: d_i = 1 - normalized_sq_distance_to_major_cluster_center
    #
    # Per the paper: K-means (K=2) clusters client update deltas in
    # parameter space.  The centroid of the *larger* cluster is used as
    # the reference point — this naturally ignores the minority
    # (malicious) cluster entirely.
    # ------------------------------------------------------------------
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

        # K-means clustering to find the major cluster center
        if len(cids) >= 3:
            kmeans = KMeans(n_clusters=2, n_init=10, random_state=0)
            labels = kmeans.fit_predict(flat_params)
            # Identify the larger cluster
            counts = np.bincount(labels, minlength=2)
            major_label = int(np.argmax(counts))
            centroid = kmeans.cluster_centers_[major_label]
        else:
            # Too few clients for meaningful clustering — fall back to mean
            centroid = np.mean(flat_params, axis=0)

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

    # ------------------------------------------------------------------
    # Reputation update — Eq. 1-3 from the paper
    # ------------------------------------------------------------------
    def _update_reputations(self, truth_values: dict[int, float],
                            server_round: int | None = None) -> None:
        rnd = server_round if server_round is not None else self.current_round + 1
        self.current_round = rnd

        for cid, d in truth_values.items():
            prev_rep = self.reputations.get(cid, self.initial_reputation)

            if rnd == 1:
                # Eq. 1: R_i^{t0} = d_i
                new_rep = d
            else:
                # Eq. 2-3
                decay_term = prev_rep / rnd
                if d >= self.truth_threshold:
                    # Linear growth: R + d + R/t
                    new_rep = prev_rep + d + decay_term
                else:
                    # Exponential decay: R + d - e^{-(1-d)} * R/t
                    new_rep = prev_rep + d - np.exp(-(1 - d)) * decay_term

            self.reputations[cid] = float(np.clip(new_rep, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Trust indicator — Eq. 4-6 from the paper
    # ------------------------------------------------------------------
    def _update_trust(self, truth_values: dict[int, float]) -> None:
        beta = self.smoothing_beta

        for cid, d in truth_values.items():
            R = self.reputations[cid]

            # Eq. 4: raw trust from reputation and truth
            raw_trust = np.sqrt(R**2 + d**2) - np.sqrt((1 - R)**2 + (1 - d)**2)

            # Eq. 5: exponential smoothing with previous trust
            prev_trust = self.trust_scores.get(cid, 0.0)
            smoothed = beta * raw_trust + (1 - beta) * prev_trust

            # Eq. 6: clip to [0, 1]
            self.trust_scores[cid] = float(np.clip(smoothed, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Client selection — threshold-based on trust (not top-k)
    # ------------------------------------------------------------------
    def _select_clients(
        self, results: list[tuple[int, FitRes]], server_round: int,
    ) -> list[tuple[int, FitRes]]:
        # During warm-up, include all clients
        if server_round <= self.warmup_rounds:
            return list(results)

        # After warm-up: exclude clients whose trust < θ
        selected = []
        for cid, fit_res in results:
            trust = self.trust_scores.get(cid, 0.0)
            if trust >= self.trust_exclusion_threshold:
                selected.append((cid, fit_res))

        # Safety: always keep at least one client
        if not selected:
            best_cid = max(
                [(cid, self.trust_scores.get(cid, 0.0)) for cid, _ in results],
                key=lambda x: x[1],
            )[0]
            selected = [(cid, fr) for cid, fr in results if cid == best_cid]

        return selected

    def get_reputations(self) -> dict[int, float]:
        return dict(self.reputations)

    def get_trust_scores(self) -> dict[int, float]:
        return dict(self.trust_scores)

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

        # Step 1-2: compute truth, update reputation
        truth_values = self._compute_truth_values(all_weights)
        self._update_reputations(truth_values, server_round=server_round)

        # Step 3: compute trust indicator
        self._update_trust(truth_values)

        # Step 4: select clients based on trust threshold
        selected = self._select_clients(results, server_round)

        if not selected:
            return None, {}

        # FedAvg over selected clients only
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
            "client_scores": json.dumps({
                str(k): float(v) for k, v in self.trust_scores.items()
            }),
        }
        return aggregated, metrics


def build(initial_parameters, num_clients, num_malicious=0, **kwargs):
    """Build a ReputationStrategy from plugin params."""
    truth_threshold = kwargs.pop("truth_threshold", PARAMS["truth_threshold"]["default"])
    trust_exclusion_threshold = kwargs.pop(
        "trust_exclusion_threshold",
        PARAMS["trust_exclusion_threshold"]["default"],
    )
    initial_reputation = kwargs.pop("initial_reputation", PARAMS["initial_reputation"]["default"])
    warmup_rounds = kwargs.pop("warmup_rounds", PARAMS["warmup_rounds"]["default"])
    smoothing_beta = kwargs.pop("smoothing_beta", PARAMS["smoothing_beta"]["default"])
    # Consume legacy param if passed
    kwargs.pop("selection_fraction", None)
    return ReputationStrategy(
        num_clients=num_clients,
        truth_threshold=truth_threshold,
        trust_exclusion_threshold=trust_exclusion_threshold,
        initial_reputation=initial_reputation,
        warmup_rounds=warmup_rounds,
        smoothing_beta=smoothing_beta,
        initial_parameters=initial_parameters,
        **kwargs,
    )
