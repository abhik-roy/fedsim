"""Reputation V2 — Per-Layer Cosine Trust with Median Clipping.

An enhanced Reputation and Trust strategy that replaces the original K-means
Euclidean clustering (Chuprov et al., ASIA'24) with a per-layer cosine
similarity approach inspired by Garcia-Marquez et al. (Knowledge-Based
Systems, 2025) and FLTrust (Cao et al., NDSS 2021).

Problem with V1 (original Reputation)
--------------------------------------
The original strategy runs K-means (K=2) on the full flattened parameter
vector to compute truth values.  For high-dimensional models like ResNet-18
(~11M parameters), this fails due to the curse of dimensionality: Euclidean
distances concentrate, K-means produces arbitrary clusters, and malicious
clients are indistinguishable from benign ones.  Empirically, only 1 out of
4 malicious clients was ever excluded under CIFAR-10/ResNet-18 with label
flipping (see experiments/clustering_sanity_check.py).

Solution: Per-Layer Cosine Similarity
--------------------------------------
Instead of operating on the full flattened vector, we decompose the
comparison into per-layer sub-problems:

1. **Layer decomposition** (Garcia-Marquez et al., Prop. 1):  For each
   layer j in the model, we compute client update deltas independently.
   This reduces the effective dimensionality from d to m_j per layer,
   where d = m_1 + ... + m_k.  The theoretical (alpha, f)-Byzantine
   resilience is preserved under layerwise application.

2. **Median gradient clipping** (Garcia-Marquez et al., Sec. 3.2):  Before
   computing cosine similarity per layer, we clip each client's layer
   delta to the median norm of all clients' deltas for that layer.  This
   prevents magnitude attacks where a malicious client submits a
   correct-direction but enormous-norm update.  Cosine distance alone is
   blind to magnitude; clipping restores sensitivity.

3. **Server reference update** (FLTrust, Cao et al., NDSS 2021):  The
   server trains the global model for 1 epoch on its validation set to
   obtain a clean reference update direction.  This provides a robust
   baseline that is immune to non-IID client heterogeneity — the signal
   that causes peer-based references (median, trimmed mean) to fail on
   high-dimensional models.  When server data is unavailable, falls back
   to a per-layer trimmed mean of client deltas.

4. **Per-layer cosine similarity**:  For each layer, we compute
   cos(client_delta_j, reference_j).  This yields one similarity score
   per layer per client.

5. **Robust aggregation of layer scores**:  We use the 25th percentile
   (first quartile) of per-layer cosine similarities as the client's
   truth value, rather than the mean.  This prevents a targeted attack
   where a malicious client makes most layers look correct while
   poisoning a few high-impact layers (e.g., the classification head).

The resulting truth value d_i in [0, 1] feeds into the same reputation
and trust pipeline from Chuprov et al. (Eq. 1-6).

Mathematical Formulation
------------------------
Let V_i^t denote client i's parameters at round t, and V_G^t the global
model.  The per-client update delta is:

    delta_i = V_i^t - V_G^t

Decompose into layers:  delta_i = (delta_{i,1}, ..., delta_{i,k})

For each layer j:

    1. Median clipping:
       m_j = median({||delta_{1,j}||, ..., ||delta_{n,j}||})
       delta_hat_{i,j} = delta_{i,j} * min(1, m_j / ||delta_{i,j}||)

    2. Trimmed mean reference (exclude top/bottom by norm):
       ref_j = mean({delta_hat_{i,j} : i in trimmed_set_j})

    3. Cosine similarity:
       s_{i,j} = cos(delta_hat_{i,j}, ref_j)
               = <delta_hat_{i,j}, ref_j> / (||delta_hat_{i,j}|| * ||ref_j||)

    4. Robust aggregation (25th percentile across layers):
       d_i = P25({s_{i,1}, ..., s_{i,k}})

    5. Normalize to [0, 1]:
       d_i = max(0, d_i)  (cosine similarity is in [-1, 1])

The rest of the pipeline (reputation update, trust smoothing, threshold-
based exclusion) follows Chuprov et al. exactly:

    - Eq. 1:  R_i^{t0} = d_i                         (round 1)
    - Eq. 2:  R = R + d + R/t                         (d >= alpha, linear growth)
    - Eq. 3:  R = R + d - exp(-(1-d)) * R/t           (d < alpha, exponential decay)
    - Eq. 4:  T_raw = sqrt(R^2 + d^2) - sqrt((1-R)^2 + (1-d)^2)
    - Eq. 5:  T = beta * T_raw + (1-beta) * T_prev    (exponential smoothing)
    - Eq. 6:  T = clip(T, 0, 1)

After warmup_rounds, clients with T < theta are excluded from FedAvg.

References
----------
[1] Garcia-Marquez M., Rodriguez-Barroso N., Luzon M.V., Herrera F.
    "Improving (alpha, f)-Byzantine Resilience in Federated Learning via
    Layerwise Aggregation and Cosine Distance", Knowledge-Based Systems,
    2025.  arXiv:2503.21244.

[2] Patel H., Chuprov S., et al. "Improving Federated Learning Security
    with Trust Evaluation to Detect Adversarial Attacks", ASIA'24, 2024.

[3] Cao X., Fang M., Liu J., Gong N. "FLTrust: Byzantine-robust Federated
    Learning via Trust Bootstrapping", NDSS, 2021.

[4] Chuprov S. et al. "Reputation and Trust Approach for Security and
    Safety Assurance in Intersection Management System", Energies 12(23),
    2019.
"""
import json
from typing import Any

import numpy as np
from fl_core import FedAvg, FitRes, NDArrays

NAME = "ReputationV2"
DESCRIPTION = (
    "Per-layer cosine trust with median clipping — enhanced Reputation "
    "strategy for high-dimensional models (Garcia-Marquez + Chuprov)"
)

PARAMS = {
    "truth_threshold": {
        "type": "float", "default": 0.5, "min": 0.1, "max": 1.0, "step": 0.05,
        "label": "Truth Threshold (alpha)",
    },
    "trust_exclusion_threshold": {
        "type": "float", "default": 0.15, "min": 0.0, "max": 1.0, "step": 0.05,
        "label": "Trust Exclusion Threshold (theta)",
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
        "label": "Trust Smoothing (beta)",
    },
    "trim_fraction": {
        "type": "float", "default": 0.2, "min": 0.0, "max": 0.45, "step": 0.05,
        "label": "Trimmed Mean Fraction",
    },
    "layer_score_percentile": {
        "type": "int", "default": 25, "min": 5, "max": 50, "step": 5,
        "label": "Layer Score Percentile",
    },
}


class ReputationV2Strategy(FedAvg):
    """Per-Layer Cosine Reputation strategy for high-dimensional FL models.

    Replaces K-means Euclidean truth values with per-layer cosine similarity
    to a robust trimmed-mean reference, with median gradient clipping.
    Reputation and trust pipeline unchanged from Chuprov et al.
    """

    def __init__(
        self,
        num_clients: int,
        truth_threshold: float = 0.5,
        trust_exclusion_threshold: float = 0.15,
        initial_reputation: float = 0.5,
        warmup_rounds: int = 3,
        smoothing_beta: float = 0.85,
        trim_fraction: float = 0.2,
        layer_score_percentile: int = 25,
        # Consumed but unused — kept for compat with runner kwargs
        selection_fraction: float | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_clients = num_clients
        self.truth_threshold = truth_threshold
        self.trust_exclusion_threshold = trust_exclusion_threshold
        self.initial_reputation = initial_reputation
        self.warmup_rounds = warmup_rounds
        self.smoothing_beta = smoothing_beta
        self.trim_fraction = trim_fraction
        self.layer_score_percentile = layer_score_percentile

        self.reputations: dict[int, float] = {}
        self.trust_scores: dict[int, float] = {}
        self.current_round = 0
        # Store global params as list of per-layer arrays (preserving layer structure)
        self.global_params_layers: list[np.ndarray] | None = None

        # Server-side resources (set via configure() from the runner)
        self._valloader = None
        self._model_name: str | None = None
        self._dataset_name: str | None = None
        self._learning_rate: float = 0.01
        self._device = None

    # ------------------------------------------------------------------
    # configure() — called by the runner after strategy construction
    # ------------------------------------------------------------------
    def configure(self, valloader=None, model_name=None, dataset_name=None,
                  learning_rate=0.01, device=None, **kwargs):
        """Receive server-side resources from the runner.

        The validation loader is used to compute a server reference update
        each round (FLTrust-style), providing a clean directional baseline
        that is robust to non-IID client heterogeneity.
        """
        self._valloader = valloader
        self._model_name = model_name
        self._dataset_name = dataset_name
        self._learning_rate = learning_rate
        self._device = device
        self._loss_name = kwargs.get("loss_name", "cross_entropy")

    # ------------------------------------------------------------------
    # Server reference update (FLTrust-style)
    # ------------------------------------------------------------------
    def _compute_server_reference(self) -> list[np.ndarray] | None:
        """Train the global model for 1 epoch on the server validation set.

        Returns a list of per-layer numpy arrays (the server-updated params),
        or None if server-side resources are unavailable.
        """
        if (self._valloader is None or self._model_name is None
                or self._dataset_name is None or self.global_params_layers is None):
            return None

        import torch
        import torch.nn as nn
        from collections import OrderedDict
        from models import get_model

        device = self._device or torch.device("cpu")
        model = get_model(self._model_name, self._dataset_name).to(device)

        # Load current global params
        state_dict = OrderedDict(
            {k: torch.from_numpy(np.asarray(v))
             for k, v in zip(model.state_dict().keys(), self.global_params_layers)}
        )
        model.load_state_dict(state_dict, strict=True)

        # Train for 1 epoch on validation data
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=self._learning_rate)
        # Use the configured loss function instead of hardcoded CrossEntropyLoss
        _loss = getattr(self, '_loss_name', 'cross_entropy')
        if _loss == "bce_with_logits":
            criterion = nn.BCEWithLogitsLoss()
        elif _loss == "nll":
            criterion = nn.NLLLoss()
        elif _loss == "mse":
            criterion = nn.MSELoss()
        else:
            criterion = nn.CrossEntropyLoss()
        for batch in self._valloader:
            if isinstance(batch, (list, tuple)):
                inputs, targets = batch[0].to(device), batch[1].to(device)
            else:
                continue
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        return [v.detach().cpu().numpy() for v in model.state_dict().values()]

    # ------------------------------------------------------------------
    # Truth value: per-layer cosine similarity with median clipping
    # ------------------------------------------------------------------
    def _compute_truth_values(
        self, all_weights: dict[int, list[np.ndarray]]
    ) -> dict[int, float]:
        if not all_weights:
            return {}

        cids = list(all_weights.keys())
        n = len(cids)

        if n < 2:
            return {cid: 1.0 for cid in cids}

        # Filter NaN/Inf clients
        valid_mask = []
        for cid in cids:
            is_valid = all(
                bool(np.all(np.isfinite(layer))) for layer in all_weights[cid]
            )
            valid_mask.append(is_valid)
        invalid_cids = [c for c, v in zip(cids, valid_mask) if not v]

        if not any(valid_mask):
            return {cid: 0.0 for cid in cids}

        if invalid_cids:
            valid_cids = [c for c, v in zip(cids, valid_mask) if v]
            # Rebuild with only valid clients
            valid_weights = {c: all_weights[c] for c in valid_cids}
            truth = self._compute_truth_values(valid_weights)
            for cid in invalid_cids:
                truth[cid] = 0.0
            return truth

        num_layers = len(all_weights[cids[0]])

        # Compute server reference update (FLTrust-style)
        server_params = self._compute_server_reference()

        # Compute per-layer deltas
        per_layer_deltas: list[np.ndarray] = []
        server_layer_deltas: list[np.ndarray | None] = []
        for layer_idx in range(num_layers):
            layer_deltas = []
            for cid in cids:
                client_layer = all_weights[cid][layer_idx].flatten()
                if self.global_params_layers is not None:
                    global_layer = self.global_params_layers[layer_idx].flatten()
                    delta = client_layer - global_layer
                else:
                    delta = client_layer
                layer_deltas.append(delta.astype(np.float64))
            per_layer_deltas.append(np.array(layer_deltas))

            # Server delta for this layer
            if server_params is not None and self.global_params_layers is not None:
                s_delta = (server_params[layer_idx].flatten().astype(np.float64)
                           - self.global_params_layers[layer_idx].flatten().astype(np.float64))
                server_layer_deltas.append(s_delta)
            else:
                server_layer_deltas.append(None)

        # Compute per-layer cosine similarity
        per_client_layer_sims: dict[int, list[float]] = {cid: [] for cid in cids}

        for layer_idx in range(num_layers):
            deltas = per_layer_deltas[layer_idx]
            layer_dim = deltas.shape[1]

            # Skip tiny layers (bias terms, BN scalars)
            if layer_dim < 2:
                for cid in cids:
                    per_client_layer_sims[cid].append(1.0)
                continue

            # Step 1: Median gradient clipping (Garcia-Marquez Sec. 3.2)
            norms = np.linalg.norm(deltas, axis=1)
            median_norm = np.median(norms)

            if median_norm < 1e-10:
                for cid in cids:
                    per_client_layer_sims[cid].append(1.0)
                continue

            clipped_deltas = np.copy(deltas)
            for i in range(n):
                if norms[i] > median_norm and norms[i] > 1e-10:
                    clipped_deltas[i] = deltas[i] * (median_norm / norms[i])

            # Step 2: Reference direction
            # Prefer server reference (FLTrust); fall back to trimmed mean
            s_delta = server_layer_deltas[layer_idx]
            if s_delta is not None and np.linalg.norm(s_delta) > 1e-10:
                # Clip server delta to median norm too
                s_norm = np.linalg.norm(s_delta)
                if s_norm > median_norm:
                    s_delta = s_delta * (median_norm / s_norm)
                reference = s_delta
            else:
                # Fallback: trimmed mean of clipped client deltas.
                # k_trim=0 when trim_fraction=0.0 (honor user intent: no trimming).
                clipped_norms = np.linalg.norm(clipped_deltas, axis=1)
                k_trim = int(n * self.trim_fraction)
                sorted_idx = np.argsort(clipped_norms)
                if k_trim > 0 and 2 * k_trim < n:
                    trimmed_idx = sorted_idx[k_trim:-k_trim]
                else:
                    trimmed_idx = sorted_idx
                reference = np.mean(clipped_deltas[trimmed_idx], axis=0)

            ref_norm = np.linalg.norm(reference)
            if ref_norm < 1e-10:
                for cid in cids:
                    per_client_layer_sims[cid].append(1.0)
                continue

            # Step 3: Cosine similarity to reference
            for i, cid in enumerate(cids):
                client_norm = np.linalg.norm(clipped_deltas[i])
                if client_norm < 1e-10:
                    per_client_layer_sims[cid].append(0.0)
                else:
                    cos_sim = float(
                        np.dot(clipped_deltas[i], reference) / (client_norm * ref_norm)
                    )
                    per_client_layer_sims[cid].append(cos_sim)

        # Step 4: Robust aggregation + relative normalization
        #
        # First, compute each client's raw score as the percentile of their
        # per-layer cosine similarities.  Then normalize across clients using
        # median/MAD so that truth values reflect relative standing within
        # the group, not absolute cosine magnitude.  This is critical because
        # even benign clients can have low absolute cosine to the server
        # reference under non-IID data.
        raw_scores = {}
        for cid in cids:
            sims = per_client_layer_sims[cid]
            if not sims:
                raw_scores[cid] = 1.0
            else:
                raw_scores[cid] = float(np.percentile(sims, self.layer_score_percentile))

        # Relative normalization: map raw scores to [0, 1] via min-max
        # within the current round's client pool
        scores_arr = np.array([raw_scores[c] for c in cids])
        s_min = np.min(scores_arr)
        s_max = np.max(scores_arr)

        truth = {}
        if s_max - s_min < 1e-10:
            # All clients have identical scores — all equally trustworthy
            for cid in cids:
                truth[cid] = 1.0
        else:
            for cid in cids:
                # Linear mapping: worst client → 0, best → 1
                normalized = (raw_scores[cid] - s_min) / (s_max - s_min)
                truth[cid] = float(np.clip(normalized, 0.0, 1.0))
        return truth

    # ------------------------------------------------------------------
    # Reputation update — Eq. 1-3 from Chuprov et al.
    # (Identical to V1)
    # ------------------------------------------------------------------
    def _update_reputations(self, truth_values: dict[int, float],
                            server_round: int | None = None) -> None:
        rnd = server_round if server_round is not None else self.current_round + 1
        self.current_round = rnd

        for cid, d in truth_values.items():
            prev_rep = self.reputations.get(cid, self.initial_reputation)

            if rnd == 1:
                new_rep = d
            else:
                decay_term = prev_rep / rnd
                if d >= self.truth_threshold:
                    new_rep = prev_rep + d + decay_term
                else:
                    new_rep = prev_rep + d - np.exp(-(1 - d)) * decay_term

            self.reputations[cid] = float(np.clip(new_rep, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Trust indicator — Eq. 4-6 from Chuprov et al.
    # (Identical to V1)
    # ------------------------------------------------------------------
    def _update_trust(self, truth_values: dict[int, float]) -> None:
        beta = self.smoothing_beta

        for cid, d in truth_values.items():
            R = self.reputations[cid]
            raw_trust = np.sqrt(R**2 + d**2) - np.sqrt((1 - R)**2 + (1 - d)**2)
            prev_trust = self.trust_scores.get(cid, 0.0)
            smoothed = beta * raw_trust + (1 - beta) * prev_trust
            # Smooth mapping from [-sqrt(2), sqrt(2)] to [0, 1]
            _SQRT2 = 1.4142135623730951
            normalized = (smoothed + _SQRT2) / (2 * _SQRT2)
            self.trust_scores[cid] = float(np.clip(normalized, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Client selection — threshold-based on trust
    # (Identical to V1)
    # ------------------------------------------------------------------
    def _select_clients(
        self, results: list[tuple[int, FitRes]], server_round: int,
    ) -> list[tuple[int, FitRes]]:
        if server_round <= self.warmup_rounds:
            return list(results)

        selected = []
        for cid, fit_res in results:
            trust = self.trust_scores.get(cid, 0.0)
            if trust >= self.trust_exclusion_threshold:
                selected.append((cid, fit_res))

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

    # ------------------------------------------------------------------
    # Main aggregation entry point
    # ------------------------------------------------------------------
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

        # Step 1-2: compute per-layer cosine truth, update reputation
        truth_values = self._compute_truth_values(all_weights)
        self._update_reputations(truth_values, server_round=server_round)

        # Step 3: compute trust indicator
        self._update_trust(truth_values)

        # Step 4: select clients based on trust threshold
        selected = self._select_clients(results, server_round)

        # Filter out clients with NaN/Inf parameters
        selected = [
            (cid, fr) for cid, fr in selected
            if all(bool(np.all(np.isfinite(layer))) for layer in fr.parameters)
        ]

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

        # Store global params preserving layer structure for next round's deltas
        self.global_params_layers = [a.copy() for a in aggregated]

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
    """Build a ReputationV2Strategy from plugin params."""
    truth_threshold = kwargs.pop("truth_threshold", PARAMS["truth_threshold"]["default"])
    trust_exclusion_threshold = kwargs.pop(
        "trust_exclusion_threshold",
        PARAMS["trust_exclusion_threshold"]["default"],
    )
    initial_reputation = kwargs.pop("initial_reputation", PARAMS["initial_reputation"]["default"])
    warmup_rounds = kwargs.pop("warmup_rounds", PARAMS["warmup_rounds"]["default"])
    smoothing_beta = kwargs.pop("smoothing_beta", PARAMS["smoothing_beta"]["default"])
    trim_fraction = kwargs.pop("trim_fraction", PARAMS["trim_fraction"]["default"])
    layer_score_percentile = kwargs.pop("layer_score_percentile", PARAMS["layer_score_percentile"]["default"])
    # Consume legacy params if passed
    kwargs.pop("selection_fraction", None)
    return ReputationV2Strategy(
        num_clients=num_clients,
        truth_threshold=truth_threshold,
        trust_exclusion_threshold=trust_exclusion_threshold,
        initial_reputation=initial_reputation,
        warmup_rounds=warmup_rounds,
        smoothing_beta=smoothing_beta,
        trim_fraction=trim_fraction,
        layer_score_percentile=layer_score_percentile,
        initial_parameters=initial_parameters,
        **kwargs,
    )
