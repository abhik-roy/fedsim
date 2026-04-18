import numpy as np
import pytest
from fl_core import FitRes


def _make_results(weights_list, cids):
    return [(cid, FitRes(parameters=w, num_examples=10)) for cid, w in zip(cids, weights_list)]


def test_reputation_v1_arbitrary_client_ids():
    from custom.strategies.reputation import ReputationStrategy
    cids = [10, 25, 43, 57, 99]
    rng = np.random.default_rng(42)
    weights = [[rng.normal(size=(8,)).astype(np.float32)] for _ in range(5)]
    strategy = ReputationStrategy(num_clients=5, initial_parameters=weights[0])
    results = _make_results(weights, cids)
    agg, metrics = strategy.aggregate_fit(1, results, [])
    assert agg is not None
    trust = strategy.get_trust_scores()
    for cid in cids:
        assert cid in trust, f"Client {cid} missing from trust scores"


def test_reputation_v2_arbitrary_client_ids():
    from custom.strategies.reputation_v2 import ReputationV2Strategy
    cids = [10, 25, 43, 57, 99]
    rng = np.random.default_rng(42)
    weights = [[rng.normal(size=(8,)).astype(np.float32)] for _ in range(5)]
    strategy = ReputationV2Strategy(num_clients=5, initial_parameters=weights[0])
    results = _make_results(weights, cids)
    agg, metrics = strategy.aggregate_fit(1, results, [])
    assert agg is not None
    trust = strategy.get_trust_scores()
    for cid in cids:
        assert cid in trust


def test_reputation_trust_smooth_mapping():
    from custom.strategies.reputation import ReputationStrategy
    strategy = ReputationStrategy(num_clients=3, initial_parameters=[np.zeros(4)])
    rng = np.random.default_rng(0)
    weights = [[rng.normal(size=(4,)).astype(np.float32)] for _ in range(3)]
    results = _make_results(weights, [0, 1, 2])
    strategy.aggregate_fit(1, results, [])
    trust = strategy.get_trust_scores()
    vals = list(trust.values())
    assert all(0.0 <= v <= 1.0 for v in vals)
    # With smooth mapping, values should NOT all be 0 or 1
    assert not all(v == 0.0 for v in vals), "All trust scores are 0 — smooth mapping not working"


def test_reputation_nan_input_handled():
    from custom.strategies.reputation import ReputationStrategy
    rng = np.random.default_rng(42)
    weights = [[rng.normal(size=(4,)).astype(np.float32)] for _ in range(5)]
    weights[2] = [np.array([np.nan] * 4, dtype=np.float32)]
    strategy = ReputationStrategy(num_clients=5, initial_parameters=weights[0])
    results = _make_results(weights, [0, 1, 2, 3, 4])
    agg, metrics = strategy.aggregate_fit(1, results, [])
    assert agg is not None
    assert all(np.isfinite(layer).all() for layer in agg)


def test_reputation_v2_nan_input_handled():
    from custom.strategies.reputation_v2 import ReputationV2Strategy
    rng = np.random.default_rng(42)
    weights = [[rng.normal(size=(4,)).astype(np.float32)] for _ in range(5)]
    weights[2] = [np.array([np.nan] * 4, dtype=np.float32)]
    strategy = ReputationV2Strategy(num_clients=5, initial_parameters=weights[0])
    results = _make_results(weights, [0, 1, 2, 3, 4])
    agg, metrics = strategy.aggregate_fit(1, results, [])
    assert agg is not None
    assert all(np.isfinite(layer).all() for layer in agg)
