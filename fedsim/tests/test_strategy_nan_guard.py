import numpy as np
import pytest
from fl_core import FitRes
from strategies.utils import filter_nan_clients
from strategies.krum import KrumStrategy
from strategies.bulyan import BulyanStrategy
from strategies.trimmed_mean import TrimmedMean
from strategies.median import MedianStrategy
from strategies.rfa import RFAStrategy


def test_filter_nan_removes_bad_clients():
    weights = [
        [np.array([1.0, 2.0]), np.array([3.0])],
        [np.array([np.nan, 2.0]), np.array([3.0])],
        [np.array([4.0, 5.0]), np.array([6.0])],
    ]
    cids = [10, 20, 30]
    clean_w, clean_cids = filter_nan_clients(weights, cids)
    assert len(clean_w) == 2
    assert clean_cids == [10, 30]


def test_filter_inf_removes_bad_clients():
    weights = [
        [np.array([1.0, 2.0])],
        [np.array([np.inf, 2.0])],
    ]
    cids = [0, 1]
    clean_w, clean_cids = filter_nan_clients(weights, cids)
    assert len(clean_w) == 1
    assert clean_cids == [0]


def test_filter_nan_all_bad_returns_empty():
    weights = [[np.array([np.nan])]]
    cids = [0]
    clean_w, clean_cids = filter_nan_clients(weights, cids)
    assert len(clean_w) == 0
    assert clean_cids == []


def test_filter_nan_all_clean_passthrough():
    weights = [
        [np.array([1.0, 2.0])],
        [np.array([3.0, 4.0])],
    ]
    cids = [5, 6]
    clean_w, clean_cids = filter_nan_clients(weights, cids)
    assert len(clean_w) == 2
    assert clean_cids == [5, 6]


# ---------------------------------------------------------------------------
# Integration tests: NaN guard per strategy
# ---------------------------------------------------------------------------

def _make_results(weights_list, cids=None):
    if cids is None:
        cids = list(range(len(weights_list)))
    return [(cid, FitRes(parameters=w, num_examples=10)) for cid, w in zip(cids, weights_list)]


def _good_weights(n=5, dim=4):
    rng = np.random.default_rng(42)
    return [[rng.normal(size=(dim,)).astype(np.float32)] for _ in range(n)]


def test_krum_nan_client_excluded():
    weights = _good_weights(5)
    weights[2] = [np.array([np.nan, np.nan, np.nan, np.nan], dtype=np.float32)]
    results = _make_results(weights)
    agg, metrics = KrumStrategy(num_malicious=1).aggregate_fit(1, results, [])
    assert agg is not None
    assert all(np.isfinite(layer).all() for layer in agg)


def test_bulyan_nan_client_excluded():
    weights = _good_weights(8)
    weights[3] = [np.array([np.nan] * 4, dtype=np.float32)]
    results = _make_results(weights)
    agg, metrics = BulyanStrategy(num_malicious=1).aggregate_fit(1, results, [])
    assert agg is not None
    assert all(np.isfinite(layer).all() for layer in agg)


def test_bulyan_has_client_scores():
    weights = _good_weights(8)
    results = _make_results(weights)
    agg, metrics = BulyanStrategy(num_malicious=1).aggregate_fit(1, results, [])
    assert "client_scores" in metrics


def test_trimmed_mean_nan_client_excluded():
    weights = _good_weights(5)
    weights[1] = [np.array([np.inf] * 4, dtype=np.float32)]
    results = _make_results(weights)
    agg, metrics = TrimmedMean(beta=0.1).aggregate_fit(1, results, [])
    assert agg is not None
    assert all(np.isfinite(layer).all() for layer in agg)


def test_median_nan_client_excluded():
    weights = _good_weights(5)
    weights[0] = [np.array([np.nan] * 4, dtype=np.float32)]
    results = _make_results(weights)
    agg, metrics = MedianStrategy().aggregate_fit(1, results, [])
    assert agg is not None
    assert all(np.isfinite(layer).all() for layer in agg)


def test_rfa_nan_client_excluded():
    weights = _good_weights(5)
    weights[4] = [np.array([np.nan] * 4, dtype=np.float32)]
    results = _make_results(weights)
    agg, metrics = RFAStrategy().aggregate_fit(1, results, [])
    assert agg is not None
    assert all(np.isfinite(layer).all() for layer in agg)


def test_all_nan_returns_none():
    weights = [[np.array([np.nan] * 4, dtype=np.float32)]] * 3
    results = _make_results(weights)
    agg, _ = KrumStrategy(num_malicious=1).aggregate_fit(1, results, [])
    assert agg is None
