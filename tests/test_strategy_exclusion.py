import json
import numpy as np
import pytest
from fl_core import FitRes

from strategies.krum import KrumStrategy
from custom.strategies.reputation import ReputationStrategy
from strategies.bulyan import BulyanStrategy
from strategies.rfa import RFAStrategy
from strategies.trimmed_mean import TrimmedMean
from strategies.median import MedianStrategy


def _make_results(num_clients, param_dim=100, seed=42):
    rng = np.random.default_rng(seed)
    results = []
    for i in range(num_clients):
        params = [rng.normal(size=(param_dim,)).astype(np.float32)]
        fit_res = FitRes(parameters=params, num_examples=100)
        results.append((i, fit_res))
    return results


def _make_initial_params(param_dim=100):
    return [np.zeros(param_dim, dtype=np.float32)]


def test_krum_returns_exclusion_metadata():
    init_params = _make_initial_params()
    strategy = KrumStrategy(num_malicious=2, multi_krum=True,
                            initial_parameters=init_params,
                            min_fit_clients=5, min_available_clients=5,
                            min_evaluate_clients=5)
    results = _make_results(5)
    _, metrics = strategy.aggregate_fit(1, results, [])
    assert "included_clients" in metrics
    assert "excluded_clients" in metrics
    included = json.loads(metrics["included_clients"])
    excluded = json.loads(metrics["excluded_clients"])
    assert len(included) + len(excluded) == 5
    assert len(included) > 0


def test_reputation_returns_exclusion_metadata():
    init_params = _make_initial_params()
    strategy = ReputationStrategy(num_clients=5, selection_fraction=0.6,
                                  initial_parameters=init_params,
                                  min_fit_clients=5, min_available_clients=5,
                                  min_evaluate_clients=5)
    results = _make_results(5)
    _, metrics = strategy.aggregate_fit(1, results, [])
    assert "included_clients" in metrics
    assert "excluded_clients" in metrics
    included = json.loads(metrics["included_clients"])
    excluded = json.loads(metrics["excluded_clients"])
    assert len(included) == 3  # 60% of 5 = 3
    assert len(excluded) == 2


def test_bulyan_returns_exclusion_metadata():
    init_params = _make_initial_params()
    strategy = BulyanStrategy(num_malicious=1,
                              initial_parameters=init_params,
                              min_fit_clients=5, min_available_clients=5,
                              min_evaluate_clients=5)
    results = _make_results(5)
    _, metrics = strategy.aggregate_fit(1, results, [])
    assert "included_clients" in metrics
    assert "excluded_clients" in metrics


def test_rfa_returns_weight_metadata():
    init_params = _make_initial_params()
    strategy = RFAStrategy(initial_parameters=init_params,
                           min_fit_clients=5, min_available_clients=5,
                           min_evaluate_clients=5)
    results = _make_results(5)
    _, metrics = strategy.aggregate_fit(1, results, [])
    assert "client_scores" in metrics
    scores = json.loads(metrics["client_scores"])
    assert len(scores) == 5


def test_trimmed_mean_returns_metadata():
    init_params = _make_initial_params()
    strategy = TrimmedMean(beta=0.2, initial_parameters=init_params,
                           min_fit_clients=5, min_available_clients=5,
                           min_evaluate_clients=5)
    results = _make_results(5)
    _, metrics = strategy.aggregate_fit(1, results, [])
    assert "trim_count" in metrics


def test_median_returns_metadata():
    init_params = _make_initial_params()
    strategy = MedianStrategy(initial_parameters=init_params,
                              min_fit_clients=5, min_available_clients=5,
                              min_evaluate_clients=5)
    results = _make_results(5)
    _, metrics = strategy.aggregate_fit(1, results, [])
    assert "aggregation_method" in metrics


def test_fedavg_returns_no_exclusion_metadata():
    """FedAvg should not exclude anyone — TP and FP should both be 0."""
    from simulation.runner import SimulationConfig, AttackConfig, run_simulation
    config = SimulationConfig(
        model_name="cnn", dataset_name="mnist",
        num_clients=3, num_rounds=1, local_epochs=1,
        strategies=["fedavg"],
        attack=AttackConfig(),
        seed=42,
    )
    results = run_simulation(config)
    r = results[0]
    assert r.anomaly_summary["cumulative_tp"] == 0
    assert r.anomaly_summary["cumulative_fp"] == 0
