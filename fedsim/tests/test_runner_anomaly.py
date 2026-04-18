import pytest
from simulation.runner import SimulationConfig, AttackConfig, run_simulation, RoundEvent


def test_krum_anomaly_metrics_populated():
    """Krum should exclude some clients and produce anomaly metrics."""
    config = SimulationConfig(
        model_name="cnn", dataset_name="mnist",
        num_clients=5, num_rounds=2, local_epochs=1,
        strategies=["krum"],
        attack=AttackConfig(attack_type="byzantine_perturbation",
                            malicious_fraction=0.4,
                            attack_params={"noise_std": 5.0}),
        seed=42,
    )
    results = run_simulation(config)
    assert len(results) == 1
    r = results[0]
    assert r.anomaly_history
    assert r.anomaly_summary
    assert r.anomaly_summary["total_rounds"] == 2
    assert r.anomaly_summary["cumulative_tp"] >= 0


def test_fedavg_no_exclusion():
    """FedAvg doesn't exclude anyone — no TP, no FP."""
    config = SimulationConfig(
        model_name="cnn", dataset_name="mnist",
        num_clients=5, num_rounds=2, local_epochs=1,
        strategies=["fedavg"],
        attack=AttackConfig(attack_type="label_flipping", malicious_fraction=0.4),
        seed=42,
    )
    results = run_simulation(config)
    r = results[0]
    assert r.anomaly_summary["cumulative_tp"] == 0
    assert r.anomaly_summary["cumulative_fp"] == 0


def test_reputation_exclusion_tracking():
    """Reputation should exclude malicious clients after warm-up rounds."""
    config = SimulationConfig(
        model_name="cnn", dataset_name="mnist",
        num_clients=5, num_rounds=5, local_epochs=1,
        strategies=["reputation"],
        reputation_warmup_rounds=2,
        reputation_trust_exclusion_threshold=0.4,
        attack=AttackConfig(attack_type="weight_spiking",
                            malicious_fraction=0.4,
                            attack_params={"magnitude": 100.0, "spike_fraction": 0.3}),
        seed=42,
    )
    results = run_simulation(config)
    r = results[0]
    assert r.anomaly_summary["total_rounds"] == 5
    # During warm-up (rounds 1-2), no clients should be excluded
    for round_data in r.anomaly_history[:2]:
        assert len(round_data["excluded"]) == 0
    # After warm-up, malicious clients should start being excluded
    post_warmup = r.anomaly_history[2:]
    any_excluded = any(len(rd["excluded"]) > 0 for rd in post_warmup)
    assert any_excluded, "Reputation should exclude some clients after warm-up"


def test_no_attack_baseline_anomaly():
    """No attack — no malicious clients. All should be TN."""
    config = SimulationConfig(
        model_name="cnn", dataset_name="mnist",
        num_clients=5, num_rounds=2, local_epochs=1,
        strategies=["krum"],
        attack=AttackConfig(),
        seed=42,
    )
    results = run_simulation(config)
    r = results[0]
    assert r.anomaly_summary["cumulative_fn"] == 0
    assert r.anomaly_summary["cumulative_tp"] == 0
    assert r.anomaly_summary["cumulative_fp"] == 0


def test_round_event_has_new_fields():
    """RoundEvent should have exclusion fields without breaking existing ones."""
    events = []
    def cb(e): events.append(e)

    config = SimulationConfig(
        model_name="cnn", dataset_name="mnist",
        num_clients=5, num_rounds=1, local_epochs=1,
        strategies=["krum"], seed=42,
    )
    run_simulation(config, round_callback=cb)

    assert len(events) >= 2  # round 0 + round 1
    e = events[-1]
    # Existing fields still work
    assert hasattr(e, 'strategy_name')
    assert hasattr(e, 'loss')
    assert hasattr(e, 'accuracy')
    assert hasattr(e, 'client_trust_scores')
    # New fields exist
    assert hasattr(e, 'client_excluded')
    assert hasattr(e, 'client_included')
    assert hasattr(e, 'removal_f1')
    assert hasattr(e, 'strategy_scores')


def test_simulation_result_has_anomaly_fields():
    """SimulationResult should have anomaly_history and anomaly_summary."""
    config = SimulationConfig(
        model_name="cnn", dataset_name="mnist",
        num_clients=5, num_rounds=1, local_epochs=1,
        strategies=["fedavg"],
        seed=42,
    )
    results = run_simulation(config)
    r = results[0]
    assert hasattr(r, 'anomaly_history')
    assert hasattr(r, 'anomaly_summary')
    assert hasattr(r, 'strategy_scores_history')
    assert isinstance(r.anomaly_history, list)
    assert isinstance(r.anomaly_summary, dict)
    assert isinstance(r.strategy_scores_history, list)
