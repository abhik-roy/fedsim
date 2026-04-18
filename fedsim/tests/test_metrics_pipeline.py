import pytest
from simulation.runner import SimulationConfig, AttackConfig, run_simulation, RoundEvent


class TestMetricsPipeline:
    def test_custom_metrics_field_is_dict(self):
        config = SimulationConfig(
            model_name="cnn", dataset_name="mnist",
            num_clients=3, num_rounds=1, local_epochs=1,
            learning_rate=0.01, strategies=["fedavg"],
            attack=AttackConfig(), active_metrics=[],
        )
        results = run_simulation(config)
        for r in results:
            assert isinstance(r.custom_metrics, dict)

    def test_round_event_has_custom_metrics(self):
        events = []
        config = SimulationConfig(
            model_name="cnn", dataset_name="mnist",
            num_clients=3, num_rounds=1, local_epochs=1,
            learning_rate=0.01, strategies=["fedavg"],
            attack=AttackConfig(), active_metrics=[],
        )
        run_simulation(config, round_callback=lambda e: events.append(e))
        for e in events:
            assert hasattr(e, "custom_metrics")

    def test_inactive_metrics_produce_empty_dict(self):
        """With active_metrics=[], custom_metrics should be empty even if plugins exist."""
        config = SimulationConfig(
            model_name="cnn", dataset_name="mnist",
            num_clients=3, num_rounds=2, local_epochs=1,
            learning_rate=0.01, strategies=["fedavg"],
            attack=AttackConfig(), active_metrics=[],
        )
        results = run_simulation(config)
        for r in results:
            assert r.custom_metrics == {}
