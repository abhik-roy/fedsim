"""Tests for three-tier evaluation dispatch in the runner."""

import pytest
from simulation.runner import SimulationConfig, AttackConfig, run_simulation


class TestEvalTierDefault:
    def test_builtin_eval_returns_valid_metrics(self):
        config = SimulationConfig(
            model_name="cnn",
            dataset_name="mnist",
            num_clients=3,
            num_rounds=1,
            local_epochs=1,
            learning_rate=0.01,
            strategies=["fedavg"],
            attack=AttackConfig(),
        )
        results = run_simulation(config)
        r = results[0]
        assert r.round_losses[-1] >= 0.0
        assert 0.0 <= r.round_accuracies[-1] <= 1.0

    def test_textcnn_uses_default_eval(self):
        config = SimulationConfig(
            model_name="custom:TextCNN",
            dataset_name="custom:AG News (4-class)",
            num_clients=3,
            num_rounds=1,
            local_epochs=1,
            learning_rate=0.01,
            strategies=["fedavg"],
            attack=AttackConfig(),
        )
        results = run_simulation(config)
        r = results[0]
        assert len(r.round_losses) >= 2
        assert isinstance(r.round_losses[-1], float)
