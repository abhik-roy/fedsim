"""Tests for parallel client training."""
import pytest
from simulation.runner import SimulationConfig, AttackConfig, run_simulation


class TestSequentialParity:
    def test_sequential_default(self):
        """max_parallel_clients=1 produces valid results (default behavior)."""
        config = SimulationConfig(
            model_name="cnn", dataset_name="mnist",
            num_clients=3, num_rounds=2, local_epochs=1,
            learning_rate=0.01, strategies=["fedavg"],
            attack=AttackConfig(), seed=42,
            max_parallel_clients=1,
        )
        results = run_simulation(config)
        assert len(results) == 1
        assert results[0].round_losses[-1] > 0
        assert results[0].round_accuracies[-1] > 0

    def test_sequential_with_attacks(self):
        """Sequential mode with attacks still works."""
        config = SimulationConfig(
            model_name="cnn", dataset_name="mnist",
            num_clients=5, num_rounds=2, local_epochs=1,
            learning_rate=0.01, strategies=["fedavg", "krum"],
            attack=AttackConfig(attack_type="label_flipping", malicious_fraction=0.4),
            seed=42, max_parallel_clients=1,
        )
        results = run_simulation(config)
        assert len(results) == 2


class TestParallelExecution:
    def test_parallel_completes(self):
        """Parallel mode completes without errors."""
        config = SimulationConfig(
            model_name="cnn", dataset_name="mnist",
            num_clients=5, num_rounds=2, local_epochs=1,
            learning_rate=0.01, strategies=["fedavg"],
            attack=AttackConfig(), seed=42,
            max_parallel_clients=3,
        )
        results = run_simulation(config)
        assert len(results) == 1
        assert results[0].round_losses[-1] > 0
        assert results[0].round_accuracies[-1] > 0

    def test_parallel_with_attacks(self):
        """Parallel mode with attacks and multiple strategies."""
        config = SimulationConfig(
            model_name="cnn", dataset_name="mnist",
            num_clients=5, num_rounds=2, local_epochs=1,
            learning_rate=0.01, strategies=["fedavg", "krum"],
            attack=AttackConfig(attack_type="label_flipping", malicious_fraction=0.4),
            seed=42, max_parallel_clients=3,
        )
        results = run_simulation(config)
        assert len(results) == 2
        for r in results:
            assert len(r.round_losses) == 3  # round 0 + 2 rounds

    def test_parallel_max_exceeds_clients(self):
        """max_parallel_clients > num_clients is clamped safely."""
        config = SimulationConfig(
            model_name="cnn", dataset_name="mnist",
            num_clients=3, num_rounds=1, local_epochs=1,
            learning_rate=0.01, strategies=["fedavg"],
            attack=AttackConfig(), seed=42,
            max_parallel_clients=10,
        )
        results = run_simulation(config)
        assert len(results) == 1

    def test_parallel_results_reasonable(self):
        """Parallel results are in reasonable range (not corrupted)."""
        config = SimulationConfig(
            model_name="cnn", dataset_name="mnist",
            num_clients=5, num_rounds=3, local_epochs=1,
            learning_rate=0.01, strategies=["fedavg"],
            attack=AttackConfig(), seed=42,
            max_parallel_clients=5,
        )
        results = run_simulation(config)
        acc = results[0].round_accuracies[-1]
        assert 0.0 < acc <= 1.0
        loss = results[0].round_losses[-1]
        assert 0.0 < loss < 10.0
