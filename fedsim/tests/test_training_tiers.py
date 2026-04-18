import pytest
from simulation.runner import SimulationConfig, AttackConfig, run_simulation

class TestTrainingTierDefault:
    def test_builtin_model_default_loop(self):
        config = SimulationConfig(
            model_name="cnn", dataset_name="mnist",
            num_clients=3, num_rounds=1, local_epochs=1,
            learning_rate=0.01, strategies=["fedavg"],
            attack=AttackConfig(),
        )
        results = run_simulation(config)
        assert len(results) == 1
        assert results[0].round_losses[-1] > 0
        assert results[0].round_accuracies[-1] > 0

class TestTrainingReturnFormat:
    def test_round_losses_populated(self):
        config = SimulationConfig(
            model_name="cnn", dataset_name="mnist",
            num_clients=3, num_rounds=2, local_epochs=1,
            learning_rate=0.01, strategies=["fedavg"],
            attack=AttackConfig(),
        )
        results = run_simulation(config)
        r = results[0]
        assert len(r.round_losses) == 3  # round 0 + 2 rounds
        assert len(r.round_accuracies) == 3
        assert all(isinstance(v, float) for v in r.round_losses)


class TestTextCNNIntegration:
    def test_textcnn_ag_news_full_simulation(self):
        """Full simulation with TextCNN + AG News + plugin_params."""
        config = SimulationConfig(
            model_name="custom:TextCNN",
            dataset_name="custom:AG News (4-class)",
            num_clients=3, num_rounds=1, local_epochs=1,
            learning_rate=0.01, strategies=["fedavg"],
            attack=AttackConfig(),
            plugin_params={"models": {"embed_dim": 64}},
        )
        results = run_simulation(config)
        assert len(results) == 1
        assert isinstance(results[0].round_losses[-1], float)
        assert results[0].round_losses[-1] > 0

    def test_builtin_model_backward_compat(self):
        """Built-in models continue to work without changes."""
        config = SimulationConfig(
            model_name="cnn", dataset_name="mnist",
            num_clients=3, num_rounds=2, local_epochs=1,
            learning_rate=0.01, strategies=["fedavg", "krum"],
            attack=AttackConfig(attack_type="label_flipping", malicious_fraction=0.3),
        )
        results = run_simulation(config)
        assert len(results) == 2
        for r in results:
            assert len(r.round_losses) == 3
            assert all(isinstance(v, float) for v in r.round_losses)
            assert all(isinstance(v, float) for v in r.round_accuracies)
