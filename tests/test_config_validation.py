import pytest

from simulation.runner import SimulationConfig, AttackConfig, _validate_config


class TestConfigValidation:
    def test_valid_config_passes(self):
        """Default config with minimal overrides should not raise."""
        config = SimulationConfig(
            model_name="cnn", dataset_name="mnist",
            num_clients=5, num_rounds=2, local_epochs=1,
            strategies=["fedavg"], seed=42,
        )
        _validate_config(config)  # should not raise

    def test_too_few_clients_raises(self):
        config = SimulationConfig(
            model_name="cnn", dataset_name="mnist",
            num_clients=1, num_rounds=2, local_epochs=1,
        )
        with pytest.raises(ValueError, match="num_clients"):
            _validate_config(config)

    def test_zero_learning_rate_raises(self):
        config = SimulationConfig(
            model_name="cnn", dataset_name="mnist",
            num_clients=5, num_rounds=2, local_epochs=1,
            learning_rate=0,
        )
        with pytest.raises(ValueError, match="learning_rate"):
            _validate_config(config)

    def test_negative_learning_rate_raises(self):
        config = SimulationConfig(
            model_name="cnn", dataset_name="mnist",
            num_clients=5, num_rounds=2, local_epochs=1,
            learning_rate=-0.01,
        )
        with pytest.raises(ValueError, match="learning_rate"):
            _validate_config(config)

    def test_malicious_fraction_too_high_raises(self):
        config = SimulationConfig(
            model_name="cnn", dataset_name="mnist",
            num_clients=5, num_rounds=2, local_epochs=1,
            attack=AttackConfig(
                attack_type="label_flipping",
                malicious_fraction=1.0,
            ),
        )
        with pytest.raises(ValueError, match="malicious_fraction"):
            _validate_config(config)

    def test_zero_rounds_raises(self):
        config = SimulationConfig(
            model_name="cnn", dataset_name="mnist",
            num_clients=5, num_rounds=0, local_epochs=1,
        )
        with pytest.raises(ValueError, match="num_rounds"):
            _validate_config(config)
