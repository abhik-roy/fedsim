import pytest
from simulation.runner import _validate_compatibility, SimulationConfig, AttackConfig


class TestCompatibilityValidation:
    def test_compatible_config_no_errors(self):
        config = SimulationConfig(
            model_name="cnn", dataset_name="mnist",
            strategies=["fedavg"],
            attack=AttackConfig(),
        )
        errors, warnings = _validate_compatibility(config)
        assert len(errors) == 0

    def test_model_task_mismatch_warns(self):
        """ResNet on a text classification dataset should warn."""
        config = SimulationConfig(
            model_name="resnet18", dataset_name="custom:AG News (4-class)",
            strategies=["fedavg"],
            attack=AttackConfig(),
        )
        errors, warnings = _validate_compatibility(config)
        assert any("not designed for" in w for w in warnings)

    def test_label_flipping_on_classification_no_error(self):
        """Label flipping on mnist (classification) should not error."""
        config = SimulationConfig(
            model_name="cnn", dataset_name="mnist",
            strategies=["fedavg"],
            attack=AttackConfig(attack_type="label_flipping"),
        )
        errors, warnings = _validate_compatibility(config)
        assert len(errors) == 0  # no errors for valid combo

    def test_compatible_custom_model_no_warning(self):
        """TextCNN with AG News should not warn."""
        config = SimulationConfig(
            model_name="custom:TextCNN",
            dataset_name="custom:AG News (4-class)",
            strategies=["fedavg"],
            attack=AttackConfig(),
        )
        errors, warnings = _validate_compatibility(config)
        assert len(errors) == 0

    def test_unknown_dataset_errors(self):
        config = SimulationConfig(
            model_name="cnn", dataset_name="nonexistent",
            strategies=["fedavg"],
            attack=AttackConfig(),
        )
        errors, warnings = _validate_compatibility(config)
        assert any("Unknown dataset" in e for e in errors)
