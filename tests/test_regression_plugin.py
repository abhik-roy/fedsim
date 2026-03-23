import pytest
import torch

from simulation.runner import SimulationConfig, AttackConfig, run_simulation


class TestCaliforniaHousingPlugin:
    def test_dataset_loads(self):
        from custom.datasets.california_housing import load
        train, test = load()
        assert len(train) > 0
        assert len(test) > 0
        # Check features are float tensors
        x, y = train[0]
        assert x.dtype == torch.float32
        assert y.dtype == torch.float32
        # Check targets attribute exists for partitioning
        assert hasattr(train, "targets")
        assert len(train.targets) == len(train)

    def test_model_builds(self):
        from custom.models.regression_mlp import build
        model = build({"input_size": 8, "num_classes": 1, "task_type": "regression"})
        assert model is not None
        # Test forward pass
        x = torch.randn(4, 8)
        out = model(x)
        assert out.shape == (4,)

    def test_full_simulation(self):
        config = SimulationConfig(
            model_name="custom:RegressionMLP",
            dataset_name="custom:California Housing",
            num_clients=3, num_rounds=2, local_epochs=1,
            learning_rate=0.001, strategies=["fedavg"],
            attack=AttackConfig(),
        )
        results = run_simulation(config)
        assert len(results) == 1
        r = results[0]
        assert len(r.round_losses) == 3  # init + 2 rounds
        assert all(isinstance(v, float) for v in r.round_losses)
        # Accuracy should be 0.0 (regression task)
        assert all(v == 0.0 for v in r.round_accuracies)
        # MAE should be in custom_metrics under eval/mae
        assert "eval/mae" in r.custom_metrics
