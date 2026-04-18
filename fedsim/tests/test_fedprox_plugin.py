"""Tests for the FedProx strategy plugin."""
import copy
import torch
import torch.nn.functional as F
import pytest
from fl_core import FedAvg
from simulation.runner import SimulationConfig, AttackConfig, run_simulation


class TestFedProxBuild:
    def test_build_returns_fedavg_instance(self):
        """build() returns a FedAvg-compatible strategy."""
        from custom.strategies.fedprox import build
        from models import SimpleCNN
        model = SimpleCNN(1, 10, 28)
        params = [v.cpu().numpy() for v in model.state_dict().values()]
        initial = params
        strategy = build(initial_parameters=initial, num_clients=5,
                         fraction_fit=1.0, fraction_evaluate=1.0,
                         min_fit_clients=2, min_evaluate_clients=2,
                         min_available_clients=2, mu=0.01)
        assert isinstance(strategy, FedAvg)

    def test_build_filters_mu(self):
        """build() does not pass mu to FedAvg (would cause error)."""
        from custom.strategies.fedprox import build
        from models import SimpleCNN
        model = SimpleCNN(1, 10, 28)
        params = [v.cpu().numpy() for v in model.state_dict().values()]
        initial = params
        strategy = build(initial_parameters=initial, num_clients=5,
                         fraction_fit=1.0, fraction_evaluate=1.0,
                         min_fit_clients=2, min_evaluate_clients=2,
                         min_available_clients=2, mu=1.0)
        assert strategy is not None


class TestFedProxTrainStep:
    @pytest.fixture
    def setup(self):
        """Set up a simple model, batch, and optimizer for train_step tests."""
        from models import SimpleCNN
        model = SimpleCNN(1, 10, 28)
        device = torch.device("cpu")
        model.to(device)
        images = torch.randn(4, 1, 28, 28)
        labels = torch.tensor([0, 1, 2, 3])
        batch = (images, labels)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        return model, batch, optimizer, device

    def test_train_step_returns_loss(self, setup):
        from custom.strategies.fedprox import train_step
        model, batch, optimizer, device = setup
        result = train_step(model, batch, optimizer, device, mu=0.01)
        assert "loss" in result
        assert isinstance(result["loss"], float)
        assert result["loss"] > 0

    def test_train_step_returns_accuracy(self, setup):
        from custom.strategies.fedprox import train_step
        model, batch, optimizer, device = setup
        result = train_step(model, batch, optimizer, device, mu=0.01)
        assert "accuracy" in result
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_proximal_term_increases_loss(self, setup):
        """With mu>0, loss is higher than mu=0 after params diverge from global."""
        from custom.strategies.fedprox import train_step
        model, batch, optimizer, device = setup
        model_a = copy.deepcopy(model)
        opt_a = torch.optim.SGD(model_a.parameters(), lr=0.01)
        model_b = copy.deepcopy(model)
        opt_b = torch.optim.SGD(model_b.parameters(), lr=0.01)
        # First step: w == w_global, so proximal is 0 for both
        train_step(model_a, batch, opt_a, device, mu=0.0)
        train_step(model_b, batch, opt_b, device, mu=10.0)
        # Second step: params have diverged from snapshot
        result_no_prox = train_step(model_a, batch, opt_a, device, mu=0.0)
        result_with_prox = train_step(model_b, batch, opt_b, device, mu=10.0)
        assert result_with_prox["loss"] > result_no_prox["loss"]

    def test_mu_zero_no_proximal_added(self):
        """With mu=0 on first call, proximal term is zero (w == w_global).

        Uses inference mode to disable dropout for deterministic comparison.
        """
        from custom.strategies.fedprox import train_step
        from models import SimpleCNN
        device = torch.device("cpu")
        # Build two identical models in inference mode (disables dropout)
        model_a = SimpleCNN(1, 10, 28).to(device)
        model_a.requires_grad_(True)
        for m in model_a.modules():
            if isinstance(m, (torch.nn.Dropout, torch.nn.Dropout2d)):
                m.p = 0.0  # Disable dropout for deterministic test
        model_b = copy.deepcopy(model_a)
        images = torch.randn(4, 1, 28, 28)
        labels = torch.tensor([0, 1, 2, 3])
        batch = (images, labels)
        opt_a = torch.optim.SGD(model_a.parameters(), lr=0.01)
        result_mu0 = train_step(model_a, batch, opt_a, device, mu=0.0)
        opt_b = torch.optim.SGD(model_b.parameters(), lr=0.01)
        result_mu10 = train_step(model_b, batch, opt_b, device, mu=10.0)
        # First call: w == w_global so proximal=0 regardless of mu
        assert abs(result_mu0["loss"] - result_mu10["loss"]) < 1e-5

    def test_global_params_snapshot_created(self, setup):
        """After train_step, model has _fedprox_global_params attribute."""
        from custom.strategies.fedprox import train_step
        model, batch, optimizer, device = setup
        assert not hasattr(model, '_fedprox_global_params')
        train_step(model, batch, optimizer, device, mu=0.01)
        assert hasattr(model, '_fedprox_global_params')

    def test_snapshot_frozen_after_step(self, setup):
        """Global params snapshot does not change after optimizer.step()."""
        from custom.strategies.fedprox import train_step
        model, batch, optimizer, device = setup
        train_step(model, batch, optimizer, device, mu=0.01)
        snapshot_after_step1 = [p.clone() for p in model._fedprox_global_params]
        train_step(model, batch, optimizer, device, mu=0.01)
        for s1, s2 in zip(snapshot_after_step1, model._fedprox_global_params):
            assert torch.equal(s1, s2)


class TestFedProxFullSimulation:
    def test_fedprox_simulation_completes(self):
        """Full simulation with FedProx strategy plugin."""
        config = SimulationConfig(
            model_name="cnn", dataset_name="mnist",
            num_clients=3, num_rounds=2, local_epochs=1,
            learning_rate=0.01, strategies=["custom:FedProx"],
            attack=AttackConfig(),
            plugin_params={"strategies": {"mu": 0.1}},
        )
        results = run_simulation(config)
        assert len(results) == 1
        assert results[0].round_losses[-1] > 0
        assert results[0].round_accuracies[-1] > 0
        assert len(results[0].round_losses) == 3

    def test_fedprox_mu_flows_through(self):
        """mu value from plugin_params reaches train_step."""
        config = SimulationConfig(
            model_name="cnn", dataset_name="mnist",
            num_clients=3, num_rounds=1, local_epochs=1,
            learning_rate=0.01, strategies=["custom:FedProx"],
            attack=AttackConfig(),
            plugin_params={"strategies": {"mu": 0.5}},
        )
        results = run_simulation(config)
        assert len(results) == 1
        assert results[0].round_losses[-1] > 0
