import pytest
import torch
from simulation.runner import (
    SimulationConfig, AttackConfig, run_simulation,
    _build_scheduler,
)


class TestBuildScheduler:
    def test_none_returns_none(self):
        opt = torch.optim.SGD([torch.zeros(1)], lr=0.1)
        assert _build_scheduler(opt, "none") is None

    def test_step_lr(self):
        opt = torch.optim.SGD([torch.zeros(1)], lr=0.1)
        sched = _build_scheduler(opt, "step_lr", step_size=2, gamma=0.5)
        assert sched is not None
        assert isinstance(sched, torch.optim.lr_scheduler.StepLR)

    def test_cosine_annealing(self):
        opt = torch.optim.SGD([torch.zeros(1)], lr=0.1)
        sched = _build_scheduler(opt, "cosine_annealing", T_max=10)
        assert sched is not None

    def test_exponential(self):
        opt = torch.optim.SGD([torch.zeros(1)], lr=0.1)
        sched = _build_scheduler(opt, "exponential", gamma=0.9)
        assert sched is not None

    def test_unknown_raises(self):
        opt = torch.optim.SGD([torch.zeros(1)], lr=0.1)
        with pytest.raises(ValueError):
            _build_scheduler(opt, "unknown_scheduler")


class TestClientSampling:
    def test_fraction_fit_in_config(self):
        config = SimulationConfig(fraction_fit=0.5, fraction_evaluate=0.8)
        assert config.fraction_fit == 0.5
        assert config.fraction_evaluate == 0.8

    def test_simulation_with_fraction_fit(self):
        config = SimulationConfig(
            model_name="cnn", dataset_name="mnist",
            num_clients=5, num_rounds=2, local_epochs=1,
            learning_rate=0.01, strategies=["fedavg"],
            attack=AttackConfig(),
            fraction_fit=0.6,
        )
        results = run_simulation(config)
        assert len(results) == 1
        assert len(results[0].round_losses) == 3


class TestLRSchedulerIntegration:
    def test_simulation_with_step_lr(self):
        config = SimulationConfig(
            model_name="cnn", dataset_name="mnist",
            num_clients=3, num_rounds=2, local_epochs=2,
            learning_rate=0.01, strategies=["fedavg"],
            attack=AttackConfig(),
            lr_scheduler="step_lr",
            lr_scheduler_params={"step_size": 1, "gamma": 0.5},
        )
        results = run_simulation(config)
        assert len(results) == 1
        assert results[0].round_losses[-1] > 0

    def test_simulation_with_cosine(self):
        config = SimulationConfig(
            model_name="cnn", dataset_name="mnist",
            num_clients=3, num_rounds=1, local_epochs=2,
            learning_rate=0.01, strategies=["fedavg"],
            attack=AttackConfig(),
            lr_scheduler="cosine_annealing",
            lr_scheduler_params={"T_max": 2},
        )
        results = run_simulation(config)
        assert len(results) == 1
