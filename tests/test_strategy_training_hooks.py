"""Tests for strategy-side client training hooks.

Validates that strategy plugins can provide train_step/fit overrides
that are used when no model plugin provides its own training override.
"""
import types
import pytest
import torch
import numpy as np
from fl_core import FedAvg
from simulation.runner import (
    SimulationConfig, AttackConfig, run_simulation, _get_strategy,
)


def _make_strategy_plugin(has_train_step=False, has_fit=False):
    """Create a mock strategy plugin module with optional train_step/fit."""
    mod = types.ModuleType("mock_strategy_plugin")
    mod.NAME = "MockStrategy"
    mod.call_log = {"train_step": False, "fit": False, "kwargs": {}}

    def build(initial_parameters, num_clients, num_malicious=0, **kwargs):
        fedavg_keys = {
            "fraction_fit", "fraction_evaluate", "min_fit_clients",
            "min_evaluate_clients", "min_available_clients",
            "initial_parameters",
        }
        filtered = {k: v for k, v in kwargs.items() if k in fedavg_keys}
        return FedAvg(initial_parameters=initial_parameters, **filtered)

    mod.build = build

    if has_train_step:
        def train_step(model, batch, optimizer, device, **kwargs):
            mod.call_log["train_step"] = True
            mod.call_log["kwargs"] = kwargs
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = torch.nn.functional.cross_entropy(output, labels)
            loss.backward()
            optimizer.step()
            return {"loss": loss.item()}
        mod.train_step = train_step

    if has_fit:
        def fit(model, dataloader, optimizer, device, local_epochs, **kwargs):
            mod.call_log["fit"] = True
            model.train()
            last_loss = 0.0
            for epoch in range(local_epochs):
                for batch in dataloader:
                    images, labels = batch
                    images, labels = images.to(device), labels.to(device)
                    optimizer.zero_grad()
                    output = model(images)
                    loss = torch.nn.functional.cross_entropy(output, labels)
                    loss.backward()
                    optimizer.step()
                    last_loss = loss.item()
            return {"loss": last_loss}
        mod.fit = fit

    return mod


def _register_mock_strategy(monkeypatch, plugin_mod):
    """Patch discover_plugins to return our mock strategy plugin."""
    import plugins as plugins_module

    original_discover = plugins_module.discover_plugins

    def mock_discover(plugin_type):
        if plugin_type == "strategies":
            return {"MockStrategy": plugin_mod}
        return original_discover(plugin_type)

    monkeypatch.setattr("plugins.discover_plugins", mock_discover)


class TestStrategyPluginModuleAttached:
    def test_custom_strategy_has_plugin_module(self, monkeypatch):
        """_get_strategy attaches _fedsim_plugin_module to custom strategies."""
        plugin = _make_strategy_plugin(has_train_step=True)
        _register_mock_strategy(monkeypatch, plugin)

        from models import SimpleCNN
        model = SimpleCNN(1, 10, 28)
        params = [v.cpu().numpy() for v in model.state_dict().values()]
        initial = params

        strategy = _get_strategy("custom:MockStrategy", initial, num_clients=3)
        assert hasattr(strategy, '_fedsim_plugin_module')
        assert strategy._fedsim_plugin_module is plugin
        assert hasattr(strategy._fedsim_plugin_module, 'train_step')

    def test_builtin_strategy_no_plugin_module(self):
        """Built-in strategies do NOT have _fedsim_plugin_module."""
        from models import SimpleCNN
        model = SimpleCNN(1, 10, 28)
        params = [v.cpu().numpy() for v in model.state_dict().values()]
        initial = params

        strategy = _get_strategy("fedavg", initial, num_clients=3)
        assert getattr(strategy, '_fedsim_plugin_module', None) is None


class TestStrategyTrainStepDispatch:
    def test_strategy_train_step_called(self, monkeypatch):
        """Strategy plugin train_step is called when no model plugin overrides."""
        plugin = _make_strategy_plugin(has_train_step=True)
        _register_mock_strategy(monkeypatch, plugin)

        config = SimulationConfig(
            model_name="cnn", dataset_name="mnist",
            num_clients=3, num_rounds=1, local_epochs=1,
            learning_rate=0.01, strategies=["custom:MockStrategy"],
            attack=AttackConfig(),
        )
        results = run_simulation(config)
        assert len(results) == 1
        assert results[0].round_losses[-1] > 0
        assert plugin.call_log["train_step"] is True

    def test_strategy_fit_called(self, monkeypatch):
        """Strategy plugin fit is called when no model plugin overrides."""
        plugin = _make_strategy_plugin(has_fit=True)
        _register_mock_strategy(monkeypatch, plugin)

        config = SimulationConfig(
            model_name="cnn", dataset_name="mnist",
            num_clients=3, num_rounds=1, local_epochs=1,
            learning_rate=0.01, strategies=["custom:MockStrategy"],
            attack=AttackConfig(),
        )
        results = run_simulation(config)
        assert len(results) == 1
        assert plugin.call_log["fit"] is True

    def test_strategy_kwargs_forwarded(self, monkeypatch):
        """Strategy plugin_params flow to strategy train_step as kwargs."""
        plugin = _make_strategy_plugin(has_train_step=True)
        _register_mock_strategy(monkeypatch, plugin)

        config = SimulationConfig(
            model_name="cnn", dataset_name="mnist",
            num_clients=3, num_rounds=1, local_epochs=1,
            learning_rate=0.01, strategies=["custom:MockStrategy"],
            attack=AttackConfig(),
            plugin_params={"strategies": {"mu": 0.5, "custom_flag": True}},
        )
        results = run_simulation(config)
        assert plugin.call_log["kwargs"].get("mu") == 0.5
        assert plugin.call_log["kwargs"].get("custom_flag") is True


class TestModelPriorityOverStrategy:
    def test_model_train_step_beats_strategy_train_step(self, monkeypatch):
        """Model plugin train_step takes priority over strategy plugin train_step."""
        strat_plugin = _make_strategy_plugin(has_train_step=True)
        _register_mock_strategy(monkeypatch, strat_plugin)

        config = SimulationConfig(
            model_name="custom:RegressionMLP",
            dataset_name="custom:California Housing",
            num_clients=3, num_rounds=1, local_epochs=1,
            learning_rate=0.01, strategies=["custom:MockStrategy"],
            attack=AttackConfig(),
        )
        results = run_simulation(config)
        assert len(results) == 1
        assert strat_plugin.call_log["train_step"] is False

    def test_model_train_step_beats_strategy_fit(self, monkeypatch):
        """Model plugin train_step (Tier 2a) beats strategy plugin fit (Tier 2b)."""
        strat_plugin = _make_strategy_plugin(has_fit=True)
        _register_mock_strategy(monkeypatch, strat_plugin)

        config = SimulationConfig(
            model_name="custom:RegressionMLP",
            dataset_name="custom:California Housing",
            num_clients=3, num_rounds=1, local_epochs=1,
            learning_rate=0.01, strategies=["custom:MockStrategy"],
            attack=AttackConfig(),
        )
        results = run_simulation(config)
        assert len(results) == 1
        assert strat_plugin.call_log["fit"] is False


class TestBuiltinStrategyUnaffected:
    def test_fedavg_still_works(self):
        """Built-in FedAvg + built-in CNN still uses Tier 3 default loop."""
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
