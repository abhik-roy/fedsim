import os
import json
import tempfile
import pytest


def test_html_title_escaped():
    from report_html import generate_html_report
    from simulation.runner import SimulationConfig, SimulationResult
    config = SimulationConfig(model_name='<script>alert("xss")</script>')
    result = SimulationResult(
        strategy_name="fedavg", round_losses=[1.0], round_accuracies=[0.5],
        total_time=1.0,
    )
    html = generate_html_report(config, [result])
    assert '<script>alert' not in html
    assert '&lt;script&gt;' in html


def test_experiment_store_atomic_write():
    from experiment_store import save_experiment, _STORE_DIR
    from simulation.runner import SimulationConfig, SimulationResult
    config = SimulationConfig()
    result = SimulationResult(
        strategy_name="fedavg", round_losses=[1.0], round_accuracies=[0.5],
        total_time=1.0,
    )
    filepath = save_experiment(config, [result], name="test_atomic")
    assert os.path.exists(filepath)
    # No .tmp file should remain
    assert not os.path.exists(filepath + ".tmp")
    with open(filepath) as f:
        data = json.load(f)
    assert data["name"] == "test_atomic"
    os.remove(filepath)


def test_experiment_delete_path_validation():
    from experiment_store import delete_experiment
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as f:
        f.write(b'{"test": true}')
        outside_path = f.name
    try:
        with pytest.raises(ValueError, match="outside store"):
            delete_experiment(outside_path)
        assert os.path.exists(outside_path)
    finally:
        os.remove(outside_path)


def test_plugin_cache_clears_sys_modules():
    import sys
    from plugins import discover_plugins, clear_cache
    # Discover strategies to populate cache
    discover_plugins("strategies")
    # Check some custom module is in sys.modules
    custom_keys_before = [k for k in sys.modules if k.startswith("custom.")]
    assert len(custom_keys_before) > 0, "Expected custom modules after discovery"
    clear_cache()
    custom_keys_after = [k for k in sys.modules if k.startswith("custom.")]
    assert len(custom_keys_after) == 0, "clear_cache should remove custom modules from sys.modules"
