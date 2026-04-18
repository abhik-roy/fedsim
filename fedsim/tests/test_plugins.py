import types
import pytest
from plugins import discover_plugins, get_plugin_choices, get_plugin_module, clear_cache, discover_all_plugins, PLUGIN_TYPES


class TestDiscoverPlugins:
    def setup_method(self):
        clear_cache()

    def test_valid_plugin_returns_module(self):
        plugins = discover_plugins("models")
        for name, mod in plugins.items():
            if not name.startswith("[Error:"):
                assert isinstance(mod, types.ModuleType)

    def test_error_plugin_stores_error_dict(self):
        plugins = discover_plugins("models")
        for name, val in plugins.items():
            if name.startswith("[Error:"):
                assert isinstance(val, dict)
                assert "error" in val
                assert "file" in val

    def test_get_plugin_choices_excludes_errors(self):
        choices = get_plugin_choices("models")
        for name in choices:
            assert not name.startswith("[Error:")

    def test_get_plugin_module_returns_none_for_errors(self):
        result = get_plugin_module("models", "[Error: nonexistent]")
        assert result is None

    def test_get_plugin_module_returns_none_for_missing(self):
        result = get_plugin_module("models", "does_not_exist_xyz")
        assert result is None


class TestDiscoverAllPlugins:
    def setup_method(self):
        clear_cache()

    def test_returns_all_plugin_types(self):
        summary = discover_all_plugins()
        for pt in PLUGIN_TYPES:
            assert pt in summary
            assert "loaded" in summary[pt]
            assert "errors" in summary[pt]

    def test_loaded_contains_tuples(self):
        summary = discover_all_plugins()
        for pt in PLUGIN_TYPES:
            for item in summary[pt]["loaded"]:
                assert len(item) == 2
                assert isinstance(item[1], types.ModuleType)

    def test_plugin_types_constant(self):
        assert "datasets" in PLUGIN_TYPES
        assert "models" in PLUGIN_TYPES
        assert "strategies" in PLUGIN_TYPES
        assert "losses" in PLUGIN_TYPES
        assert "optimizers" in PLUGIN_TYPES
        assert "metrics" in PLUGIN_TYPES
        assert "schedulers" in PLUGIN_TYPES
