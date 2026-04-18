"""Auto-discovery of custom plugins from the custom/ directory."""

import importlib
import importlib.util
import sys
import types
from pathlib import Path


_plugin_cache = {}

PLUGIN_TYPES = ("datasets", "models", "strategies", "losses", "optimizers", "metrics", "schedulers")


def discover_plugins(plugin_type: str) -> dict[str, object]:
    """Scan custom/{plugin_type}/ for .py plugins, return {display_name: module}.

    plugin_type: "datasets", "models", or "strategies"
    Results are cached per session.
    """
    if plugin_type in _plugin_cache:
        return _plugin_cache[plugin_type]

    plugins = {}
    plugin_dir = Path(__file__).parent / "custom" / plugin_type

    if not plugin_dir.exists():
        _plugin_cache[plugin_type] = plugins
        return plugins

    for py_file in sorted(plugin_dir.glob("*.py")):
        if py_file.name.startswith("_") or py_file.name == "__init__.py":
            continue
        # Skip symlinks to prevent loading files outside the plugin directory
        if py_file.is_symlink():
            print(f"Warning: Skipping symlinked plugin {py_file}")
            continue
        try:
            module_name = f"custom.{plugin_type}.{py_file.stem}"
            if module_name in sys.modules:
                module = sys.modules[module_name]
            else:
                spec = importlib.util.spec_from_file_location(module_name, py_file)
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)

            name = getattr(module, "NAME", py_file.stem)
            plugins[name] = module
        except Exception as e:
            # Store error for UI warning, don't crash
            plugins[f"[Error: {py_file.stem}]"] = {"error": str(e), "file": str(py_file)}
            print(f"Warning: Failed to load plugin {py_file}: {e}")

    _plugin_cache[plugin_type] = plugins
    return plugins


def get_plugin_choices(plugin_type: str) -> dict[str, str]:
    """Return {display_name: key} dict for UI dropdowns.

    Keys are prefixed with 'custom:' to distinguish from built-in options.
    """
    plugins = discover_plugins(plugin_type)
    return {name: f"custom:{name}" for name, mod in plugins.items() if isinstance(mod, types.ModuleType)}


def get_plugin_module(plugin_type: str, display_name: str):
    """Get the module object for a specific plugin."""
    plugins = discover_plugins(plugin_type)
    result = plugins.get(display_name)
    return result if isinstance(result, types.ModuleType) else None


def clear_cache():
    """Clear the plugin discovery cache. Call when plugins may have changed."""
    _plugin_cache.clear()
    to_remove = [k for k in sys.modules if k.startswith("custom.")]
    for k in to_remove:
        del sys.modules[k]


def discover_all_plugins() -> dict[str, dict]:
    """Return summary across all plugin types with loaded modules and errors."""
    summary = {}
    for pt in PLUGIN_TYPES:
        plugins = discover_plugins(pt)
        loaded = []
        errors = []
        for name, val in plugins.items():
            if isinstance(val, types.ModuleType):
                loaded.append((name, val))
            elif isinstance(val, dict) and "error" in val:
                errors.append((val.get("file", name), val["error"]))
        summary[pt] = {"loaded": loaded, "errors": errors}
    return summary
