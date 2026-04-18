"""Save and load experiment results for comparison."""
import json
import math
import os
import dataclasses
from datetime import datetime
from pathlib import Path
import numpy as np

_STORE_DIR = os.path.join(os.path.expanduser("~"), ".fedsim", "experiments")


def _serializer(obj):
    """JSON serializer for numpy and dataclass types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64)):
        v = float(obj)
        if v != v or v == float('inf') or v == float('-inf'):
            return None
        return v
    if isinstance(obj, set):
        return sorted(obj)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    raise TypeError(f"Not serializable: {type(obj)}")


def _sanitize_floats(obj):
    """Recursively replace NaN/Inf floats with None so json.dump(allow_nan=False) succeeds."""
    if isinstance(obj, float):
        return None if math.isnan(obj) or math.isinf(obj) else obj
    if isinstance(obj, dict):
        return {k: _sanitize_floats(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_floats(v) for v in obj]
    if isinstance(obj, tuple):
        return [_sanitize_floats(v) for v in obj]
    return obj


def save_experiment(config, results, name=None):
    """Save experiment results to JSON. Returns the file path."""
    os.makedirs(_STORE_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if name:
        safe_name = name.replace(" ", "_").replace("/", "_")
        filename = f"{timestamp}_{safe_name}.json"
    else:
        filename = f"{timestamp}.json"

    filepath = os.path.join(_STORE_DIR, filename)

    experiment = {
        "timestamp": timestamp,
        "name": name or timestamp,
        "config": dataclasses.asdict(config),
        "results": [],
    }

    for r in results:
        result_dict = {
            "strategy_name": r.strategy_name,
            "round_losses": r.round_losses,
            "round_accuracies": r.round_accuracies,
            "total_time": r.total_time,
            "custom_metrics": r.custom_metrics,
            "anomaly_summary": r.anomaly_summary,
        }
        experiment["results"].append(result_dict)

    tmp_path = filepath + ".tmp"
    try:
        with open(tmp_path, "w") as f:
            # Pre-sanitize NaN/Inf so allow_nan=False produces valid RFC-8259 JSON
            # readable by strict parsers (jq, JavaScript, pandas).
            json.dump(_sanitize_floats(experiment), f, default=_serializer,
                      indent=2, allow_nan=False)
        os.replace(tmp_path, filepath)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    return filepath


def list_experiments():
    """List saved experiments. Returns list of dicts with name, filepath, timestamp, config_summary."""
    if not os.path.exists(_STORE_DIR):
        return []

    experiments = []
    for fname in sorted(os.listdir(_STORE_DIR), reverse=True):
        if fname.endswith(".json"):
            filepath = os.path.join(_STORE_DIR, fname)
            try:
                with open(filepath) as f:
                    data = json.load(f)
                experiments.append({
                    "name": data.get("name", fname),
                    "filepath": filepath,
                    "timestamp": data.get("timestamp", ""),
                    "config_summary": _config_summary(data.get("config", {})),
                })
            except (json.JSONDecodeError, KeyError):
                continue
    return experiments


def load_experiment(filepath):
    """Load an experiment from JSON. Returns the raw dict."""
    store_dir = os.path.realpath(_STORE_DIR)
    real_path = os.path.realpath(filepath)
    if not real_path.startswith(store_dir + os.sep) and real_path != store_dir:
        raise ValueError(f"Path outside store directory: {filepath}")
    with open(filepath) as f:
        return json.load(f)


def delete_experiment(filepath):
    """Delete a saved experiment file. Validates path is within store directory."""
    store_dir = os.path.realpath(_STORE_DIR)
    real_path = os.path.realpath(filepath)
    if not real_path.startswith(store_dir + os.sep):
        raise ValueError(f"Path outside store directory: {filepath}")
    if os.path.exists(real_path):
        os.remove(real_path)


def _config_summary(config):
    """One-line summary of config for display."""
    model = config.get("model_name", "?")
    dataset = config.get("dataset_name", "?")
    clients = config.get("num_clients", "?")
    rounds = config.get("num_rounds", "?")
    return f"{model} | {dataset} | {clients}C | {rounds}R"
