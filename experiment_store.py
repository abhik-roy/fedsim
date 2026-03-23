"""Save and load experiment results for comparison."""
import json
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
        return float(obj)
    if isinstance(obj, set):
        return sorted(obj)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    if isinstance(obj, float) and (obj != obj or obj == float('inf') or obj == float('-inf')):
        return None
    raise TypeError(f"Not serializable: {type(obj)}")


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

    with open(filepath, "w") as f:
        json.dump(experiment, f, default=_serializer, indent=2)

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
    with open(filepath) as f:
        return json.load(f)


def delete_experiment(filepath):
    """Delete a saved experiment file."""
    if os.path.exists(filepath):
        os.remove(filepath)


def _config_summary(config):
    """One-line summary of config for display."""
    model = config.get("model_name", "?")
    dataset = config.get("dataset_name", "?")
    clients = config.get("num_clients", "?")
    rounds = config.get("num_rounds", "?")
    return f"{model} | {dataset} | {clients}C | {rounds}R"
