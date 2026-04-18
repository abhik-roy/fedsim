"""FEDSIM Experiment API — programmatic experiment scripting."""
import json
import os
import time
import math
from dataclasses import asdict, fields

from simulation.runner import SimulationConfig, SimulationResult, run_simulation


def _json_serializer(obj):
    """JSON serializer for numpy types and dataclasses."""
    import numpy as np
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        v = float(obj)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if hasattr(obj, '__dataclass_fields__'):
        return asdict(obj)
    raise TypeError(f"Not serializable: {type(obj)}")


def _save_json(data, path):
    """Atomic JSON write."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, default=_json_serializer, indent=2)
    os.replace(tmp, path)


class ExperimentResults:
    """Thin wrapper around experiment results with convenience accessors."""

    def __init__(self, data=None):
        self._data = data or {}  # {name: list of SimulationResult-like dicts or objects}

    def __getitem__(self, name):
        return self._data[name]

    def __contains__(self, name):
        return name in self._data

    def __len__(self):
        return len(self._data)

    @property
    def names(self):
        return list(self._data.keys())

    def items(self):
        return self._data.items()

    def final_accuracy(self, name, strategy_idx=0):
        try:
            r = self._data[name][strategy_idx]
            accs = r.round_accuracies if hasattr(r, 'round_accuracies') else r.get('round_accuracies', [])
            return accs[-1] if accs else float('nan')
        except (IndexError, KeyError):
            return float('nan')

    def final_loss(self, name, strategy_idx=0):
        try:
            r = self._data[name][strategy_idx]
            losses = r.round_losses if hasattr(r, 'round_losses') else r.get('round_losses', [])
            return losses[-1] if losses else float('nan')
        except (IndexError, KeyError):
            return float('nan')


class Experiment:
    """Named collection of simulation runs with progress and checkpointing."""

    def __init__(self, name):
        self.name = name
        self._runs = []  # list of (name, SimulationConfig)
        self._run_names = set()
        self._results = ExperimentResults()

    def add_run(self, name, config):
        if name in self._run_names:
            raise ValueError(f"Duplicate run name: {name!r}")
        self._runs.append((name, config))
        self._run_names.add(name)

    def run(self, checkpoint_path=None):
        # Load checkpoint if exists
        if checkpoint_path and os.path.exists(checkpoint_path):
            loaded = self.load(checkpoint_path)
            for name in loaded.names:
                if name not in self._results:
                    self._results._data[name] = loaded[name]

        # Count pending
        pending = [(name, cfg) for name, cfg in self._runs if name not in self._results]
        total = len(pending)
        if total == 0:
            print(f"All {len(self._runs)} runs already completed.")
            return self._results

        print(f"\n{'='*60}")
        print(f"  {self.name}")
        print(f"  {total} runs to execute ({len(self._results)} cached)")
        print(f"{'='*60}\n")

        for i, (name, config) in enumerate(pending):
            print(f"[{i+1}/{total}] {name}: running...", end=" ", flush=True)
            t0 = time.time()
            try:
                sim_results = run_simulation(config)
                elapsed = time.time() - t0
                # Get primary accuracy for display
                acc = sim_results[0].round_accuracies[-1] if sim_results and sim_results[0].round_accuracies else float('nan')
                print(f"done ({elapsed:.1f}s, acc={acc:.4f})")
                self._results._data[name] = sim_results
            except KeyboardInterrupt:
                print("\nInterrupted. Saving checkpoint...")
                if checkpoint_path:
                    self.save(checkpoint_path)
                raise
            except Exception as e:
                print(f"FAILED ({e})")
                raise

            if checkpoint_path:
                self.save(checkpoint_path)

        return self._results

    def export_configs(self, path):
        """Export run configs to JSON — for loading into the dashboard."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        runs = []
        for name, config in self._runs:
            cfg_dict = asdict(config)
            # AttackConfig is nested — already handled by asdict
            runs.append({"name": name, "config": cfg_dict})
        data = {"name": self.name, "type": "experiment_configs", "runs": runs}
        _save_json(data, path)

    @staticmethod
    def load_configs(path):
        """Load run configs from JSON. Returns list of (name, SimulationConfig)."""
        with open(path) as f:
            data = json.load(f)
        runs = []
        for entry in data.get("runs", []):
            name = entry["name"]
            cfg = entry["config"]
            # Reconstruct AttackConfig
            attack_data = cfg.pop("attack", {})
            from simulation.runner import AttackConfig
            attack = AttackConfig(**attack_data) if attack_data else AttackConfig()
            config = SimulationConfig(attack=attack, **cfg)
            runs.append((name, config))
        return data.get("name", "Loaded Experiment"), runs

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        # Convert SimulationResult objects to dicts for serialization
        serializable = {}
        for name, results_list in self._results.items():
            serializable[name] = [
                asdict(r) if hasattr(r, '__dataclass_fields__') else r
                for r in results_list
            ]
        data = {"name": self.name, "results": serializable}
        _save_json(data, path)

    @staticmethod
    def load(path):
        with open(path) as f:
            data = json.load(f)
        raw = data.get("results", {})
        # Reconstruct SimulationResult objects from dicts
        reconstructed = {}
        for name, results_list in raw.items():
            sim_results = []
            for r in results_list:
                if isinstance(r, dict):
                    # Build SimulationResult from dict — use field defaults for missing keys
                    kwargs = {}
                    valid_fields = {f.name for f in fields(SimulationResult)}
                    for k, v in r.items():
                        if k in valid_fields:
                            kwargs[k] = v
                    sim_results.append(SimulationResult(**kwargs))
                else:
                    sim_results.append(r)
            reconstructed[name] = sim_results
        return ExperimentResults(reconstructed)
