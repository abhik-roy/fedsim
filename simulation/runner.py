import copy
import time
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext

from fl_core import FedAvg, FitRes, Status, Code

from configs.defaults import (
    DATASET_INFO, ATTACK_CATEGORIES, GRADIENT_CLIP_MAX_NORM, CLIENT_EVAL_BATCH_LIMIT,
)
from data.loader import get_dataset
from data.partitioner import partition_dataset
from models import get_model
from strategies import KrumStrategy, MedianStrategy, TrimmedMean, BulyanStrategy, RFAStrategy
from attacks.data_poisoning import apply_label_flipping, apply_gaussian_noise, apply_token_replacement
from attacks.model_poisoning import (
    apply_weight_spiking,
    apply_gradient_scaling,
    apply_byzantine_perturbation,
)
import json
import types as _types_module
from anomaly.metrics import AnomalyMetrics


def _build_dataset_info(dataset_name: str) -> dict:
    """Assemble dataset_info dict from built-in DATASET_INFO or custom plugin attributes."""
    if dataset_name.startswith("custom:"):
        from plugins import get_plugin_module
        plugin_name = dataset_name.replace("custom:", "")
        mod = get_plugin_module("datasets", plugin_name)
        if mod is None:
            raise ValueError(f"Custom dataset plugin not found: {plugin_name}")
        _ch = getattr(mod, "INPUT_CHANNELS", 3)
        _sz = getattr(mod, "IMAGE_SIZE", 32)
        info = {
            "task_type": getattr(mod, "TASK_TYPE", "image_classification"),
            "num_classes": getattr(mod, "NUM_CLASSES", 10),
            "input_channels": _ch,
            "image_size": _sz,
            "input_size": getattr(mod, "INPUT_SIZE", _ch * _sz * _sz),
            "vocab_size": getattr(mod, "VOCAB_SIZE", None),
            "seq_length": getattr(mod, "SEQ_LENGTH", None),
        }
        # Merge custom keys without overwriting standard fields
        extra = getattr(mod, "DATASET_INFO", {})
        for k, v in extra.items():
            info.setdefault(k, v)
        return info
    else:
        info = {**DATASET_INFO[dataset_name]}
        info.setdefault("task_type", "image_classification")
        info.setdefault("vocab_size", None)
        info.setdefault("seq_length", None)
        return info


def _get_model_plugin(model_name: str):
    """Return the model plugin module if custom, else None."""
    if not model_name.startswith("custom:"):
        return None
    from plugins import get_plugin_module
    plugin_name = model_name.replace("custom:", "")
    return get_plugin_module("models", plugin_name)


def _get_batch_size(batch):
    """Extract batch size from tuple, dict, or tensor batches."""
    if isinstance(batch, (tuple, list)):
        return len(batch[0])
    elif isinstance(batch, dict):
        return len(next(iter(batch.values())))
    elif hasattr(batch, '__len__'):
        return len(batch)
    else:
        return 1


@dataclass
class AttackConfig:
    attack_type: str = "none"
    malicious_fraction: float = 0.2
    attack_params: dict = field(default_factory=dict)
    schedule_type: str = "static"
    attack_rounds: list[int] | None = None


@dataclass
class SimulationConfig:
    model_name: str = "resnet18"
    dataset_name: str = "cifar10"
    num_clients: int = 10
    num_rounds: int = 20
    local_epochs: int = 3
    learning_rate: float = 0.01
    partition_type: str = "non_iid"
    alpha: float = 0.5
    val_split: float = 0.1            # fraction of training data held out for server validation
    strategies: list[str] = field(default_factory=lambda: ["fedavg"])
    batch_size: int = 32
    seed: int = 42
    attack: AttackConfig = field(default_factory=AttackConfig)
    reputation_truth_threshold: float = 0.7
    reputation_selection_fraction: float = 0.6
    reputation_initial_reputation: float = 0.5
    # Training customization
    optimizer: str = "sgd"
    loss_function: str = "cross_entropy"
    weight_decay: float = 0.0
    eval_frequency: int = 1           # evaluate global model every N rounds
    # Flower-parity: client sampling
    fraction_fit: float = 1.0
    fraction_evaluate: float = 1.0
    min_fit_clients: int = 2
    min_evaluate_clients: int = 2
    min_available_clients: int = 2
    lr_scheduler: str = "none"
    lr_scheduler_params: dict = field(default_factory=dict)
    active_metrics: list[str] = field(default_factory=list)
    plugin_params: dict = field(default_factory=dict)
    max_parallel_clients: int = 1  # 1 = sequential (current behavior)
    # GPU acceleration (auto-disabled on CPU)
    use_amp: bool = False           # Mixed precision (float16) — ~1.5-2x on tensor core GPUs
    compile_model: bool = False     # torch.compile optimization — ~1.2-1.5x (PyTorch 2.0+)
    pin_memory: bool = False        # Pin DataLoader memory for faster CPU→GPU transfer


@dataclass
class SimulationResult:
    strategy_name: str
    round_losses: list[float]
    round_accuracies: list[float]
    total_time: float
    trust_history: dict[int, list[float]] = field(default_factory=dict)
    reputation_history: dict[int, list[float]] = field(default_factory=dict)
    final_client_params: dict[int, list[np.ndarray]] | None = None
    client_statuses_history: list[dict[int, str]] = field(default_factory=list)
    anomaly_history: list[dict] = field(default_factory=list)
    anomaly_summary: dict = field(default_factory=dict)
    strategy_scores_history: list[dict[int, float]] = field(default_factory=list)
    custom_metrics: dict = field(default_factory=dict)
    # Final test-set evaluation (held out during training)
    test_accuracy: float = 0.0
    test_loss: float = 0.0


@dataclass
class RoundEvent:
    """Emitted after each FL round completes."""
    strategy_name: str
    strategy_idx: int
    num_strategies: int
    round_num: int
    num_rounds: int
    loss: float
    accuracy: float
    loss_delta: float
    accuracy_delta: float
    client_statuses: dict[int, str]
    elapsed: float
    # Per-client metrics for this round
    client_trust_scores: dict[int, float] = field(default_factory=dict)
    client_accuracies: dict[int, float] = field(default_factory=dict)
    client_reputation_scores: dict[int, float] = field(default_factory=dict)
    client_excluded: set[int] = field(default_factory=set)
    client_included: set[int] = field(default_factory=set)
    removal_precision: float = 0.0
    removal_recall: float = 0.0
    removal_f1: float = 0.0
    strategy_scores: dict[int, float] = field(default_factory=dict)
    custom_metrics: dict = field(default_factory=dict)


@dataclass
class ClientTrainEvent:
    """Emitted when a single client finishes local training."""
    strategy_name: str
    round_num: int
    client_id: int
    num_clients: int
    is_malicious: bool
    attack_applied: bool
    client_loss: float
    client_accuracy: float = 0.0


def _validate_config(config: SimulationConfig):
    """Validate simulation configuration, raising ValueError for invalid settings."""
    if config.num_clients < 2:
        raise ValueError(f"num_clients must be >= 2, got {config.num_clients}")
    if config.num_rounds < 1:
        raise ValueError(f"num_rounds must be >= 1, got {config.num_rounds}")
    if config.local_epochs < 1:
        raise ValueError(f"local_epochs must be >= 1, got {config.local_epochs}")
    if config.learning_rate <= 0:
        raise ValueError(f"learning_rate must be > 0, got {config.learning_rate}")
    if config.batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {config.batch_size}")
    if config.eval_frequency <= 0:
        raise ValueError(f"eval_frequency must be > 0, got {config.eval_frequency}")
    if config.fraction_fit <= 0 or config.fraction_fit > 1:
        raise ValueError(f"fraction_fit must be in (0, 1], got {config.fraction_fit}")
    if config.fraction_evaluate <= 0 or config.fraction_evaluate > 1:
        raise ValueError(f"fraction_evaluate must be in (0, 1], got {config.fraction_evaluate}")

    atk = config.attack
    if atk.attack_type != "none":
        if not 0 < atk.malicious_fraction < 1:
            raise ValueError(f"malicious_fraction must be in (0, 1), got {atk.malicious_fraction}")
        if atk.attack_type == "gaussian_noise":
            snr = atk.attack_params.get("snr_db", 20.0)
            if snr <= 0:
                raise ValueError(f"SNR must be > 0 dB, got {snr}")
        elif atk.attack_type == "weight_spiking":
            mag = atk.attack_params.get("magnitude", 100.0)
            if mag <= 0:
                raise ValueError(f"Weight spiking magnitude must be > 0, got {mag}")
        elif atk.attack_type == "gradient_scaling":
            sf = atk.attack_params.get("scale_factor", 10.0)
            if sf <= 0:
                raise ValueError(f"Gradient scale_factor must be > 0, got {sf}")
        elif atk.attack_type == "byzantine_perturbation":
            ns = atk.attack_params.get("noise_std", 1.0)
            if ns <= 0:
                raise ValueError(f"Byzantine noise_std must be > 0, got {ns}")


def _validate_compatibility(config: SimulationConfig) -> tuple[list[str], list[str]]:
    """Pre-run compatibility validation.

    Returns (errors: list[str], warnings: list[str]).
    """
    from configs.defaults import MODEL_COMPATIBLE_TASKS, ATTACK_COMPATIBLE_TASKS
    errors = []
    warnings = []

    # Build dataset info for task type
    try:
        ds_info = _build_dataset_info(config.dataset_name)
    except (ValueError, KeyError):
        errors.append(f"Unknown dataset: {config.dataset_name}")
        return errors, warnings

    task_type = ds_info.get("task_type", "image_classification")
    num_classes = ds_info.get("num_classes")

    # Check model compatibility
    if config.model_name.startswith("custom:"):
        model_plugin = _get_model_plugin(config.model_name)
        if model_plugin:
            compat = getattr(model_plugin, "COMPATIBLE_TASKS", None)
            if compat and task_type not in compat:
                model_display = getattr(model_plugin, "NAME", config.model_name)
                warnings.append(
                    f"Model '{model_display}' is not designed for '{task_type}' tasks. "
                    f"Results may be incorrect."
                )
    else:
        compat = MODEL_COMPATIBLE_TASKS.get(config.model_name)
        if compat and task_type not in compat:
            warnings.append(
                f"Model '{config.model_name}' is not designed for '{task_type}' tasks. "
                f"Results may be incorrect."
            )

    # Check attack compatibility
    atk = config.attack.attack_type
    if atk != "none":
        # Label flipping with non-classification is an ERROR (will crash)
        if atk == "label_flipping" and (num_classes is None or "classification" not in task_type):
            errors.append(
                f"Label flipping requires integer class labels but dataset task type is "
                f"'{task_type}'. Change attack type or dataset."
            )
        # Other attack compat is a warning
        atk_compat = ATTACK_COMPATIBLE_TASKS.get(atk)
        if atk_compat and task_type not in atk_compat and atk != "label_flipping":
            warnings.append(
                f"Attack '{atk}' is designed for {atk_compat} tasks, "
                f"not '{task_type}'."
            )

    # Check custom model without overrides on non-classification
    if config.model_name.startswith("custom:") and "classification" not in task_type:
        model_plugin = _get_model_plugin(config.model_name)
        if model_plugin and not hasattr(model_plugin, "fit") and not hasattr(model_plugin, "train_step"):
            warnings.append(
                f"Custom model has no train_step/fit override for '{task_type}' task. "
                f"Default classification loop will be used."
            )

    # Model instantiation dry-run
    try:
        from models import get_model
        test_model = get_model(config.model_name, config.dataset_name,
                               **config.plugin_params.get("models", {}))
        del test_model
    except Exception as e:
        err_str = str(e)
        if "vocab" in err_str.lower() or "embedding" in err_str.lower():
            warnings.append(
                f"Model dry-run skipped -- dataset attributes not yet loaded. "
                f"Validation will occur at runtime."
            )
        else:
            errors.append(f"Model failed to build: {e}")

    return errors, warnings


def _get_strategy(name, initial_parameters, num_clients, num_malicious=0,
                  reputation_truth_threshold=0.7, reputation_selection_fraction=0.6,
                  reputation_initial_reputation=0.5, plugin_kwargs=None,
                  fraction_fit=1.0, fraction_evaluate=1.0,
                  min_fit_clients=2, min_evaluate_clients=2, min_available_clients=2):
    # Backward compat: "reputation" maps to the plugin
    if name == "reputation":
        name = "custom:Reputation"
        # Merge legacy config fields into plugin_kwargs
        plugin_kwargs = dict(plugin_kwargs or {})
        plugin_kwargs.setdefault("truth_threshold", reputation_truth_threshold)
        plugin_kwargs.setdefault("selection_fraction", reputation_selection_fraction)
        plugin_kwargs.setdefault("initial_reputation", reputation_initial_reputation)

    common = {
        "fraction_fit": fraction_fit,
        "fraction_evaluate": fraction_evaluate,
        "min_fit_clients": min(min_fit_clients, num_clients),
        "min_evaluate_clients": min(min_evaluate_clients, num_clients),
        "min_available_clients": min(min_available_clients, num_clients),
        "initial_parameters": initial_parameters,
    }
    if name == "fedavg":
        return FedAvg(**common)
    elif name == "trimmed_mean":
        beta = max(0.1, num_malicious / num_clients) if num_clients > 0 else 0.1
        return TrimmedMean(beta=beta, **common)
    elif name == "krum":
        return KrumStrategy(num_malicious=num_malicious, multi_krum=True, **common)
    elif name == "median":
        return MedianStrategy(**common)
    elif name == "bulyan":
        return BulyanStrategy(num_malicious=num_malicious, **common)
    elif name == "rfa":
        return RFAStrategy(**common)
    elif name.startswith("custom:"):
        from plugins import discover_plugins
        plugin_name = name.replace("custom:", "")
        plugins = discover_plugins("strategies")
        for pname, mod in plugins.items():
            if pname == plugin_name and isinstance(mod, _types_module.ModuleType):
                strategy = mod.build(
                    num_clients=num_clients,
                    num_malicious=num_malicious,
                    **common,
                    **(plugin_kwargs or {}),
                )
                strategy._fedsim_plugin_module = mod
                return strategy
        raise ValueError(f"Custom strategy plugin not found: {plugin_name}")
    else:
        raise ValueError(f"Unknown strategy: {name}")


def _select_malicious_clients(num_clients, malicious_fraction, seed):
    rng = np.random.default_rng(seed)
    num_malicious = max(1, int(num_clients * malicious_fraction))
    num_malicious = min(num_malicious, num_clients - 1)
    return set(rng.choice(num_clients, size=num_malicious, replace=False).tolist())


def _build_optimizer(model, optimizer_name, lr, weight_decay=0.0, **plugin_kwargs):
    """Create an optimizer instance from a name string or custom plugin."""
    if optimizer_name.startswith("custom:"):
        from plugins import discover_plugins
        plugin_name = optimizer_name.replace("custom:", "")
        plugins = discover_plugins("optimizers")
        for pname, mod in plugins.items():
            if pname == plugin_name and isinstance(mod, _types_module.ModuleType):
                return mod.build(model.parameters(), lr=lr, weight_decay=weight_decay, **plugin_kwargs)
        raise ValueError(f"Custom optimizer plugin not found: {plugin_name}")
    elif optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:  # default: sgd
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)


def _build_loss(loss_name, **plugin_kwargs):
    """Create a loss function instance from a name string or custom plugin."""
    if loss_name.startswith("custom:"):
        from plugins import discover_plugins
        plugin_name = loss_name.replace("custom:", "")
        plugins = discover_plugins("losses")
        for pname, mod in plugins.items():
            if pname == plugin_name and isinstance(mod, _types_module.ModuleType):
                return mod.build(**plugin_kwargs)
        raise ValueError(f"Custom loss plugin not found: {plugin_name}")
    elif loss_name == "bce_with_logits":
        return nn.BCEWithLogitsLoss()
    elif loss_name == "nll":
        return nn.NLLLoss()
    else:  # default: cross_entropy
        return nn.CrossEntropyLoss()


def _build_scheduler(optimizer, scheduler_name, local_epochs=1, **kwargs):
    """Create an LR scheduler from a name string or custom plugin.

    The scheduler steps per epoch during local client training.
    Returns None if scheduler_name is 'none'.
    """
    if scheduler_name == "none" or not scheduler_name:
        return None

    if scheduler_name.startswith("custom:"):
        from plugins import discover_plugins
        plugin_name = scheduler_name.replace("custom:", "")
        plugins = discover_plugins("schedulers")
        for pname, mod in plugins.items():
            if pname == plugin_name and isinstance(mod, _types_module.ModuleType):
                return mod.build(optimizer, **kwargs)
        raise ValueError(f"Custom scheduler plugin not found: {plugin_name}")
    elif scheduler_name == "step_lr":
        step_size = kwargs.get("step_size", 5)
        gamma = kwargs.get("gamma", 0.1)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_name == "cosine_annealing":
        T_max = kwargs.get("T_max", local_epochs)
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
    elif scheduler_name == "exponential":
        gamma = kwargs.get("gamma", 0.95)
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")


def _train_client(model, trainloader, local_epochs, lr, device,
                  optimizer_name="sgd", loss_name="cross_entropy", weight_decay=0.0,
                  optimizer_plugin_kwargs=None, loss_plugin_kwargs=None,
                  model_plugin=None, model_plugin_kwargs=None,
                  strategy_plugin=None, strategy_plugin_kwargs=None,
                  scheduler_name="none", scheduler_params=None,
                  use_amp=False):
    """Train a model on one client's data.

    Returns:
        (parameters: list[np.ndarray], metrics: dict[str, float])
        metrics always contains "loss". May contain "accuracy" and others.
    """
    model.to(device)

    # Abort early if model corrupted
    for p in model.parameters():
        if torch.isnan(p).any() or torch.isinf(p).any():
            return [v.cpu().numpy() for v in model.state_dict().values()], {"loss": float('inf')}

    optimizer = _build_optimizer(model, optimizer_name, lr, weight_decay,
                                 **(optimizer_plugin_kwargs or {}))
    scheduler = _build_scheduler(optimizer, scheduler_name, local_epochs, **(scheduler_params or {}))
    plugin_kwargs = model_plugin_kwargs or {}
    strat_kwargs = strategy_plugin_kwargs or {}

    if model_plugin and hasattr(model_plugin, "fit"):
        # Tier 1: model plugin full control — plugin manages epochs internally
        # Pass scheduler via kwargs so plugins can optionally use it
        model.train()
        fit_kwargs = {**plugin_kwargs}
        if scheduler is not None:
            fit_kwargs["scheduler"] = scheduler
        metrics = model_plugin.fit(model, trainloader, optimizer, device,
                                    local_epochs, **fit_kwargs)
    elif model_plugin and hasattr(model_plugin, "train_step"):
        # Tier 2a: model plugin per-step control, runner manages epoch/batch loops
        model.train()
        total_loss = 0.0
        batch_count = 0
        step_metrics = {"loss": 0.0}
        for epoch in range(local_epochs):
            for batch in trainloader:
                step_metrics = model_plugin.train_step(model, batch, optimizer,
                                                        device, **plugin_kwargs)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP_MAX_NORM)
                total_loss += step_metrics.get("loss", 0.0)
                batch_count += 1
            if scheduler is not None:
                scheduler.step()
        if batch_count > 0:
            step_metrics["loss"] = total_loss / batch_count
        metrics = step_metrics
    elif strategy_plugin and hasattr(strategy_plugin, "fit"):
        # Tier 2b: strategy plugin full control (NEW)
        model.train()
        fit_kwargs = {**strat_kwargs}
        if scheduler is not None:
            fit_kwargs["scheduler"] = scheduler
        metrics = strategy_plugin.fit(model, trainloader, optimizer, device,
                                       local_epochs, **fit_kwargs)
    elif strategy_plugin and hasattr(strategy_plugin, "train_step"):
        # Tier 2c: strategy plugin per-step control (NEW)
        model.train()
        total_loss = 0.0
        batch_count = 0
        step_metrics = {"loss": 0.0}
        for epoch in range(local_epochs):
            for batch in trainloader:
                step_metrics = strategy_plugin.train_step(model, batch, optimizer,
                                                           device, **strat_kwargs)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP_MAX_NORM)
                total_loss += step_metrics.get("loss", 0.0)
                batch_count += 1
            if scheduler is not None:
                scheduler.step()
        if batch_count > 0:
            step_metrics["loss"] = total_loss / batch_count
        metrics = step_metrics
    else:
        # Tier 3: default classification loop (existing code)
        # Warn if custom model uses default classification loop with non-classification task
        if model_plugin is not None:
            import warnings
            warnings.warn(
                f"Custom model plugin has no train_step/fit override. "
                f"Using default classification training loop."
            )
        model.train()
        criterion = _build_loss(loss_name, **(loss_plugin_kwargs or {}))
        last_loss = 0.0
        amp_enabled = use_amp and device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda") if amp_enabled else None
        for epoch in range(local_epochs):
            epoch_loss, batch_count = 0.0, 0
            for images, labels in trainloader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.autocast(device_type=device.type, enabled=amp_enabled):
                    loss = criterion(model(images), labels)
                if not torch.isfinite(loss):
                    params = [val.cpu().numpy() for val in model.state_dict().values()]
                    return params, {"loss": float('inf')}
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP_MAX_NORM)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP_MAX_NORM)
                    optimizer.step()
                epoch_loss += loss.item()
                batch_count += 1
            last_loss = epoch_loss / max(batch_count, 1)
            if scheduler is not None:
                scheduler.step()
        metrics = {"loss": last_loss}

    # Ensure "loss" key exists
    if "loss" not in metrics:
        import warnings
        warnings.warn("train_step/fit did not return 'loss' key. Using 0.0.")
        metrics["loss"] = 0.0

    updated_params = [val.cpu().numpy() for val in model.state_dict().values()]
    return updated_params, metrics


def _evaluate_client_accuracy(model, params, testloader, device,
                               model_plugin=None, model_plugin_kwargs=None):
    """Quick evaluation of a client's model on test data. Returns accuracy float."""
    _set_model_params(model, params)
    model.to(device)
    model.eval()
    kwargs = model_plugin_kwargs or {}

    if model_plugin and hasattr(model_plugin, "evaluate"):
        with torch.no_grad():
            metrics = model_plugin.evaluate(model, testloader, device, **kwargs)
        return metrics.get("accuracy", 0.0)
    elif model_plugin and hasattr(model_plugin, "eval_step"):
        all_metrics, total = [], 0
        with torch.no_grad():
            for i, batch in enumerate(testloader):
                if i >= CLIENT_EVAL_BATCH_LIMIT:
                    break
                bs = _get_batch_size(batch)
                step = model_plugin.eval_step(model, batch, device, **kwargs)
                all_metrics.append((step, bs))
                total += bs
        if total == 0:
            return 0.0
        return sum(m.get("accuracy", 0.0) * n for m, n in all_metrics) / total
    else:
        # Default classification eval
        correct, total = 0, 0
        with torch.no_grad():
            for i, (images, labels) in enumerate(testloader):
                if i >= CLIENT_EVAL_BATCH_LIMIT:
                    break
                images, labels = images.to(device), labels.to(device)
                correct += (model(images).argmax(1) == labels).sum().item()
                total += labels.size(0)
        return correct / total if total > 0 else 0.0


def _compute_trust_scores(client_params_list, global_params):
    """Compute trust score per client combining direction and magnitude signals.

    Uses a two-component trust metric:
    1. Cosine similarity: measures alignment of client delta with the median delta
       (catches directional attacks like label flipping)
    2. Normalized L2 distance: measures how far the client delta is from the median
       (catches magnitude attacks like weight spiking, Byzantine, gradient scaling)

    The final score is: 0.5 * cosine_component + 0.5 * distance_component
    Both components are mapped to [0, 1] where 1 = most trustworthy.

    This combined metric handles non-IID scenarios better than cosine alone,
    because a benign non-IID client may have a different direction but similar
    magnitude, while a malicious client differs in both direction AND magnitude.
    """
    global_flat = np.concatenate([p.flatten() for p in global_params])

    # Compute update deltas
    deltas = []
    for client_params in client_params_list:
        client_flat = np.concatenate([p.flatten() for p in client_params])
        deltas.append(client_flat - global_flat)
    deltas_arr = np.array(deltas)

    # Coordinate-wise median is robust to Byzantine outliers
    median_delta = np.median(deltas_arr, axis=0)
    median_norm = np.linalg.norm(median_delta)

    # Near convergence, all deltas are tiny — trust all clients
    if median_norm < 1e-10:
        return {cid: 1.0 for cid in range(len(client_params_list))}

    # Compute L2 distances from each client delta to the median delta
    l2_distances = np.array([np.linalg.norm(d - median_delta) for d in deltas])
    max_l2 = np.max(l2_distances)

    scores = {}
    for cid, delta in enumerate(deltas):
        delta_norm = np.linalg.norm(delta)
        if delta_norm < 1e-10:
            # A zero delta means the client didn't update — suspicious, not trustworthy
            scores[cid] = 0.5  # Neutral score instead of perfect trust
            continue

        # Component 1: Cosine similarity (direction alignment) mapped to [0, 1]
        cosine_sim = np.dot(delta, median_delta) / (delta_norm * median_norm)
        cosine_component = float(np.clip((cosine_sim + 1.0) / 2.0, 0.0, 1.0))

        # Component 2: Normalized L2 distance (magnitude consistency) mapped to [0, 1]
        if max_l2 > 1e-10:
            distance_component = float(1.0 - l2_distances[cid] / max_l2)
        else:
            distance_component = 1.0

        # Combined score: equal weight to direction and magnitude
        scores[cid] = 0.5 * cosine_component + 0.5 * distance_component

    return scores


def _apply_model_attack(parameters, attack_type, attack_params, seed,
                        global_parameters=None):
    if attack_type == "weight_spiking":
        return apply_weight_spiking(parameters, seed=seed, **attack_params)
    elif attack_type == "gradient_scaling":
        return apply_gradient_scaling(
            parameters, global_parameters=global_parameters, **attack_params
        )
    elif attack_type == "byzantine_perturbation":
        return apply_byzantine_perturbation(parameters, seed=seed, **attack_params)
    return parameters


def _evaluate_model(model, testloader, device, loss_name="cross_entropy",
                    model_plugin=None, model_plugin_kwargs=None):
    """Evaluate model on test set. Returns dict with at least 'loss'."""
    model.to(device)
    model.eval()
    plugin_kwargs = model_plugin_kwargs or {}

    if model_plugin and hasattr(model_plugin, "evaluate"):
        with torch.no_grad():
            metrics = model_plugin.evaluate(model, testloader, device, **plugin_kwargs)
    elif model_plugin and hasattr(model_plugin, "eval_step"):
        all_metrics, total_samples = [], 0
        with torch.no_grad():
            for batch in testloader:
                bs = _get_batch_size(batch)
                step = model_plugin.eval_step(model, batch, device, **plugin_kwargs)
                all_metrics.append((step, bs))
                total_samples += bs
        if total_samples == 0:
            return {"loss": 0.0, "accuracy": 0.0}
        # Aggregate with weighted mean, handling varying keys
        all_keys = set()
        for m, _ in all_metrics:
            all_keys.update(m.keys())
        metrics = {}
        for key in all_keys:
            metrics[key] = sum(m.get(key, 0.0) * n for m, n in all_metrics) / total_samples
    else:
        # Default classification eval
        criterion = _build_loss(loss_name)
        correct, total, total_loss = 0, 0, 0.0
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                total_loss += criterion(outputs, labels).item() * labels.size(0)
                correct += (outputs.argmax(1) == labels).sum().item()
                total += labels.size(0)
        if total == 0:
            return {"loss": 0.0, "accuracy": 0.0}
        metrics = {"loss": total_loss / total, "accuracy": correct / total}

    metrics.setdefault("loss", 0.0)
    return metrics


def _set_model_params(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict(
        {k: torch.from_numpy(np.asarray(v)) if isinstance(v, np.ndarray)
         else torch.from_numpy(np.asarray(v)) if not isinstance(v, torch.Tensor)
         else v.clone().detach()
         for k, v in params_dict}
    )
    model.load_state_dict(state_dict, strict=True)



def _aggregate_with_strategy(strategy, server_round, client_results, num_samples_list, client_ids):
    results = [
        (cid, FitRes(parameters=params, num_examples=n))
        for cid, params, n in zip(client_ids, client_results, num_samples_list)
    ]
    aggregated, strategy_metrics = strategy.aggregate_fit(server_round, results, [])
    return aggregated, strategy_metrics


def _is_attack_active(attack_cfg, rnd):
    """Check if an attack should be active this round, respecting schedule."""
    if attack_cfg.attack_type == "none":
        return False
    if attack_cfg.schedule_type == "static" or attack_cfg.attack_rounds is None:
        return True
    return rnd in attack_cfg.attack_rounds


def _setup_data(config: SimulationConfig) -> dict:
    """Load dataset, partition, build trainloaders, select malicious clients,
    and build poisoned loaders.

    Returns a dict with all setup state needed by the simulation loop.
    """
    train_dataset, test_dataset = get_dataset(config.dataset_name, **config.plugin_params.get("datasets", {}))

    dataset_info = _build_dataset_info(config.dataset_name)

    # Split training data into train + validation
    val_split = config.val_split
    if val_split > 0:
        val_size = int(len(train_dataset) * val_split)
        train_size = len(train_dataset) - val_size
        g = torch.Generator().manual_seed(config.seed)
        train_data, val_data = torch.utils.data.random_split(
            train_dataset, [train_size, val_size], generator=g
        )
    else:
        train_data = train_dataset
        val_data = None

    # Partition training data — keep clean copies for dynamic data poisoning
    clean_partitions = partition_dataset(
        train_data, config.num_clients,
        config.partition_type, config.alpha, config.seed,
    )

    # Attack setup
    attack_cfg = config.attack
    attack_category = ATTACK_CATEGORIES.get(attack_cfg.attack_type)
    malicious_clients = set()
    if attack_category is not None:
        malicious_clients = _select_malicious_clients(
            config.num_clients, attack_cfg.malicious_fraction, config.seed
        )

    # Pre-build poisoned partitions for data poisoning attacks
    poisoned_partitions = {}
    if attack_category == "data":
        for cid in malicious_clients:
            partition = clean_partitions[cid]
            if attack_cfg.attack_type == "label_flipping":
                # Guard: label flipping requires integer class labels
                sample_target = train_dataset[0][1] if len(train_dataset) > 0 else 0
                if isinstance(sample_target, float) or (hasattr(sample_target, 'is_floating_point') and sample_target.is_floating_point()):
                    import warnings
                    warnings.warn("Label flipping requires integer class labels but dataset has float targets. Skipping attack.")
                else:
                    poisoned_partitions[cid] = apply_label_flipping(
                        partition, dataset_info["num_classes"], seed=config.seed + cid
                    )
            elif attack_cfg.attack_type == "gaussian_noise":
                poisoned_partitions[cid] = apply_gaussian_noise(
                    partition, seed=config.seed + cid, **attack_cfg.attack_params
                )
            elif attack_cfg.attack_type == "token_replacement":
                poisoned_partitions[cid] = apply_token_replacement(
                    partition, seed=config.seed + cid, **attack_cfg.attack_params
                )

    # Model poisoning config
    model_attack_type = None
    model_attack_params = {}
    if attack_category == "model":
        model_attack_type = attack_cfg.attack_type
        model_attack_params = attack_cfg.attack_params

    pin = config.pin_memory and torch.cuda.is_available()
    testloader = DataLoader(test_dataset, batch_size=128, shuffle=False, pin_memory=pin)

    # Validation loader — used for per-round evaluation (not test set)
    # When val_split > 0, per-round eval uses validation data (no data leakage).
    # When val_split == 0, falls back to test set (legacy behavior).
    if val_data is not None:
        valloader = DataLoader(val_data, batch_size=128, shuffle=False, pin_memory=pin)
    else:
        valloader = testloader  # Legacy fallback

    # Build clean trainloaders (always available) — seeded for reproducibility
    clean_trainloaders = []
    for i, p in enumerate(clean_partitions):
        g = torch.Generator().manual_seed(config.seed + i)
        clean_trainloaders.append(
            DataLoader(p, batch_size=config.batch_size, shuffle=True, generator=g, pin_memory=pin)
        )
    # Build poisoned trainloaders
    poisoned_trainloaders = {}
    for cid, poisoned_part in poisoned_partitions.items():
        g = torch.Generator().manual_seed(config.seed + cid + 10000)
        poisoned_trainloaders[cid] = DataLoader(
            poisoned_part, batch_size=config.batch_size, shuffle=True, generator=g
        )

    # Eval model instance reused for per-client accuracy checks
    eval_model = get_model(config.model_name, config.dataset_name, **config.plugin_params.get("models", {}))
    # Template model for deepcopy — avoids expensive get_model() per client per round
    template_model = get_model(config.model_name, config.dataset_name, **config.plugin_params.get("models", {}))

    return {
        "dataset_info": dataset_info,
        "clean_partitions": clean_partitions,
        "attack_category": attack_category,
        "malicious_clients": malicious_clients,
        "model_attack_type": model_attack_type,
        "model_attack_params": model_attack_params,
        "testloader": testloader,
        "valloader": valloader,
        "clean_trainloaders": clean_trainloaders,
        "poisoned_trainloaders": poisoned_trainloaders,
        "eval_model": eval_model,
        "template_model": template_model,
    }


def _run_client_round(
    cid, config, setup, global_params, attack_active, rnd,
    template_model, eval_model, device, round_poisoned_loaders,
    model_plugin=None,
    strategy_plugin=None,
):
    """Train one client, apply attacks if needed, evaluate accuracy.

    Returns (updated_params, client_loss, client_acc, status, attack_applied).
    """
    attack_category = setup["attack_category"]
    malicious_clients = setup["malicious_clients"]
    model_attack_type = setup["model_attack_type"]
    model_attack_params = setup["model_attack_params"]
    clean_trainloaders = setup["clean_trainloaders"]
    evalloader = setup["valloader"]  # Use validation set for per-client eval

    is_malicious = cid in malicious_clients
    attack_applied = False

    # Select trainloader: poisoned if data attack is active this round
    if is_malicious and attack_category == "data" and attack_active and cid in round_poisoned_loaders:
        trainloader = round_poisoned_loaders[cid]
        attack_applied = True
    else:
        trainloader = clean_trainloaders[cid]

    client_model = copy.deepcopy(template_model)
    _set_model_params(client_model, global_params)

    # torch.compile: wraps the model for graph compilation on first forward pass.
    # Falls back gracefully if compilation fails at runtime (e.g., missing dev headers).
    compiled = False
    if not getattr(config, '_compile_failed', False) and config.compile_model and hasattr(torch, "compile"):
        client_model = torch.compile(client_model)
        compiled = True

    _train_kwargs = dict(
        optimizer_name=config.optimizer,
        loss_name=config.loss_function,
        weight_decay=config.weight_decay,
        optimizer_plugin_kwargs=config.plugin_params.get("optimizers", {}),
        loss_plugin_kwargs=config.plugin_params.get("losses", {}),
        model_plugin=model_plugin,
        model_plugin_kwargs=config.plugin_params.get("models", {}),
        strategy_plugin=strategy_plugin,
        strategy_plugin_kwargs=config.plugin_params.get("strategies", {}),
        scheduler_name=config.lr_scheduler,
        scheduler_params=config.lr_scheduler_params,
        use_amp=config.use_amp,
    )
    try:
        updated_params, train_metrics = _train_client(
            client_model, trainloader,
            config.local_epochs, config.learning_rate, device,
            **_train_kwargs,
        )
    except Exception as e:
        if compiled:
            import warnings
            warnings.warn(f"torch.compile failed ({e}). Falling back to eager mode.")
            config._compile_failed = True
            client_model = copy.deepcopy(template_model)
            _set_model_params(client_model, global_params)
            updated_params, train_metrics = _train_client(
                client_model, trainloader,
                config.local_epochs, config.learning_rate, device,
                **_train_kwargs,
            )
        else:
            raise
    client_loss = train_metrics.get("loss", 0.0)

    # Model poisoning
    if is_malicious and model_attack_type and attack_active:
        updated_params = _apply_model_attack(
            updated_params, model_attack_type, model_attack_params,
            seed=config.seed * 100000 + cid * 1000 + rnd,
            global_parameters=global_params,
        )
        attack_applied = True

    # Determine client status
    if not is_malicious:
        status = "benign"
    elif attack_applied:
        status = "attacked"
    else:
        status = "malicious_idle"

    # Per-client accuracy (quick eval on subset of validation data)
    client_acc = _evaluate_client_accuracy(
        eval_model, updated_params, evalloader, device,
        model_plugin=model_plugin,
        model_plugin_kwargs=config.plugin_params.get("models", {}),
    )

    return updated_params, client_loss, client_acc, status, attack_applied


def _process_round_results(
    client_results, num_samples_list, global_params, global_model,
    config, strategy, rnd, malicious_clients, anomaly_tracker, evalloader, device,
    model_plugin=None, model_plugin_kwargs=None,
    selected_cids=None,
):
    """Compute trust scores, aggregate, extract exclusion metadata, compute anomaly
    metrics, and evaluate the global model.

    Mutates global_params list in-place and updates global_model when aggregation
    succeeds. Returns a dict with all per-round computed data.
    """
    # Compute trust scores before aggregation
    trust_scores = _compute_trust_scores(client_results, global_params)

    # Notify strategy of current global params before aggregation.
    # In Flower's canonical loop, configure_fit() captures this before distributing
    # to clients.  FEDSIM skips configure_fit (no ClientManager), so we set the
    # state directly for strategies that need it (e.g., ReputationStrategy uses
    # global_params to compute update deltas for truth values).
    if hasattr(strategy, 'global_params'):
        strategy.global_params = np.concatenate([p.flatten() for p in global_params])

    # Aggregate
    aggregated, strategy_metrics = _aggregate_with_strategy(
        strategy, rnd, client_results, num_samples_list, selected_cids or list(range(len(client_results)))
    )
    if aggregated is not None:
        global_params[:] = aggregated
        _set_model_params(global_model, global_params)

    # Extract exclusion metadata from strategy return
    excluded_set = set()
    included_set = set(range(config.num_clients))
    client_scores_round = {}
    if "excluded_clients" in strategy_metrics:
        excluded_set = set(json.loads(strategy_metrics["excluded_clients"]))
    if "included_clients" in strategy_metrics:
        included_set = set(json.loads(strategy_metrics["included_clients"]))
    if "client_scores" in strategy_metrics:
        raw_scores = json.loads(strategy_metrics["client_scores"])
        client_scores_round = {int(k): float(v) for k, v in raw_scores.items()}

    # Use the set of clients that actually participated in this round's
    # aggregation (included + excluded).  When fraction_fit < 1.0, not all
    # clients are considered by the strategy — non-participants should not
    # count toward TP/FP/TN/FN.
    participating_clients = included_set | excluded_set
    if not participating_clients:
        # Fallback: if strategy didn't report inclusion metadata, assume all
        participating_clients = set(range(config.num_clients))
    anomaly_result = anomaly_tracker.compute_round(
        malicious_clients, excluded_set, participating_clients
    )

    # Retrieve reputation scores if strategy supports them
    reputation_scores = {}
    if hasattr(strategy, 'get_reputations'):
        reputation_scores = strategy.get_reputations()

    # Evaluate global model on validation set
    eval_metrics = _evaluate_model(global_model, evalloader, device,
                                    loss_name=config.loss_function,
                                    model_plugin=model_plugin,
                                    model_plugin_kwargs=model_plugin_kwargs)
    loss = eval_metrics.get("loss", 0.0)
    acc = eval_metrics.get("accuracy", 0.0)

    # Collect non-standard eval metrics for custom_metrics accumulation
    extra_eval_metrics = {k: v for k, v in eval_metrics.items()
                          if k not in ("loss", "accuracy")}

    return {
        "trust_scores": trust_scores,
        "excluded_set": excluded_set,
        "included_set": included_set,
        "client_scores_round": client_scores_round,
        "anomaly_result": anomaly_result,
        "reputation_scores": reputation_scores,
        "loss": loss,
        "acc": acc,
        "extra_eval_metrics": extra_eval_metrics,
    }


def _run_clients_parallel(
    selected_cids, config, setup, global_params, attack_active, rnd,
    template_model, device, round_poisoned_loaders,
    model_plugin=None, strategy_plugin=None,
):
    """Train selected clients, optionally in parallel.

    When max_parallel_clients=1, preserves exact sequential behavior.
    When >1, uses ThreadPoolExecutor with per-thread eval models and
    optional CUDA streams for GPU parallelism.

    Returns:
        client_results: list[NDArrays] — ordered matching selected_cids
        num_samples_list: list[int] — ordered matching selected_cids
        client_statuses: dict[int, str]
        client_accs: dict[int, float]
        client_events: list[tuple] — (cid, client_loss, client_acc, status, attack_applied)
    """
    max_workers = min(config.max_parallel_clients, len(selected_cids))
    malicious_clients = setup["malicious_clients"]
    clean_partitions = setup["clean_partitions"]

    def _train_one(cid):
        """Train a single client. Thread-safe: all mutable state is local."""
        stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None
        ctx = torch.cuda.stream(stream) if stream else nullcontext()
        with ctx:
            thread_eval_model = copy.deepcopy(template_model)
            updated_params, client_loss, client_acc, status, attack_applied = (
                _run_client_round(
                    cid, config, setup, global_params, attack_active, rnd,
                    template_model, thread_eval_model, device,
                    round_poisoned_loaders,
                    model_plugin=model_plugin,
                    strategy_plugin=strategy_plugin,
                )
            )
        if stream:
            stream.synchronize()
        return cid, updated_params, client_loss, client_acc, status, attack_applied

    # --- Sequential path (max_parallel_clients == 1) ---
    if max_workers <= 1:
        thread_eval_model = copy.deepcopy(template_model)
        client_results = []
        num_samples_list = []
        statuses = {}
        accs = {}
        events = []
        for cid in selected_cids:
            updated_params, client_loss, client_acc, status, attack_applied = (
                _run_client_round(
                    cid, config, setup, global_params, attack_active, rnd,
                    template_model, thread_eval_model, device,
                    round_poisoned_loaders,
                    model_plugin=model_plugin,
                    strategy_plugin=strategy_plugin,
                )
            )
            client_results.append(updated_params)
            num_samples_list.append(len(clean_partitions[cid]))
            statuses[cid] = status
            accs[cid] = client_acc
            events.append((cid, client_loss, client_acc, status, attack_applied))
        return client_results, num_samples_list, statuses, accs, events

    # --- Parallel path ---
    client_results = [None] * len(selected_cids)
    num_samples_list = [None] * len(selected_cids)
    statuses = {}
    accs = {}
    events = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_idx = {}
        for idx, cid in enumerate(selected_cids):
            future = pool.submit(_train_one, cid)
            future_to_idx[future] = idx

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            cid, updated_params, client_loss, client_acc, status, attack_applied = (
                future.result()
            )
            client_results[idx] = updated_params
            num_samples_list[idx] = len(clean_partitions[cid])
            statuses[cid] = status
            accs[cid] = client_acc
            events.append((cid, client_loss, client_acc, status, attack_applied))

    return client_results, num_samples_list, statuses, accs, events


def run_simulation(
    config: SimulationConfig,
    progress_callback=None,
    round_callback=None,
    client_callback=None,
) -> list[SimulationResult]:
    """Run a federated learning simulation across multiple strategies.

    Executes the FL simulation loop: partitions data, trains clients locally,
    applies attacks, aggregates with each strategy, and collects metrics.

    Args:
        config: Full simulation configuration including model, dataset,
                strategies, attack settings, and FL hyperparameters.
        progress_callback: Called when a new strategy begins.
            Signature: (strategy_name: str, strategy_idx: int, num_strategies: int)
        round_callback: Called after each FL round completes.
            Signature: (event: RoundEvent)
        client_callback: Called after each client finishes local training.
            Signature: (event: ClientTrainEvent)

    Returns:
        List of SimulationResult, one per strategy in config.strategies.

    Raises:
        ValueError: If configuration parameters are invalid.
    """
    _validate_config(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    setup = _setup_data(config)

    # Resolve model plugin once for three-tier dispatch
    model_plugin = _get_model_plugin(config.model_name)
    model_plugin_kwargs = config.plugin_params.get("models", {})

    dataset_info = setup["dataset_info"]
    clean_partitions = setup["clean_partitions"]
    attack_category = setup["attack_category"]
    malicious_clients = setup["malicious_clients"]
    poisoned_trainloaders = setup["poisoned_trainloaders"]
    testloader = setup["testloader"]
    valloader = setup["valloader"]
    eval_model = setup["eval_model"]
    template_model = setup["template_model"]

    attack_cfg = config.attack

    results = []

    for strat_idx, strategy_name in enumerate(config.strategies):
        if progress_callback:
            progress_callback(strategy_name, strat_idx, len(config.strategies))

        global_model = get_model(config.model_name, config.dataset_name, **config.plugin_params.get("models", {}))
        global_params = [val.cpu().numpy() for val in global_model.state_dict().values()]
        # Reset models to initial state for this strategy
        _set_model_params(template_model, global_params)
        _set_model_params(eval_model, global_params)
        initial_parameters = global_params
        strategy = _get_strategy(
            strategy_name, initial_parameters, config.num_clients,
            num_malicious=len(malicious_clients),
            reputation_truth_threshold=config.reputation_truth_threshold,
            reputation_selection_fraction=config.reputation_selection_fraction,
            reputation_initial_reputation=config.reputation_initial_reputation,
            plugin_kwargs=config.plugin_params.get("strategies", {}),
            fraction_fit=config.fraction_fit,
            fraction_evaluate=config.fraction_evaluate,
            min_fit_clients=config.min_fit_clients,
            min_evaluate_clients=config.min_evaluate_clients,
            min_available_clients=config.min_available_clients,
        )
        strategy_plugin = getattr(strategy, '_fedsim_plugin_module', None)

        round_losses = []
        round_accuracies = []
        trust_hist = {}
        rep_hist = {}
        status_hist = []
        final_params = None
        anomaly_tracker = AnomalyMetrics()
        scores_hist = []
        custom_metrics_acc = {}  # {metric_key: [per_round_values]}

        # Evaluate initial model on validation set (not test set — avoids data leakage)
        eval_metrics = _evaluate_model(global_model, valloader, device,
                                        loss_name=config.loss_function,
                                        model_plugin=model_plugin,
                                        model_plugin_kwargs=model_plugin_kwargs)
        loss = eval_metrics.get("loss", 0.0)
        acc = eval_metrics.get("accuracy", 0.0)
        round_losses.append(loss)
        round_accuracies.append(acc)

        start_time = time.time()

        # Emit initial round event (round 0)
        if round_callback:
            client_statuses = {}
            for cid in range(config.num_clients):
                if cid in malicious_clients:
                    client_statuses[cid] = "malicious_idle"
                else:
                    client_statuses[cid] = "benign"
            round_callback(RoundEvent(
                strategy_name=strategy_name,
                strategy_idx=strat_idx,
                num_strategies=len(config.strategies),
                round_num=0,
                num_rounds=config.num_rounds,
                loss=loss,
                accuracy=acc,
                loss_delta=0.0,
                accuracy_delta=0.0,
                client_statuses=client_statuses,
                elapsed=0.0,
            ))

        # RNG for client sampling (fraction_fit)
        client_rng = np.random.default_rng(config.seed + 7919)

        for rnd in range(1, config.num_rounds + 1):
            attack_active = _is_attack_active(attack_cfg, rnd)

            # Sample clients for this round based on fraction_fit
            all_cids = list(range(config.num_clients))
            num_selected = max(1, int(config.num_clients * config.fraction_fit))
            if num_selected >= config.num_clients:
                selected_cids = all_cids
            else:
                selected_cids = sorted(client_rng.choice(
                    config.num_clients, size=num_selected, replace=False
                ).tolist())

            # For dynamic data attacks, regenerate poisoned loaders with per-round seeds
            # so the poisoning varies across rounds (more realistic simulation)
            round_poisoned_loaders = poisoned_trainloaders
            if (attack_category == "data" and attack_active
                    and attack_cfg.schedule_type == "dynamic"):
                round_poisoned_loaders = {}
                for cid in malicious_clients:
                    if cid not in selected_cids:
                        continue
                    partition = clean_partitions[cid]
                    round_seed = config.seed + cid + rnd * 1000
                    if attack_cfg.attack_type == "label_flipping":
                        # Guard: label flipping requires integer class labels
                        sample_target = partition[0][1] if len(partition) > 0 else 0
                        if isinstance(sample_target, float) or (hasattr(sample_target, 'is_floating_point') and sample_target.is_floating_point()):
                            import warnings
                            warnings.warn("Label flipping requires integer class labels but dataset has float targets. Skipping attack.")
                            continue
                        pp = apply_label_flipping(
                            partition, dataset_info["num_classes"], seed=round_seed
                        )
                    elif attack_cfg.attack_type == "gaussian_noise":
                        pp = apply_gaussian_noise(
                            partition, seed=round_seed, **attack_cfg.attack_params
                        )
                    elif attack_cfg.attack_type == "token_replacement":
                        pp = apply_token_replacement(
                            partition, seed=round_seed, **attack_cfg.attack_params
                        )
                    else:
                        continue
                    g = torch.Generator().manual_seed(round_seed)
                    round_poisoned_loaders[cid] = DataLoader(
                        pp, batch_size=config.batch_size, shuffle=True, generator=g
                    )

            client_results, num_samples_list, client_statuses, client_accs, client_events = (
                _run_clients_parallel(
                    selected_cids, config, setup, global_params, attack_active, rnd,
                    template_model, device, round_poisoned_loaders,
                    model_plugin=model_plugin,
                    strategy_plugin=strategy_plugin,
                )
            )

            # Dispatch callbacks sequentially after all clients complete
            if client_callback:
                for cid_ev, client_loss, client_acc, status, attack_applied in client_events:
                    client_callback(ClientTrainEvent(
                        strategy_name=strategy_name,
                        round_num=rnd,
                        client_id=cid_ev,
                        num_clients=config.num_clients,
                        is_malicious=cid_ev in malicious_clients,
                        attack_applied=attack_applied,
                        client_loss=client_loss,
                        client_accuracy=client_acc,
                    ))

            # Mark non-selected clients as idle in statuses
            for cid in all_cids:
                if cid not in client_statuses:
                    client_statuses[cid] = "idle"

            round_data = _process_round_results(
                client_results, num_samples_list, global_params, global_model,
                config, strategy, rnd, malicious_clients, anomaly_tracker, valloader, device,
                model_plugin=model_plugin, model_plugin_kwargs=model_plugin_kwargs,
                selected_cids=selected_cids,
            )

            trust_scores = round_data["trust_scores"]
            excluded_set = round_data["excluded_set"]
            included_set = round_data["included_set"]
            client_scores_round = round_data["client_scores_round"]
            anomaly_result = round_data["anomaly_result"]
            reputation_scores = round_data["reputation_scores"]
            loss = round_data["loss"]
            acc = round_data["acc"]
            extra_eval_metrics = round_data.get("extra_eval_metrics", {})

            # Accumulate trust scores
            for cid, score in trust_scores.items():
                trust_hist.setdefault(cid, []).append(score)

            # Accumulate client statuses
            status_hist.append(client_statuses)

            # Capture final-round client params for PCA visualization
            if rnd == config.num_rounds:
                final_params = {cid: client_results[i] for i, cid in enumerate(selected_cids)}

            scores_hist.append(client_scores_round)

            # Accumulate reputation scores
            for cid, score in reputation_scores.items():
                rep_hist.setdefault(cid, []).append(score)

            # Execute custom metrics on evaluation rounds
            round_custom_metrics = {}
            if config.active_metrics and rnd % config.eval_frequency == 0:
                from plugins import discover_plugins
                metrics_plugins = discover_plugins("metrics")
                metrics_kwargs = config.plugin_params.get("metrics", {})
                for mname, mmod in metrics_plugins.items():
                    if not isinstance(mmod, _types_module.ModuleType):
                        continue
                    plugin_display = getattr(mmod, "NAME", mname)
                    if plugin_display not in config.active_metrics:
                        continue
                    try:
                        metric_result = mmod.compute(global_model, valloader, device, **metrics_kwargs)
                        for mk, mv in metric_result.items():
                            round_custom_metrics[f"{plugin_display}/{mk}"] = mv
                    except Exception as exc:
                        print(f"Warning: Metric plugin '{plugin_display}' failed on round {rnd}: {exc}")

            for mk, mv in round_custom_metrics.items():
                if mk not in custom_metrics_acc:
                    custom_metrics_acc[mk] = []
                custom_metrics_acc[mk].append(mv)

            # Accumulate non-standard eval metrics from plugin evaluate/eval_step
            for mk, mv in extra_eval_metrics.items():
                key = f"eval/{mk}"
                if key not in custom_metrics_acc:
                    custom_metrics_acc[key] = []
                custom_metrics_acc[key].append(mv)

            prev_loss = round_losses[-1]
            prev_acc = round_accuracies[-1]
            round_losses.append(loss)
            round_accuracies.append(acc)

            elapsed = time.time() - start_time

            if round_callback:
                round_callback(RoundEvent(
                    strategy_name=strategy_name,
                    strategy_idx=strat_idx,
                    num_strategies=len(config.strategies),
                    round_num=rnd,
                    num_rounds=config.num_rounds,
                    loss=loss,
                    accuracy=acc,
                    loss_delta=loss - prev_loss,
                    accuracy_delta=acc - prev_acc,
                    client_statuses=client_statuses,
                    elapsed=elapsed,
                    client_trust_scores=trust_scores,
                    client_accuracies=client_accs,
                    client_reputation_scores=reputation_scores,
                    client_excluded=excluded_set,
                    client_included=included_set,
                    removal_precision=anomaly_result["precision"],
                    removal_recall=anomaly_result["recall"],
                    removal_f1=anomaly_result["f1"],
                    strategy_scores=client_scores_round,
                    custom_metrics={**round_custom_metrics,
                                    **{f"eval/{k}": v for k, v in extra_eval_metrics.items()}},
                ))

        elapsed = time.time() - start_time

        # Final evaluation on held-out test set (never seen during training)
        test_metrics = _evaluate_model(global_model, testloader, device,
                                        loss_name=config.loss_function,
                                        model_plugin=model_plugin,
                                        model_plugin_kwargs=model_plugin_kwargs)

        results.append(SimulationResult(
            strategy_name=strategy_name,
            round_losses=round_losses,
            round_accuracies=round_accuracies,
            total_time=elapsed,
            trust_history=trust_hist,
            reputation_history=rep_hist,
            final_client_params=final_params,
            client_statuses_history=status_hist,
            anomaly_history=anomaly_tracker.rounds,
            anomaly_summary=anomaly_tracker.summary(),
            strategy_scores_history=scores_hist,
            custom_metrics=custom_metrics_acc,
            test_accuracy=test_metrics.get("accuracy", 0.0),
            test_loss=test_metrics.get("loss", 0.0),
        ))

    return results
