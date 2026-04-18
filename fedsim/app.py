import sys
import os
import json
import dataclasses
import html as _html

# Ensure project root is importable (removed during packaging; kept for dev)
_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# ── App Constants ────────────────────────────────────────────────────
APP_NAME = "FEDSIM"
APP_TAGLINE = "Federated Learning Simulation & Visualization Framework"

import streamlit as st
import pandas as pd
import numpy as np

from configs.defaults import (
    SUPPORTED_MODELS, SUPPORTED_DATASETS, SUPPORTED_STRATEGIES, PARTITION_TYPES,
    DEFAULT_MODEL, DEFAULT_DATASET, DEFAULT_PARTITION_TYPE, DEFAULT_STRATEGIES,
    DEFAULT_ATTACK, DEFAULT_NUM_CLIENTS, DEFAULT_NUM_ROUNDS, DEFAULT_LOCAL_EPOCHS,
    DEFAULT_LEARNING_RATE, DEFAULT_ALPHA, SUPPORTED_ATTACKS, ATTACK_CATEGORIES,
    ATTACK_SCHEDULE_TYPES, DEFAULT_MALICIOUS_FRACTION,
    SUPPORTED_OPTIMIZERS, SUPPORTED_LOSSES, DEFAULT_OPTIMIZER, DEFAULT_LOSS,
    PRESETS, MODEL_COMPATIBLE_TASKS, ATTACK_COMPATIBLE_TASKS,
    DEFAULT_FRACTION_FIT, DEFAULT_FRACTION_EVALUATE,
    SUPPORTED_SCHEDULERS,
)
from simulation.runner import SimulationConfig, AttackConfig, run_simulation, RoundEvent, _build_dataset_info, _validate_config, _validate_compatibility
from plugins import get_plugin_choices, get_plugin_module, discover_plugins, discover_all_plugins, clear_cache
if "plugins_initialized" not in st.session_state:
    clear_cache()
    st.session_state["plugins_initialized"] = True
from pathlib import Path


def _get_task_type(dataset_name: str) -> str:
    """Get task type for a dataset, defaulting to image_classification."""
    try:
        return _build_dataset_info(dataset_name).get("task_type", "image_classification")
    except (ValueError, KeyError):
        return "image_classification"


def _task_metric_labels(task_type: str) -> tuple[str, str]:
    """Returns (primary_metric_label, loss_label) for a task type."""
    if "classification" in task_type:
        return "Accuracy", "Loss"
    elif task_type == "regression":
        return "MAE", "MSE Loss"
    elif task_type == "language_modeling":
        return "Perplexity", "Cross-Entropy Loss"
    return "Primary Metric", "Loss"


def _extract_malicious_clients(results):
    """Extract set of malicious client IDs from simulation results."""
    malicious = set()
    for r in results:
        for rnd_st in r.client_statuses_history:
            for cid, status in rnd_st.items():
                if status in ("attacked", "malicious_idle"):
                    malicious.add(cid)
    return malicious


from visualization.plots import (
    plot_live_loss, plot_live_accuracy, plot_client_grid,
    plot_custom_metric, plot_client_sparklines, STRATEGY_DISPLAY_NAMES,
)
from visualization.plots_3d import (
    plot_accuracy_surface, plot_trust_reputation_landscape,
    plot_attack_impact, plot_client_pca,
)
from visualization.anomaly_plots import (
    plot_removal_f1_over_rounds, plot_exclusion_timeline,
    plot_confusion_summary, plot_client_score_distribution,
)

# ── Page Config ──────────────────────────────────────────────────────
st.set_page_config(page_title=APP_NAME, layout="wide", initial_sidebar_state="expanded")

# ── Custom CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Tighter spacing */
    .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }
    [data-testid="stSidebar"] { width: 280px; }
    [data-testid="stSidebar"] .block-container { padding-top: 1rem; }

    /* Header */
    .app-header {
        display: flex; align-items: baseline; gap: 12px;
        margin-bottom: 0.25rem;
    }
    .app-header h1 { margin: 0; font-size: 1.8rem; font-weight: 700; color: #7FB5A0; letter-spacing: -0.02em; }
    .app-header span { font-size: 0.85rem; color: #8B919E; }

    /* Status bar */
    .status-bar {
        background: #1C1F26; border: 1px solid #2D3140; border-radius: 8px;
        padding: 8px 16px; font-family: 'SF Mono', 'Fira Code', monospace;
        font-size: 0.82rem; color: #C8CCD4; margin-bottom: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    }
    .status-bar strong { color: #7FB5A0; }
    .status-bar .acc { color: #7FB5A0; }
    .status-bar .loss { color: #D4726A; }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] { gap: 0; }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 20px; font-size: 0.85rem; font-weight: 500;
    }

    /* Sidebar section labels */
    [data-testid="stSidebar"] h3 {
        font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em;
        color: #7FB5A0; margin-top: 1rem; margin-bottom: 0.25rem;
    }

    /* Compact table */
    .stTable { font-size: 0.85rem; }
    .stTable td, .stTable th { padding: 6px 12px; }

    /* Grid legend */
    .grid-legend {
        display: flex; gap: 14px; align-items: center;
        font-size: 0.78rem; color: #8B919E; margin-bottom: 4px;
    }
    .grid-legend .dot {
        display: inline-block; width: 10px; height: 10px;
        border-radius: 2px; margin-right: 4px; vertical-align: middle;
    }

    /* Download buttons */
    .stDownloadButton button {
        font-size: 0.8rem; padding: 4px 14px;
        background: transparent; border: 1px solid #2D3140;
        transition: background 0.2s, border-color 0.2s;
    }
    .stDownloadButton button:hover {
        border-color: #7FB5A0;
        background: rgba(127,181,160,0.08);
    }

    /* Hide streamlit branding */
    #MainMenu, footer { visibility: hidden; }

    /* Empty state */
    .empty-state {
        display: flex; flex-direction: column; align-items: center;
        justify-content: center; min-height: 350px; color: #8B919E;
        text-align: center;
        border: 1px solid rgba(127,181,160,0.25);
        border-radius: 8px;
    }
    .empty-state p { font-size: 0.9rem; margin: 4px 0; }
    .empty-state .hint { font-size: 0.78rem; color: #8B919E; }

    /* Metric cards */
    [data-testid="stMetricValue"] { font-size: 1.4rem; }

    /* Run button mint glow */
    [data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background: #5A9E87;
        border: none;
        box-shadow: 0 0 12px rgba(127,181,160,0.3);
    }
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────
st.markdown(
    f'<div class="app-header"><h1>{APP_NAME}</h1>'
    f'<span>{APP_TAGLINE}</span></div>',
    unsafe_allow_html=True,
)

# Show config summary if a simulation has been run
if "config" in st.session_state:
    cfg = st.session_state["config"]
    atk = cfg.attack.attack_type.replace("_", " ").title() if cfg.attack.attack_type != "none" else "None"
    atk_safe = _html.escape(atk)
    model_safe = _html.escape(cfg.model_name.upper())
    dataset_safe = _html.escape(cfg.dataset_name.upper())
    st.markdown(
        f'<div style="background:#1C1F26;border:1px solid #2D3140;border-radius:4px;'
        f'padding:6px 14px;font-size:0.78rem;color:#8B919E;margin-bottom:8px;">'
        f'{model_safe} &middot; {dataset_safe} &middot; '
        f'{cfg.num_clients} clients &middot; {cfg.num_rounds} rounds &middot; '
        f'Attack: {atk_safe} &middot; Seed: {cfg.seed}</div>',
        unsafe_allow_html=True,
    )

# ── Plugin discovery ─────────────────────────────────────────────────
def _prefixed_plugin_choices(plugin_type):
    """Return plugin choices with [Plugin] prefix for display names."""
    return {f"[Plugin] {name}": key for name, key in get_plugin_choices(plugin_type).items()}

def _render_plugin_params(plugin_type, plugin_key):
    """Render PARAMS controls for a selected custom plugin. Returns param values dict."""
    if not plugin_key.startswith("custom:"):
        return {}
    plugin_name = plugin_key.replace("custom:", "")
    mod = get_plugin_module(plugin_type, plugin_name)
    if mod is None or not hasattr(mod, "PARAMS"):
        return {}

    params_spec = mod.PARAMS
    plugin_display = getattr(mod, "NAME", plugin_name)
    values = {}

    with st.expander(f"{plugin_display} Settings", expanded=True):
        for pname, pspec in params_spec.items():
            ptype = pspec.get("type", "")
            default = pspec.get("default")
            label = pspec.get("label", pname.replace("_", " ").title())
            # Use plugin_key to ensure uniqueness across plugins of same type
            safe_key = plugin_key.replace(":", "_").replace(" ", "_")
            widget_key = f"pp_{plugin_type}_{safe_key}_{pname}"

            try:
                if ptype == "int":
                    values[pname] = st.slider(label, int(pspec["min"]), int(pspec["max"]),
                                              int(default), int(pspec.get("step", 1)), key=widget_key)
                elif ptype == "float":
                    values[pname] = st.slider(label, float(pspec["min"]), float(pspec["max"]),
                                              float(default), float(pspec.get("step", 0.01)), key=widget_key)
                elif ptype == "select":
                    idx = pspec["options"].index(default) if default in pspec["options"] else 0
                    values[pname] = st.selectbox(label, pspec["options"], index=idx, key=widget_key)
                elif ptype == "bool":
                    values[pname] = st.checkbox(label, value=bool(default), key=widget_key)
                else:
                    st.warning(f"Unknown param type '{ptype}' for {pname}")
            except (KeyError, TypeError) as e:
                st.warning(f"Malformed PARAMS for {pname}: {e}")
    return values

model_choices = {**SUPPORTED_MODELS, **_prefixed_plugin_choices("models")}
dataset_choices = {**SUPPORTED_DATASETS, **_prefixed_plugin_choices("datasets")}
strategy_choices = {**SUPPORTED_STRATEGIES, **get_plugin_choices("strategies")}
optimizer_choices = {**SUPPORTED_OPTIMIZERS, **_prefixed_plugin_choices("optimizers")}
loss_choices = {**SUPPORTED_LOSSES, **_prefixed_plugin_choices("losses")}

model_keys = list(model_choices.keys())
dataset_keys = list(dataset_choices.keys())
_midx = next((i for i, v in enumerate(model_choices.values()) if v == DEFAULT_MODEL), 0)
_didx = next((i for i, v in enumerate(dataset_choices.values()) if v == DEFAULT_DATASET), 0)
_aidx = next((i for i, v in enumerate(SUPPORTED_ATTACKS.values()) if v == DEFAULT_ATTACK), 0)
# ── Sidebar ──────────────────────────────────────────────────────────
plugin_params = {}
with st.sidebar:
    # ── Preset selector ──────────────────────────────────────────
    preset_names = ["Custom"] + list(PRESETS.keys())

    # Preset loading: use _preset_applied flag to prevent infinite rerun
    _prev_preset = st.session_state.get("_preset_applied", "Custom")

    preset_choice = st.selectbox("Preset", preset_names,
                                  key="preset_selector",
                                  help="Load a pre-configured experiment setup")

    _pd = {}
    if preset_choice != "Custom" and preset_choice != _prev_preset:
        # User just selected a NEW preset — write widget keys directly and rerun
        preset = PRESETS[preset_choice]
        st.session_state["_preset_applied"] = preset_choice
        # Write preset data for index overrides on the rerun pass
        st.session_state["_preset_data"] = preset
        # Write directly to strategy checkbox keys
        for disp, key in strategy_choices.items():
            st.session_state[f"strat_{key}"] = key in preset.get("strategies", [])
        st.rerun()
    elif preset_choice == "Custom" and _prev_preset != "Custom":
        # User switched back to Custom
        st.session_state["_preset_applied"] = "Custom"
        st.session_state.pop("_preset_data", None)

    _pd = st.session_state.pop("_preset_data", {})

    st.markdown("### Model & Data")
    # Override default indices with preset values
    _pmidx = _midx
    if _pd.get("model"):
        _pmidx = next((i for i, v in enumerate(model_choices.values()) if v == _pd["model"]), _pmidx)
    _pdidx = _didx
    if _pd.get("dataset"):
        _pdidx = next((i for i, v in enumerate(dataset_choices.values()) if v == _pd["dataset"]), _pdidx)

    model_display = st.selectbox("Model", model_keys, index=_pmidx, label_visibility="collapsed",
                                     help="Neural network architecture for client training")
    model_name = model_choices[model_display]
    _mp = _render_plugin_params("models", model_name)
    if _mp:
        plugin_params["models"] = _mp
    dataset_display = st.selectbox("Dataset", dataset_keys, index=_pdidx, label_visibility="collapsed",
                                       help="Training/test dataset for the FL simulation")
    dataset_name = dataset_choices[dataset_display]
    # Show task type
    _task = _get_task_type(dataset_name)
    st.caption(f"Task: {_task}")

    # Compatibility warning for model
    if _task:
        try:
            if not model_name.startswith("custom:"):
                _mcompat = MODEL_COMPATIBLE_TASKS.get(model_name)
                if _mcompat and _task not in _mcompat:
                    st.warning(f"Model may not be compatible with '{_task}' tasks.")
            else:
                _cmp = get_plugin_module("models", model_name.replace("custom:", ""))
                if _cmp:
                    _mcompat = getattr(_cmp, "COMPATIBLE_TASKS", None)
                    if _mcompat and _task not in _mcompat:
                        st.warning(f"Model may not be compatible with '{_task}' tasks.")
        except (ValueError, KeyError):
            pass
    _dp = _render_plugin_params("datasets", dataset_name)
    if _dp:
        plugin_params["datasets"] = _dp

    _p_part = _pd.get("partition_type", DEFAULT_PARTITION_TYPE)
    _pidx = 1 if _p_part == "non_iid" else 0
    partition_display = st.radio("Partition", PARTITION_TYPES, index=_pidx, horizontal=True)
    partition_type = "iid" if partition_display == "IID" else "non_iid"
    alpha = _pd.get("alpha", DEFAULT_ALPHA)
    if partition_type == "non_iid":
        alpha = st.slider("Dirichlet α", 0.01, 10.0, alpha, 0.01,
                          help="Lower values = more heterogeneous data across clients. 0.1 = extreme skew, 1.0 = moderate, 10.0 = nearly IID")

    st.markdown("---")

    with st.expander("FL Parameters", expanded=True):
        val_split = st.slider("Validation Split", 0.0, 0.3, 0.1, 0.05,
                               help="Fraction of training data held out for server-side validation. "
                                    "Per-round evaluation uses this split — test set is reserved for final evaluation only. "
                                    "Set to 0 for legacy behavior (uses test set during training).")
        c1, c2 = st.columns(2)
        num_clients = c1.slider("Clients", 2, 20, _pd.get("num_clients", DEFAULT_NUM_CLIENTS),
                                help="Number of federated learning participants")
        num_rounds = c2.slider("Rounds", 1, 50, _pd.get("num_rounds", DEFAULT_NUM_ROUNDS),
                               help="Number of aggregation rounds (communication cycles)")
        c3, c4 = st.columns(2)
        local_epochs = c3.slider("Epochs", 1, 10, _pd.get("local_epochs", DEFAULT_LOCAL_EPOCHS),
                                 help="Local training epochs per client per round")
        learning_rate = c4.number_input("LR", 0.0001, 1.0, _pd.get("learning_rate", DEFAULT_LEARNING_RATE), 0.001, format="%.4f",
                                        help="SGD/optimizer step size for local training")

        parallel_clients = st.slider(
            "Parallel Clients", 1, num_clients, 1,
            help="Number of clients to train concurrently. Higher values use more GPU memory. "
                 "Set to 1 for sequential (deterministic) training.",
        )

        st.markdown("**Client Sampling**")
        cs1, cs2 = st.columns(2)
        fraction_fit = cs1.slider("Fraction Fit", 0.1, 1.0,
            _pd.get("fraction_fit", DEFAULT_FRACTION_FIT), 0.1,
            help="Fraction of clients selected for training each round")
        fraction_evaluate = cs2.slider("Fraction Eval", 0.1, 1.0,
            _pd.get("fraction_evaluate", DEFAULT_FRACTION_EVALUATE), 0.1,
            help="Fraction of clients selected for evaluation each round")

    with st.expander("Training", expanded=False):
        c_opt, c_loss = st.columns(2)
        _opt_keys = list(optimizer_choices.keys())
        _opt_idx = next((i for i, v in enumerate(optimizer_choices.values()) if v == DEFAULT_OPTIMIZER), 0)
        if _pd.get("optimizer"):
            _opt_idx = next((i for i, v in enumerate(optimizer_choices.values()) if v == _pd["optimizer"]), _opt_idx)
        optimizer_display = c_opt.selectbox("Optimizer", _opt_keys, index=_opt_idx,
                                            help="Optimization algorithm for local client training")
        optimizer_name = optimizer_choices[optimizer_display]
        _opp = _render_plugin_params("optimizers", optimizer_name)
        if _opp:
            plugin_params["optimizers"] = _opp
        _loss_keys = list(loss_choices.keys())
        _loss_idx = next((i for i, v in enumerate(loss_choices.values()) if v == DEFAULT_LOSS), 0)
        if _pd.get("loss_function"):
            _loss_idx = next((i for i, v in enumerate(loss_choices.values()) if v == _pd["loss_function"]), _loss_idx)
        loss_display = c_loss.selectbox("Loss", _loss_keys, index=_loss_idx,
                                        help="Loss function for model training")
        loss_name = loss_choices[loss_display]
        if model_name.startswith("custom:"):
            _mp = get_plugin_module("models", model_name.replace("custom:", ""))
            if _mp and (hasattr(_mp, "fit") or hasattr(_mp, "train_step")):
                st.caption("Plugin controls training and loss internally.")
        _lp = _render_plugin_params("losses", loss_name)
        if _lp:
            plugin_params["losses"] = _lp

        scheduler_choices = {**SUPPORTED_SCHEDULERS, **_prefixed_plugin_choices("schedulers")}
        scheduler_display = st.selectbox("LR Scheduler", list(scheduler_choices.keys()),
            index=0, help="Learning rate schedule applied per epoch during local training")
        scheduler_name = scheduler_choices[scheduler_display]

        lr_scheduler_params = {}
        if scheduler_name == "step_lr":
            lr_scheduler_params["step_size"] = st.slider("Step Size", 1, 20, 5,
                help="Decay LR every N epochs")
            lr_scheduler_params["gamma"] = st.slider("Gamma", 0.01, 1.0, 0.1, 0.01,
                help="Multiplicative factor of LR decay")
        elif scheduler_name == "cosine_annealing":
            lr_scheduler_params["T_max"] = st.slider("T_max", 1, 50, local_epochs,
                help="Maximum number of iterations for cosine annealing")
        elif scheduler_name == "exponential":
            lr_scheduler_params["gamma"] = st.slider("Exp Gamma", 0.5, 0.999, 0.95, 0.005,
                help="Multiplicative factor of LR decay per epoch")
        elif scheduler_name.startswith("custom:"):
            sched_params = _render_plugin_params("schedulers", scheduler_name)
            if sched_params:
                lr_scheduler_params = sched_params

    # ── Metrics plugins ──────────────────────────────────────────────
    metrics_plugin_names = list(get_plugin_choices("metrics").keys())
    active_metrics = []
    if metrics_plugin_names:
        st.markdown("### Metrics")
        active_metrics = st.multiselect(
            "Active Metrics", metrics_plugin_names,
            default=metrics_plugin_names,
            help="Custom metrics computed each evaluation round"
        )
        for mname in active_metrics:
            mkey = f"custom:{mname}"
            _mparams = _render_plugin_params("metrics", mkey)
            if _mparams:
                plugin_params.setdefault("metrics", {}).update(_mparams)

    with st.expander("Strategies", expanded=True):
        _preset_strats = _pd.get("strategies")
        strategy_keys = []
        for display_name, key in strategy_choices.items():
            default_on = key in (_preset_strats if _preset_strats else DEFAULT_STRATEGIES)
            if st.checkbox(display_name, value=default_on, key=f"strat_{key}"):
                strategy_keys.append(key)
                _sp = _render_plugin_params("strategies", key)
                if _sp:
                    plugin_params.setdefault("strategies", {})[key] = _sp

    st.markdown("---")

    with st.expander("Attack Configuration", expanded=False):
        _p_aidx = _aidx
        if _pd.get("attack_type"):
            _p_aidx = next((i for i, v in enumerate(SUPPORTED_ATTACKS.values()) if v == _pd["attack_type"]), _p_aidx)
        attack_display = st.selectbox("Attack", list(SUPPORTED_ATTACKS.keys()), index=_p_aidx,
                                      label_visibility="collapsed")
        attack_type = SUPPORTED_ATTACKS[attack_display]

        # Attack compatibility warning
        if attack_type != "none" and _task:
            try:
                _atk_compat = ATTACK_COMPATIBLE_TASKS.get(attack_type)
                if _atk_compat and _task not in _atk_compat:
                    st.warning(f"Attack '{attack_type}' is designed for {_atk_compat} tasks.")
            except NameError:
                pass  # _task not defined if dataset_info failed

        attack_params = {}
        _p_mal = _pd.get("malicious_fraction", DEFAULT_MALICIOUS_FRACTION)
        malicious_fraction = _p_mal
        schedule_type, attack_rounds_list = "static", None

        if attack_type != "none":
            malicious_fraction = st.slider("Malicious %", 0.05, 0.9, _p_mal, 0.05,
                                           help="Fraction of clients performing adversarial attacks")
            if attack_type == "gaussian_noise":
                attack_params = {
                    "snr_db": st.slider("SNR (dB)", 1.0, 40.0, 20.0, 1.0,
                                        help="Signal-to-noise ratio; lower = more noise injected"),
                    "attack_fraction": st.slider("Frac", 0.1, 1.0, 1.0, 0.1,
                                                 help="Fraction of data samples to corrupt"),
                }
            elif attack_type == "token_replacement":
                attack_params = {
                    "replacement_fraction": st.slider("Replace frac", 0.1, 1.0, 0.3, 0.05,
                                                    help="Fraction of tokens replaced with random values"),
                }
            elif attack_type == "weight_spiking":
                attack_params = {
                    "magnitude": st.slider("Magnitude", 1.0, 500.0, 100.0, 10.0,
                                           help="Multiplier for spiked weight values"),
                    "spike_fraction": st.slider("Spike frac", 0.01, 1.0, 0.1, 0.01,
                                                help="Fraction of model weights to spike"),
                }
            elif attack_type == "gradient_scaling":
                attack_params = {"scale_factor": st.slider("Scale", 1.0, 100.0, 10.0, 1.0,
                                                           help="Factor by which gradients are scaled up")}
            elif attack_type == "byzantine_perturbation":
                attack_params = {"noise_std": st.slider("Noise std", 0.01, 10.0, 1.0, 0.1,
                                                        help="Standard deviation of Byzantine noise added to updates")}
            # Attack schedule
            sched = st.radio("Schedule", ATTACK_SCHEDULE_TYPES, horizontal=True)
            if "Dynamic" in sched:
                schedule_type = "dynamic"
                rr = st.slider("Active rounds", 1, num_rounds, (1, num_rounds))
                attack_rounds_list = list(range(rr[0], rr[1] + 1))

    with st.expander("Advanced Settings", expanded=False):
        seed = st.number_input("Random Seed", min_value=0, max_value=99999, value=42, step=1,
                               help="Random seed for reproducible experiments")
        weight_decay = st.number_input("Weight Decay", min_value=0.0, max_value=0.1, value=0.0,
                                       step=0.0001, format="%.4f",
                                       help="L2 regularization penalty for optimizer")
        # GPU acceleration — only shown when CUDA is available
        import torch as _torch
        _gpu_available = _torch.cuda.is_available()
        if _gpu_available:
            st.markdown(f"**GPU Acceleration** ({_torch.cuda.get_device_name(0)})")
            use_amp = st.checkbox("Mixed Precision (AMP)",
                help="Use float16 for training — ~1.5-2x faster on tensor core GPUs")
            compile_model = st.checkbox("Compile Model",
                help="torch.compile optimization — ~1.2x faster (first round is slower due to compilation)")
            pin_memory = st.checkbox("Pin Memory", value=True,
                help="Pin DataLoader memory for faster CPU-to-GPU transfer")
        else:
            use_amp = False
            compile_model = False
            pin_memory = False

    # ── Plugins section ──────────────────────────────────────────────
    plugin_summary = discover_all_plugins()
    total_loaded = sum(len(s["loaded"]) for s in plugin_summary.values())
    total_errors = sum(len(s["errors"]) for s in plugin_summary.values())

    err_suffix = f", {total_errors} error{'s' if total_errors != 1 else ''}" if total_errors > 0 else ""
    header = f"Plugins ({total_loaded} loaded{err_suffix})" if total_loaded > 0 or total_errors > 0 else "Plugins"
    with st.expander(header, expanded=False):
        # ── Refresh button ──
        if st.button("Refresh Plugins", key="refresh_plugins", help="Re-scan plugin directories"):
            clear_cache()
            st.rerun()

        # ── Counts summary ──
        if total_loaded > 0:
            counts = []
            for pt in ("datasets", "models", "strategies", "metrics", "losses", "optimizers", "schedulers"):
                n = len(plugin_summary[pt]["loaded"])
                if n > 0:
                    counts.append(f"{pt.title()}: {n}")
            if counts:
                st.caption(" | ".join(counts))

        # ── Show loaded plugins with delete buttons ──
        for pt in ("datasets", "models", "strategies", "metrics", "losses", "optimizers", "schedulers"):
            loaded = plugin_summary[pt]["loaded"]
            if loaded:
                for pname, pmod in loaded:
                    pcol1, pcol2 = st.columns([4, 1])
                    pcol1.caption(f"{pt.title()}: {pname}")
                    # Get the file path from the module
                    plugin_file = getattr(pmod, "__spec__", None)
                    if plugin_file and hasattr(plugin_file, "origin"):
                        fpath = plugin_file.origin
                    else:
                        fpath = None
                    if fpath and pcol2.button("Delete", key=f"del_{pt}_{pname}", type="secondary"):
                        try:
                            os.remove(fpath)
                            clear_cache()
                            st.success(f"Deleted: {pname}")
                            st.rerun()
                        except OSError as e:
                            st.error(f"Could not delete: {e}")

        # ── Errors ──
        for pt in plugin_summary:
            for filepath, error_msg in plugin_summary[pt]["errors"]:
                st.warning(f"**{filepath}**: {error_msg}", icon="⚠️")
        if total_loaded == 0 and total_errors == 0:
            st.caption("Drop .py files in custom/ directories to add plugins")

        # ── Create Plugin section ──
        st.markdown("---")
        st.caption("Create New Plugin")
        create_cols = st.columns([2, 2, 1])
        _create_type = create_cols[0].selectbox(
            "Type", ["datasets", "models", "strategies", "losses", "optimizers", "metrics", "schedulers"],
            key="create_plugin_type", label_visibility="collapsed"
        )
        _create_name = create_cols[1].text_input(
            "Name", placeholder="my_plugin", key="create_plugin_name", label_visibility="collapsed"
        )
        if create_cols[2].button("Create", key="create_plugin_btn", type="secondary"):
            if _create_name and _create_name.strip():
                import shutil
                safe_name = _create_name.strip().lower().replace(" ", "_").replace("-", "_")
                if not safe_name.isidentifier():
                    st.error(f"Invalid plugin name: '{safe_name}'. Use only letters, numbers, and underscores.")
                else:
                    _proj_root = Path(os.path.dirname(os.path.abspath(__file__)))
                    template_src = _proj_root / "custom" / _create_type / "_template.py"
                    dest = _proj_root / "custom" / _create_type / f"{safe_name}.py"
                    if dest.exists():
                        st.warning(f"Plugin already exists: {dest}")
                    elif template_src.exists():
                        shutil.copy2(template_src, dest)
                        clear_cache()
                        st.success(f"Created: `{dest}` — edit this file to implement your plugin")
                        st.rerun()
                    else:
                        st.error(f"Template not found: {template_src}")
            else:
                st.warning("Enter a plugin name")

    # ── Load Experiment ─────────────────────────────────────────────
    st.markdown("---")
    with st.expander("Load Experiment", expanded=False):
        st.caption("Load experiment configs from a script. Runs execute with full live visualization.")
        uploaded = st.file_uploader("Experiment JSON", type=["json"],
                                     key="load_experiment_file")
        if uploaded is not None:
            try:
                from api.experiment import Experiment as _Exp
                import tempfile, os as _os
                with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="wb") as tmp:
                    tmp.write(uploaded.read())
                    tmp_path = tmp.name
                exp_name, loaded_runs = _Exp.load_configs(tmp_path)
                _os.unlink(tmp_path)

                if loaded_runs:
                    st.success(f"**{exp_name}** — {len(loaded_runs)} runs loaded")
                    run_names = [name for name, _ in loaded_runs]
                    selected_run = st.selectbox(
                        "Select run to execute",
                        run_names,
                        key="loaded_run_select",
                    )
                    selected_config = next(cfg for n, cfg in loaded_runs if n == selected_run)

                    # Show config summary
                    st.caption(
                        f"Model: `{selected_config.model_name}` | "
                        f"Dataset: `{selected_config.dataset_name}` | "
                        f"Clients: {selected_config.num_clients} | "
                        f"Rounds: {selected_config.num_rounds} | "
                        f"Strategies: {', '.join(selected_config.strategies)}"
                    )
                    if selected_config.attack.attack_type != "none":
                        st.caption(
                            f"Attack: `{selected_config.attack.attack_type}` | "
                            f"Malicious: {selected_config.attack.malicious_fraction:.0%}"
                        )

                    if st.button("Run this experiment", key="run_loaded_btn", type="primary"):
                        # Inject into session state so the main run block picks it up
                        st.session_state["_loaded_config"] = selected_config
                        st.session_state["_loaded_run_name"] = selected_run
                        st.rerun()
                else:
                    st.warning("No runs found in this file.")
            except Exception as e:
                st.error(f"Failed to load: {e}")

    st.markdown("---")
    run_button = st.button("Run Simulation", type="primary", width="stretch")

# ── Tabs ─────────────────────────────────────────────────────────────
tab_sim, tab_results, tab_analysis, tab_anomaly, tab_docs = st.tabs(["Simulation", "Results", "Analysis", "Client Metrics", "Docs"])

# ── Run ──────────────────────────────────────────────────────────────
# Check if a loaded experiment config should execute
_loaded_config = st.session_state.pop("_loaded_config", None)
_loaded_run_name = st.session_state.pop("_loaded_run_name", None)
_should_run = run_button or _loaded_config is not None

if _should_run:
    if _loaded_config is not None:
        # Use the config from the loaded experiment
        config = _loaded_config
        st.toast(f"Running loaded experiment: {_loaded_run_name}")
    elif not strategy_keys:
        st.error("Select at least one strategy.")
        st.stop()
    else:
        config = SimulationConfig(
            model_name=model_name, dataset_name=dataset_name,
            num_clients=num_clients, num_rounds=num_rounds,
            local_epochs=local_epochs, learning_rate=learning_rate,
            partition_type=partition_type, alpha=alpha, val_split=val_split,
            strategies=strategy_keys, seed=seed,
            attack=AttackConfig(attack_type=attack_type, malicious_fraction=malicious_fraction,
                                attack_params=attack_params, schedule_type=schedule_type,
                                attack_rounds=attack_rounds_list),
            optimizer=optimizer_name,
            loss_function=loss_name,
            weight_decay=weight_decay,
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=max(1, min(2, num_clients)),
            min_evaluate_clients=max(1, min(2, num_clients)),
            min_available_clients=max(1, min(2, num_clients)),
            lr_scheduler=scheduler_name,
            lr_scheduler_params=lr_scheduler_params,
            active_metrics=active_metrics,
            plugin_params=plugin_params,
            max_parallel_clients=parallel_clients,
            use_amp=use_amp,
            compile_model=compile_model,
            pin_memory=pin_memory,
        )

    # Validate config and surface errors to user (for all configs)
    try:
        _validate_config(config)
    except ValueError as e:
        st.error(f"Invalid configuration: {e}")
        st.stop()

    # Compatibility validation (non-fatal on dry-run failures)
    try:
        compat_errors, compat_warnings = _validate_compatibility(config)
        for w in compat_warnings:
            st.warning(w)
        if compat_errors:
            for e in compat_errors:
                st.error(e)
            st.stop()
    except Exception as ve:
        st.warning(f"Compatibility check skipped: {ve}")

    with tab_sim:
        ph_status = st.empty()
        progress_bar = st.progress(0)
        col_l, col_r = st.columns(2)
        ph_loss = col_l.empty()
        ph_acc = col_r.empty()

        _cfg_attack = config.attack.attack_type if hasattr(config, 'attack') else "none"
        with st.expander("Client Activity Grid", expanded=False):
            st.markdown(
                '<div class="grid-legend">'
                '<span><span class="dot" style="background:#7FB5A0"></span>Benign</span>'
                '<span><span class="dot" style="background:#D4726A"></span>Attacked</span>'
                '<span><span class="dot" style="background:#D4A76A"></span>Excluded (caught)</span>'
                '<span><span class="dot" style="background:#B088C4"></span>Excluded (FP)</span>'
                '<span><span class="dot" style="background:#6B7280"></span>Idle</span>'
                '</div>', unsafe_allow_html=True,
            )
            ph_grid = st.empty()

        all_losses, all_accuracies, grid_data = {}, {}, {}
        _chart_counter = [0]  # mutable counter for unique chart keys

        # Determine task-specific metric names
        _sim_task_type = _get_task_type(config.dataset_name)
        _primary_metric_name, _loss_name = _task_metric_labels(_sim_task_type)

        def round_callback(event: RoundEvent):
            s = event.strategy_name
            sd = _html.escape(STRATEGY_DISPLAY_NAMES.get(s, s))
            if s not in all_losses:
                all_losses[s], all_accuracies[s], grid_data[s] = [], [], []
            all_losses[s].append(event.loss)
            all_accuracies[s].append(event.accuracy)

            # For non-classification tasks: override 0.0 accuracy with primary eval metric
            if "classification" not in _sim_task_type and event.custom_metrics:
                for mk, mv in event.custom_metrics.items():
                    if mk.startswith("eval/") and mk not in ("eval/loss",):
                        all_accuracies[s][-1] = mv
                        break

            # Merge training status with exclusion info from the strategy
            _round_grid = []
            for c in range(config.num_clients):
                _st = event.client_statuses.get(c, "idle")
                if _st == "attacked" and c in event.client_excluded:
                    _st = "excluded"  # attacked AND caught by the strategy
                elif _st == "benign" and c in event.client_excluded:
                    _st = "false_positive"  # benign but incorrectly excluded
                _round_grid.append(_st)
            grid_data[s].append(_round_grid)

            total = (event.num_rounds + 1) * event.num_strategies
            done = event.strategy_idx * (event.num_rounds + 1) + event.round_num + 1
            progress_bar.progress(min(done / total, 1.0) if total else 0)

            if done > 0 and event.elapsed > 0:
                avg_round_time = event.elapsed / done
                remaining = avg_round_time * (total - done)
                eta_str = f" &middot; ETA {remaining:.0f}s"
            else:
                eta_str = ""

            metrics_str = ""
            if event.custom_metrics:
                # Escape plugin-authored metric keys — they reach unsafe_allow_html below.
                parts = [
                    f"{_html.escape(mk.split('/')[-1])} {mv:.3f}"
                    for mk, mv in event.custom_metrics.items()
                ]
                if parts:
                    metrics_str = " &middot; " + " &middot; ".join(parts)

            # Use the primary metric value for status bar display
            _status_metric_val = all_accuracies[s][-1] if all_accuracies[s] else event.accuracy
            ph_status.markdown(
                f'<div class="status-bar"><strong>{sd}</strong> &middot; '
                f'Round {event.round_num}/{event.num_rounds} &middot; '
                f'<span class="loss">{_loss_name} {event.loss:.4f}</span> &middot; '
                f'<span class="acc">{_primary_metric_name} {_status_metric_val:.4f}</span>'
                f'{metrics_str} &middot; '
                f'{event.elapsed:.1f}s{eta_str}</div>', unsafe_allow_html=True,
            )
            _cc = _chart_counter[0]
            _chart_counter[0] += 1
            ph_loss.plotly_chart(plot_live_loss(all_losses, config.num_rounds, loss_name=_loss_name),
                                width="stretch", key=f"ll_{_cc}")
            ph_acc.plotly_chart(plot_live_accuracy(all_accuracies, config.num_rounds, metric_name=_primary_metric_name),
                               width="stretch", key=f"la_{_cc}")
            ph_grid.plotly_chart(
                plot_client_grid(grid_data[s], config.num_clients, config.num_rounds, event.round_num),
                width="stretch", key=f"cg_{_cc}")

        try:
            results = run_simulation(config, round_callback=round_callback)
            st.session_state["results"] = results
            st.session_state["config"] = config
        except Exception as e:
            st.error(f"Simulation failed: {e}")
            progress_bar.empty()
            ph_status.empty()
            st.stop()
        progress_bar.progress(1.0)
        ph_status.markdown(
            '<div class="status-bar"><strong>Complete</strong></div>',
            unsafe_allow_html=True,
        )

# ── Results Tab ──────────────────────────────────────────────────────
with tab_results:
    if "results" in st.session_state:
        results = st.session_state["results"]

        # Determine task type for metric labels
        _results_cfg = st.session_state.get("config")
        _results_task_type = _get_task_type(_results_cfg.dataset_name) if _results_cfg else "image_classification"
        _res_metric_label, _res_loss_label = _task_metric_labels(_results_task_type)

        rows = []
        for r in results:
            n = STRATEGY_DISPLAY_NAMES.get(r.strategy_name, r.strategy_name)
            fl = r.round_losses[-1] if r.round_losses else None
            rf1 = r.anomaly_summary.get("cumulative_f1", None) if r.anomaly_summary else None

            # For non-classification: use primary eval metric instead of accuracy
            if "classification" not in _results_task_type and r.custom_metrics:
                _primary_vals = None
                for mk, vals in r.custom_metrics.items():
                    if mk.startswith("eval/") and mk != "eval/loss" and isinstance(vals, list):
                        _primary_vals = vals
                        break
                fa = _primary_vals[-1] if _primary_vals else None
                ba = max(_primary_vals) if _primary_vals else None
            else:
                fa = r.round_accuracies[-1] if r.round_accuracies else None
                ba = max(r.round_accuracies) if r.round_accuracies else None

            rows.append({
                "Strategy": n,
                f"Final {_res_loss_label}": f"{fl:.4f}" if fl is not None else "-",
                f"Final {_res_metric_label}": f"{fa:.4f}" if fa is not None else "-",
                f"Best {_res_metric_label}": f"{ba:.4f}" if ba is not None else "-",
                "Removal F1": f"{rf1:.3f}" if rf1 is not None else "-",
                "Time": f"{r.total_time:.1f}s",
            })
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

        # Delta-from-baseline toggle
        if results and len(results) > 1:
            st.markdown("---")
            baseline_options = [r.strategy_name for r in results]
            baseline_display = {STRATEGY_DISPLAY_NAMES.get(s, s): s for s in baseline_options}

            baseline_choice = st.selectbox(
                "Compare relative to baseline",
                ["None (absolute values)"] + list(baseline_display.keys()),
                key="baseline_selector",
                help="Show metrics as delta from selected baseline strategy"
            )

            if baseline_choice != "None (absolute values)":
                baseline_key = baseline_display[baseline_choice]
                baseline_result = next(r for r in results if r.strategy_name == baseline_key)
                baseline_final_loss = baseline_result.round_losses[-1] if baseline_result.round_losses else 0
                baseline_final_acc = baseline_result.round_accuracies[-1] if baseline_result.round_accuracies else 0

                delta_rows = []
                for r in results:
                    if r.strategy_name == baseline_key:
                        continue
                    dloss = (r.round_losses[-1] if r.round_losses else 0) - baseline_final_loss
                    dacc = (r.round_accuracies[-1] if r.round_accuracies else 0) - baseline_final_acc
                    delta_rows.append({
                        "Strategy": STRATEGY_DISPLAY_NAMES.get(r.strategy_name, r.strategy_name),
                        "Loss Delta": f"{dloss:+.4f}",
                        "Metric Delta": f"{dacc:+.4f}",
                        "Time": f"{r.total_time:.1f}s",
                    })

                if delta_rows:
                    st.dataframe(pd.DataFrame(delta_rows), width="stretch")

        dc1, dc2 = st.columns(2)
        csv_rows = []
        for r in results:
            n = STRATEGY_DISPLAY_NAMES.get(r.strategy_name, r.strategy_name)
            for i in range(max(len(r.round_losses), len(r.round_accuracies))):
                csv_rows.append({
                    "strategy": n, "round": i,
                    "loss": r.round_losses[i] if i < len(r.round_losses) else None,
                    "accuracy": r.round_accuracies[i] if i < len(r.round_accuracies) else None,
                })
        if csv_rows:
            dc1.download_button("Export CSV", pd.DataFrame(csv_rows).to_csv(index=False),
                                file_name="fl_results.csv", mime="text/csv")
        if "config" in st.session_state:
            cfg = st.session_state["config"]
            def _ser(o):
                if dataclasses.is_dataclass(o) and not isinstance(o, type):
                    return dataclasses.asdict(o)
                if isinstance(o, set): return sorted(o)
                if isinstance(o, np.ndarray): return o.tolist()
                if isinstance(o, (np.integer,)): return int(o)
                if isinstance(o, (np.floating,)): return float(o)
                raise TypeError(type(o))
            config_dict = {k: v for k, v in vars(cfg).items()}
            # Remove legacy reputation fields from export
            for key in ["reputation_truth_threshold", "reputation_selection_fraction",
                         "reputation_initial_reputation", "reputation_trust_exclusion_threshold",
                         "reputation_warmup_rounds", "reputation_smoothing_beta"]:
                config_dict.pop(key, None)
            dc2.download_button("Export Config", json.dumps(config_dict, indent=2, default=_ser),
                                file_name="fl_config.json", mime="application/json")

        # Full report export (config + results + anomaly metrics)
        st.markdown("---")
        st.markdown("**Full Experiment Report**")
        if "config" in st.session_state:
            cfg = st.session_state["config"]
            report = {
                "config": dataclasses.asdict(cfg),
                "results": [],
            }
            for r in results:
                report["results"].append({
                    "strategy": r.strategy_name,
                    "final_loss": r.round_losses[-1] if r.round_losses else None,
                    "final_accuracy": r.round_accuracies[-1] if r.round_accuracies else None,
                    "best_accuracy": max(r.round_accuracies) if r.round_accuracies else None,
                    "total_time": r.total_time,
                    "round_losses": r.round_losses,
                    "round_accuracies": r.round_accuracies,
                    "anomaly_summary": r.anomaly_summary,
                    "anomaly_history": r.anomaly_history,
                })
            def _ser2(o):
                if isinstance(o, np.ndarray): return o.tolist()
                if isinstance(o, (np.integer,)): return int(o)
                if isinstance(o, (np.floating,)): return float(o)
                if isinstance(o, float) and (o != o or o == float('inf') or o == float('-inf')): return None
                if isinstance(o, set): return sorted(o)
                raise TypeError(type(o))
            st.download_button(
                "Download Full Report (JSON)",
                json.dumps(report, indent=2, default=_ser2),
                file_name="fedsim_report.json",
                mime="application/json",
                width="stretch",
            )

        # HTML Report Export
        _html_cfg = st.session_state.get("config")
        if _html_cfg:
            st.markdown("**Interactive Report**")
            from report_html import generate_html_report
            html_report = generate_html_report(_html_cfg, results)
            st.download_button(
                "Download HTML Report",
                html_report,
                file_name="fedsim_report.html",
                mime="text/html",
                key="download_html_report",
            )

        # Custom Metrics charts
        import types as _types
        if results and any(r.custom_metrics for r in results):
            st.markdown("### Custom Metrics")
            all_metric_keys = set()
            for r in results:
                all_metric_keys.update((r.custom_metrics or {}).keys())
            for mk in sorted(all_metric_keys):
                chart_type = "line"
                plugin_name = mk.split("/")[0] if "/" in mk else mk
                for mname, mmod in discover_plugins("metrics").items():
                    if isinstance(mmod, _types.ModuleType) and getattr(mmod, "NAME", "") == plugin_name:
                        chart_type = getattr(mmod, "CHART_TYPE", "line")
                        break
                st.plotly_chart(plot_custom_metric(results, mk, chart_type), width="stretch")

        # ── Experiment History ─────────────────────────────────────────
        st.markdown("---")
        st.markdown("### Experiment History")

        from experiment_store import save_experiment, list_experiments, load_experiment, delete_experiment

        # Save current experiment
        _exp_cfg = st.session_state.get("config")
        if results and _exp_cfg:
            save_cols = st.columns([3, 1])
            exp_name = save_cols[0].text_input("Experiment name",
                value=f"{_exp_cfg.model_name}_{_exp_cfg.dataset_name}",
                key="save_exp_name", label_visibility="collapsed",
                placeholder="Name this experiment...")
            if save_cols[1].button("Save", key="save_experiment_btn"):
                path = save_experiment(_exp_cfg, results, name=exp_name)
                st.success(f"Saved to {path}")
                st.rerun()

        # List past experiments
        past_experiments = list_experiments()
        if past_experiments:
            st.markdown("**Saved Experiments**")
            for i, exp in enumerate(past_experiments[:10]):  # show last 10
                ec1, ec2, ec3 = st.columns([4, 1, 1])
                ec1.caption(f"**{exp['name']}** — {exp['config_summary']}")

                if ec2.button("Compare", key=f"compare_{i}"):
                    st.session_state["compare_experiment"] = exp["filepath"]
                    st.rerun()

                if ec3.button("Delete", key=f"delete_exp_{i}", type="secondary"):
                    delete_experiment(exp["filepath"])
                    st.rerun()

            # Comparison view
            if "compare_experiment" in st.session_state and results:
                compare_path = st.session_state["compare_experiment"]
                try:
                    past = load_experiment(compare_path)
                    st.markdown(f"### Comparing with: {past['name']}")

                    # Config diff
                    with st.expander("Config Differences", expanded=False):
                        current_cfg = dataclasses.asdict(_exp_cfg)
                        past_cfg = past.get("config", {})
                        diffs = []
                        all_keys = set(current_cfg.keys()) | set(past_cfg.keys())
                        for k in sorted(all_keys):
                            cv = current_cfg.get(k)
                            pv = past_cfg.get(k)
                            if str(cv) != str(pv):
                                diffs.append(f"**{k}**: `{pv}` → `{cv}`")
                        if diffs:
                            for d in diffs:
                                st.markdown(d)
                        else:
                            st.caption("Identical configurations")

                    # Side-by-side loss comparison
                    import plotly.graph_objects as go
                    fig = go.Figure()
                    # Current results
                    for r in results:
                        fig.add_trace(go.Scatter(
                            x=list(range(len(r.round_losses))),
                            y=r.round_losses,
                            mode="lines+markers",
                            name=f"Current: {r.strategy_name}",
                            line=dict(width=2),
                        ))
                    # Past results
                    for r in past.get("results", []):
                        fig.add_trace(go.Scatter(
                            x=list(range(len(r["round_losses"]))),
                            y=r["round_losses"],
                            mode="lines",
                            name=f"Past: {r['strategy_name']}",
                            line=dict(width=2, dash="dash"),
                        ))
                    fig.update_layout(
                        title="Loss Comparison (Current vs Past)",
                        xaxis_title="Round", yaxis_title="Loss",
                        template="plotly_dark", height=350,
                        margin=dict(t=40, b=50, l=60, r=20),
                    )
                    st.plotly_chart(fig, width="stretch")

                    if st.button("Clear Comparison", key="clear_compare"):
                        del st.session_state["compare_experiment"]
                        st.rerun()

                except Exception as e:
                    st.error(f"Could not load experiment: {e}")
                    del st.session_state["compare_experiment"]
        else:
            st.caption("No saved experiments yet. Run a simulation and click Save.")

    else:
        st.markdown(
            '<div class="empty-state">'
            '<p>No results yet</p>'
            '<p class="hint">Configure parameters in the sidebar and click Run Simulation.<br>'
            'This tab shows summary metrics, CSV/JSON export, and report generation.</p>'
            '</div>', unsafe_allow_html=True)

# ── Analysis Tab ─────────────────────────────────────────────────────
with tab_analysis:
    if "results" in st.session_state:
        results = st.session_state["results"]
        cfg = st.session_state.get("config")
        nr = cfg.num_rounds if cfg else 10

        # Determine malicious clients
        malicious = _extract_malicious_clients(results)

        # Determine task type for metric labels in analysis charts
        _analysis_task_type = _get_task_type(cfg.dataset_name) if cfg else "image_classification"
        _analysis_metric_name, _ = _task_metric_labels(_analysis_task_type)

        _analysis_chart_label = f"Strategy {_analysis_metric_name} Comparison"
        chart_option = st.selectbox("Select visualization", [
            _analysis_chart_label,
            "Trust / Reputation Heatmap",
            "Attack Impact Matrix",
            "Client PCA",
        ], key="analysis_chart")

        if chart_option == _analysis_chart_label:
            fig = plot_accuracy_surface(results, nr, metric_name=_analysis_metric_name)
            st.plotly_chart(fig, width="stretch")

        elif chart_option == "Trust / Reputation Heatmap":
            col1, col2 = st.columns([1, 3])
            score_type = col1.radio("Metric", ["Trust", "Reputation"], key="analysis_score_type")
            strat_names = [STRATEGY_DISPLAY_NAMES.get(r.strategy_name, r.strategy_name) for r in results]
            selected_strat = col1.selectbox("Strategy", strat_names, key="analysis_strat")
            sel_idx = strat_names.index(selected_strat)
            r = results[sel_idx]
            hist = r.trust_history if score_type == "Trust" else r.reputation_history
            fig = plot_trust_reputation_landscape(hist or {}, malicious, nr, score_type)
            st.plotly_chart(fig, width="stretch")

        elif chart_option == "Attack Impact Matrix":
            if "benchmark_results" in st.session_state:
                bm = st.session_state["benchmark_results"]
                sn = [STRATEGY_DISPLAY_NAMES.get(s, s) for s in cfg.strategies]
                an = [k for k in SUPPORTED_ATTACKS.keys() if SUPPORTED_ATTACKS[k] != "none"]
                fig = plot_attack_impact(bm, sn, an)
                st.plotly_chart(fig, width="stretch")
            else:
                st.info("Run Full Benchmark from below to generate the attack impact matrix.")
                benchmark_running = st.session_state.get("benchmark_running", False)
                if st.button("Run Full Benchmark", disabled=benchmark_running):
                    st.session_state["benchmark_running"] = True
                    try:
                        from configs.defaults import DEFAULT_ATTACK_PARAMS
                        bm = {}
                        atks = {k: v for k, v in SUPPORTED_ATTACKS.items() if v != "none"}
                        pb = st.progress(0)
                        total = len(atks) * len(cfg.strategies)
                        idx = 0
                        for aname, akey in atks.items():
                            bc = SimulationConfig(
                                model_name=cfg.model_name, dataset_name=cfg.dataset_name,
                                num_clients=cfg.num_clients, num_rounds=cfg.num_rounds,
                                local_epochs=cfg.local_epochs, learning_rate=cfg.learning_rate,
                                partition_type=cfg.partition_type, alpha=cfg.alpha,
                                strategies=cfg.strategies,
                                attack=AttackConfig(attack_type=akey,
                                                    malicious_fraction=cfg.attack.malicious_fraction,
                                                    attack_params=DEFAULT_ATTACK_PARAMS.get(akey, {})),
                                plugin_params=cfg.plugin_params,
                            )
                            for r in run_simulation(bc):
                                sd = STRATEGY_DISPLAY_NAMES.get(r.strategy_name, r.strategy_name)
                                bm[(aname, sd)] = r.round_accuracies[-1] if r.round_accuracies else 0
                                idx += 1
                                pb.progress(idx / total)
                        st.session_state["benchmark_results"] = bm
                    finally:
                        st.session_state["benchmark_running"] = False
                    st.rerun()

        elif chart_option == "Client PCA":
            params, reps = None, None
            for r in results:
                if r.final_client_params:
                    params = r.final_client_params
                    if r.reputation_history:
                        reps = {c: s[-1] for c, s in r.reputation_history.items() if s}
                    break
            if params:
                fig = plot_client_pca(params, malicious, reps)
                st.plotly_chart(fig, width="stretch")
            else:
                st.info("PCA requires final-round parameters. Run a simulation first.")
    else:
        st.markdown(
            '<div class="empty-state">'
            '<p>No analysis data yet</p>'
            '<p class="hint">Configure parameters in the sidebar and click Run Simulation.<br>'
            'This tab shows accuracy comparisons, trust heatmaps, and client PCA.</p>'
            '</div>', unsafe_allow_html=True)

# ── Client Metrics Tab ────────────────────────────────────────────
with tab_anomaly:
    if "results" in st.session_state:
        results = st.session_state["results"]
        cfg = st.session_state.get("config")
        nr = cfg.num_rounds if cfg else 10
        nc = cfg.num_clients if cfg else 10
        malicious = _extract_malicious_clients(results)

        # Strategy selector
        strat_names_display = [STRATEGY_DISPLAY_NAMES.get(r.strategy_name, r.strategy_name) for r in results]
        sel_col, metric_col = st.columns([1, 2])
        selected_strat = sel_col.selectbox("Strategy", strat_names_display, key="anomaly_strat")
        sel_idx = strat_names_display.index(selected_strat)
        sel_result = results[sel_idx]

        # Build list of available per-client metrics for this strategy
        available_metrics = {}
        if sel_result.trust_history:
            available_metrics["Trust Scores"] = sel_result.trust_history
        if sel_result.reputation_history:
            available_metrics["Reputation Scores"] = sel_result.reputation_history
        if sel_result.strategy_scores_history:
            # Reshape strategy_scores_history (list of per-round dicts) into per-client lists
            _score_by_client = {}
            for rnd_scores in sel_result.strategy_scores_history:
                for cid, score in rnd_scores.items():
                    _score_by_client.setdefault(cid, []).append(score)
            if _score_by_client:
                available_metrics["Strategy Scores"] = _score_by_client

        # Metric picker
        metric_options = list(available_metrics.keys())
        if not metric_options:
            st.caption("No per-client metrics available for this strategy.")
        else:
            selected_metric = metric_col.selectbox(
                "Client Metric", metric_options, key="client_metric_select",
                help="Per-client metric to plot over rounds")
            history = available_metrics[selected_metric]

            # ── Line chart: per-client metric over rounds ──
            import plotly.graph_objects as go
            fig = go.Figure()
            from visualization import fedsim_layout_defaults, COLOR_BENIGN, COLOR_ATTACKED, COLOR_TEXT_MUTED
            for cid in sorted(history.keys()):
                vals = history[cid]
                is_mal = cid in malicious
                fig.add_trace(go.Scatter(
                    x=list(range(1, len(vals) + 1)),
                    y=vals,
                    mode="lines+markers",
                    name=f"C{cid}{'*' if is_mal else ''}",
                    line=dict(
                        width=2 if is_mal else 1.5,
                        color=COLOR_ATTACKED if is_mal else COLOR_BENIGN,
                        dash="dash" if is_mal else "solid",
                    ),
                    marker=dict(size=4),
                    opacity=0.9 if is_mal else 0.6,
                ))
            fig.update_layout(
                **fedsim_layout_defaults(),
                title=f"{selected_metric} per Client",
                xaxis_title="Round", yaxis_title=selected_metric,
                template="plotly_dark", height=380,
                margin=dict(t=40, b=50, l=60, r=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02,
                            xanchor="right", x=1, font=dict(size=9)),
            )
            if selected_metric in ("Trust Scores", "Reputation Scores"):
                fig.update_yaxes(range=[0, 1.05])
            st.plotly_chart(fig, width="stretch")

        # ── Exclusion metrics (only for strategies that exclude clients) ──
        has_exclusions = sel_result.anomaly_history and any(
            r.get("excluded") for r in sel_result.anomaly_history
        )
        if has_exclusions:
            with st.expander("Exclusion Metrics", expanded=False):
                s = sel_result.anomaly_summary
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("Removal F1", f"{s.get('cumulative_f1', 0):.3f}")
                mc2.metric("Precision", f"{s.get('cumulative_precision', 0):.3f}")
                mc3.metric("Recall", f"{s.get('cumulative_recall', 0):.3f}")

                ex_col1, ex_col2 = st.columns(2)
                with ex_col1:
                    fig = plot_removal_f1_over_rounds(sel_result.anomaly_history, nr)
                    st.plotly_chart(fig, width="stretch")
                with ex_col2:
                    fig = plot_exclusion_timeline(sel_result.anomaly_history, nc, nr, malicious)
                    st.plotly_chart(fig, width="stretch")
    else:
        st.markdown(
            '<div class="empty-state">'
            '<p>No simulation data yet</p>'
            '<p class="hint">Run a simulation to see per-client metrics here.</p>'
            '</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# DOCS TAB
# ══════════════════════════════════════════════════════════════════════
with tab_docs:
    st.markdown("""
    <style>
    .docs-hero {
        background: linear-gradient(135deg, #1C1F26 0%, #232730 50%, #2D3140 100%);
        border-radius: 16px;
        padding: 3rem 2.5rem;
        margin-bottom: 2rem;
        color: white;
        position: relative;
        overflow: hidden;
    }
    .docs-hero::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -20%;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(127,181,160,0.12) 0%, transparent 70%);
        border-radius: 50%;
    }
    .docs-hero h1 {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    .docs-hero p {
        font-size: 1.1rem;
        opacity: 0.8;
        max-width: 600px;
    }
    .docs-section {
        background: #232730;
        border-radius: 12px;
        padding: 1.8rem 2rem;
        margin-bottom: 1.5rem;
        border-left: 4px solid #7FB5A0;
    }
    .docs-section h3 {
        margin-top: 0;
        color: #C8CCD4;
        font-size: 1.3rem;
    }
    .docs-section code {
        background: rgba(127,181,160,0.15);
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 0.9em;
    }
    .docs-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1rem;
        margin: 1rem 0;
    }
    .docs-card {
        background: white;
        border-radius: 10px;
        padding: 1.2rem 1.4rem;
        border: 1px solid #2D3140;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .docs-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(127,181,160,0.08);
    }
    .docs-card h4 {
        margin: 0 0 0.4rem 0;
        color: #7FB5A0;
        font-size: 1rem;
    }
    .docs-card p {
        margin: 0;
        font-size: 0.9rem;
        color: #8B919E;
    }
    .docs-badge {
        display: inline-block;
        background: #7FB5A0;
        color: white;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-left: 8px;
        vertical-align: middle;
    }
    </style>

    <div class="docs-hero">
        <h1>FEDSIM Documentation</h1>
        <p>Simulation, visualization, and scripting for federated learning research.
        Use the dashboard for interactive exploration or the Python API for reproducible experiments.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Quick Start ───────────────────────────────────────────────
    st.markdown("""
    <div class="docs-section">
    <h3>Quick Start</h3>
    <div class="docs-grid">
        <div class="docs-card">
            <h4>1. Configure</h4>
            <p>Select model, dataset, strategies, and attack parameters in the sidebar.</p>
        </div>
        <div class="docs-card">
            <h4>2. Run</h4>
            <p>Click <strong>Run Simulation</strong>. Watch live training curves in the Simulation tab.</p>
        </div>
        <div class="docs-card">
            <h4>3. Analyze</h4>
            <p>Compare strategies in Results. Inspect anomaly detection in the Anomaly tab.</p>
        </div>
        <div class="docs-card">
            <h4>4. Export</h4>
            <p>Download CSV results, JSON configs, or full HTML reports from the Results tab.</p>
        </div>
    </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Scripting API ─────────────────────────────────────────────
    st.markdown("### Scripting API <span class='docs-badge'>NEW</span>",
                unsafe_allow_html=True)
    st.markdown("Script multi-run experiments programmatically with checkpointing and PDF reports.")

    with st.expander("Basic Experiment", expanded=True):
        st.code('''from api import Experiment, Report
from simulation.runner import SimulationConfig, AttackConfig

exp = Experiment("My First Experiment")

exp.add_run("FedAvg Baseline", SimulationConfig(
    model_name="cnn", dataset_name="mnist",
    num_clients=10, num_rounds=20, local_epochs=3,
    learning_rate=0.01, strategies=["fedavg"],
    attack=AttackConfig(),
))

exp.add_run("Under Attack", SimulationConfig(
    model_name="cnn", dataset_name="mnist",
    num_clients=10, num_rounds=20, local_epochs=3,
    learning_rate=0.01, strategies=["fedavg", "krum"],
    attack=AttackConfig(
        attack_type="label_flipping",
        malicious_fraction=0.3,
    ),
))

# Run with auto-checkpointing
results = exp.run(checkpoint_path="results/my_experiment.json")

# Access results
print(results.final_accuracy("FedAvg Baseline"))
print(results.final_accuracy("Under Attack", strategy_idx=1))  # Krum
''', language="python")

    with st.expander("Parameter Sweeps"):
        st.code('''from api import Experiment
from simulation.runner import SimulationConfig, AttackConfig

exp = Experiment("Non-IID Sweep")

# Sweep over heterogeneity levels
for alpha in [0.1, 0.5, 1.0, 5.0]:
    exp.add_run(f"alpha={alpha}", SimulationConfig(
        model_name="resnet18", dataset_name="cifar10",
        num_clients=10, num_rounds=30, local_epochs=3,
        learning_rate=0.01,
        strategies=["fedavg", "krum", "trimmed_mean"],
        partition_type="non_iid", alpha=alpha,
        attack=AttackConfig(),
        use_amp=True, pin_memory=True,  # GPU acceleration
    ))

results = exp.run(checkpoint_path="results/noniid_sweep.json")

# Compare final accuracy across alpha values
for name in results.names:
    print(f"{name}: {results.final_accuracy(name):.4f}")
''', language="python")

    with st.expander("PDF Report Generation"):
        st.code('''from api import Experiment, Report
from simulation.runner import SimulationConfig, AttackConfig

# ... run experiments (see above) ...

report = Report("Byzantine Robustness Benchmark")

# Add analysis pages
report.add_text("Introduction",
    "This report compares aggregation strategies under "
    "label flipping attacks with 30% malicious clients.")

# Convergence curves for selected runs
report.add_convergence_plot(results,
    names=["FedAvg Baseline", "Under Attack"],
    title="FedAvg: Clean vs Attacked")

# Summary table
report.add_accuracy_table(results)

# Custom heatmap (e.g., strategy x attack matrix)
report.add_heatmap(
    data=[[0.95, 0.60], [0.94, 0.93], [0.93, 0.91]],
    row_labels=["FedAvg", "Krum", "Trimmed Mean"],
    col_labels=["No Attack", "Label Flipping"],
    title="Attack Impact Matrix",
)

report.save_pdf("my_report.pdf")
''', language="python")

    with st.expander("Custom Strategy Plugin (FedProx example)"):
        st.code('''# Save as custom/strategies/fedprox.py — auto-discovered by FEDSIM

import torch
import torch.nn.functional as F
from fl_core import FedAvg

NAME = "FedProx"
DESCRIPTION = "FedAvg + proximal term (Li et al., MLSys 2020)"
PARAMS = {
    "mu": {"type": "float", "default": 0.01, "min": 0.0,
           "max": 10.0, "label": "Proximal mu"},
}

def build(initial_parameters, num_clients, num_malicious=0, **kwargs):
    filtered = {k: v for k, v in kwargs.items() if k != "mu"}
    return FedAvg(initial_parameters=initial_parameters, **filtered)

def train_step(model, batch, optimizer, device, mu=0.01, **kwargs):
    """Strategy plugins can override client training."""
    if not hasattr(model, '_fedprox_global_params'):
        model._fedprox_global_params = [
            p.clone().detach() for p in model.parameters()
        ]

    images, labels = batch
    images, labels = images.to(device), labels.to(device)

    optimizer.zero_grad()
    output = model(images)
    loss = F.cross_entropy(output, labels)

    # Proximal regularization
    if mu > 0:
        proximal = sum(
            ((p - gp) ** 2).sum()
            for p, gp in zip(model.parameters(),
                             model._fedprox_global_params)
        )
        loss = loss + (mu / 2.0) * proximal

    loss.backward()
    optimizer.step()

    return {"loss": loss.item(),
            "accuracy": (output.argmax(1) == labels).float().mean().item()}
''', language="python")

    # ── Architecture Reference ────────────────────────────────────
    st.markdown("### Architecture Reference")

    st.markdown("""
    <div class="docs-section">
    <h3>Core Components</h3>
    <div class="docs-grid">
        <div class="docs-card">
            <h4>Simulation Engine</h4>
            <p><code>simulation/runner.py</code> — Orchestrates FL rounds, client training, aggregation, anomaly tracking. Supports parallel client training.</p>
        </div>
        <div class="docs-card">
            <h4>Strategies</h4>
            <p><code>strategies/</code> — 7 built-in: FedAvg, Krum, Median, Trimmed Mean, Reputation, Bulyan, RFA. Extensible via <code>custom/strategies/</code>.</p>
        </div>
        <div class="docs-card">
            <h4>Attacks</h4>
            <p><code>attacks/</code> — 6 types: label flipping, gaussian noise, token replacement (data); weight spiking, gradient scaling, byzantine (model).</p>
        </div>
        <div class="docs-card">
            <h4>Plugin System</h4>
            <p><code>custom/</code> — Drop-in plugins for datasets, models, strategies, losses, optimizers, metrics, schedulers. Auto-discovered.</p>
        </div>
    </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Configuration Reference ───────────────────────────────────
    with st.expander("SimulationConfig Reference"):
        st.code('''SimulationConfig(
    # Model & Data
    model_name="cnn",          # cnn, mlp, resnet18, densenet121, custom:*
    dataset_name="mnist",      # mnist, cifar10, femnist, custom:*
    partition_type="non_iid",  # iid, non_iid
    alpha=0.5,                 # Dirichlet concentration (lower = more heterogeneous)

    # FL Parameters
    num_clients=10,
    num_rounds=20,
    local_epochs=3,
    learning_rate=0.01,
    batch_size=32,
    seed=42,
    fraction_fit=1.0,          # Fraction of clients per round

    # Strategies (one or more)
    strategies=["fedavg", "krum", "trimmed_mean"],

    # Attack
    attack=AttackConfig(
        attack_type="label_flipping",  # or: gaussian_noise, weight_spiking, etc.
        malicious_fraction=0.3,
        attack_params={"snr_db": 10},  # attack-specific params
    ),

    # Training
    optimizer="sgd",           # sgd, adam, adamw
    loss_function="cross_entropy",
    weight_decay=0.0,

    # GPU Acceleration
    use_amp=False,             # Mixed precision (float16)
    compile_model=False,       # torch.compile
    pin_memory=False,          # Pinned DataLoader memory
    max_parallel_clients=1,    # Parallel client training

    # Plugins
    plugin_params={"strategies": {"mu": 0.01}},
)''', language="python")

    with st.expander("Available Strategies"):
        st.markdown("""
| Strategy | Key | Byzantine Robust | Exclusion Tracking |
|----------|-----|:---:|:---:|
| **FedAvg** | `fedavg` | | |
| **Krum** | `krum` | Yes | Yes |
| **Trimmed Mean** | `trimmed_mean` | Yes | |
| **Median** | `median` | Yes | |
| **Reputation** | `reputation` | Yes | Yes |
| **Bulyan** | `bulyan` | Yes | Yes |
| **RFA** | `rfa` | Yes | |
        """)

    with st.expander("Available Attacks"):
        st.markdown("""
| Attack | Key | Type | Key Params |
|--------|-----|------|-----------|
| **Label Flipping** | `label_flipping` | Data | — |
| **Gaussian Noise** | `gaussian_noise` | Data | `snr_db` |
| **Token Replacement** | `token_replacement` | Data | `replace_fraction` |
| **Weight Spiking** | `weight_spiking` | Model | `spike_fraction`, `spike_magnitude` |
| **Gradient Scaling** | `gradient_scaling` | Model | `scale_factor` |
| **Byzantine Perturbation** | `byzantine_perturbation` | Model | `perturbation_scale` |
        """)
