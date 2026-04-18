# FEDSIM Feature Requirements & System Design

**Version:** 0.3.0
**Last Updated:** 2026-03-22
**Status:** Living document — comprehensive redesign for full customizability

---

## Product Identity

FEDSIM is a **visualization-first FL simulation and benchmarking platform** built on Flower. It supports any model, any dataset, any task type, and any aggregation strategy. Researchers bring their ML code; FEDSIM handles simulation orchestration, anomaly detection, and interactive visualization.

**Core principle:** Plug in anything, visualize everything, reproduce any experiment.

---

## Architecture: Three Pillars

```
┌─────────────────────────────────────────────────────────────┐
│                        FEDSIM                                │
├──────────────┬──────────────────┬───────────────────────────┤
│  SIMULATION  │  VISUALIZATION   │  ANOMALY DETECTION        │
│  Engine      │  Dashboard       │  Layer                    │
│              │                  │                           │
│  • Flower    │  • Live charts   │  • Client exclusion       │
│    backend   │  • Per-client    │    tracking (TP/FP/TN/FN) │
│  • Pluggable │    metrics       │  • Removal F1/P/R         │
│    training  │  • Custom metric │  • Trust scores           │
│    loop      │    rendering     │  • Reputation evolution   │
│  • Attack    │  • Anomaly viz   │  • Strategy comparison    │
│    injection │  • Export/Report │  │                        │
└──────────────┴──────────────────┴───────────────────────────┘
        ▲               ▲                    ▲
        │               │                    │
┌───────┴───────────────┴────────────────────┴────────────────┐
│                     PLUGIN SYSTEM                            │
│  Datasets · Models · Strategies · Metrics · Losses ·         │
│  Optimizers · Schedulers · Training Loops                    │
│                                                              │
│  Convention: drop a .py file, define NAME + build/load/      │
│  compute, it appears in the UI with configurable params      │
└──────────────────────────────────────────────────────────────┘
```

---

## Current State (What's Built)

### Implemented
- [x] 7 aggregation strategies (FedAvg, Trimmed Mean, Krum, Median, Reputation, Bulyan, RFA)
- [x] 7 attack types (label flipping, gaussian noise, token replacement, weight spiking, gradient scaling, byzantine perturbation)
- [x] Static and dynamic attack scheduling
- [x] IID and Non-IID (Dirichlet) partitioning
- [x] 4 built-in models (CNN, MLP, ResNet-18, DenseNet-121) + TextCNN plugin
- [x] 10 built-in datasets + AG News plugin
- [x] Plugin auto-discovery for models, datasets, strategies, losses, optimizers, metrics
- [x] Anomaly detection layer (TP/FP/TN/FN, removal F1)
- [x] Streamlit dashboard with redesigned UX (2D charts, tooltips, config strip)
- [x] Optimizer selection (SGD/Adam/AdamW + custom)
- [x] Loss function selection (CrossEntropy/BCE/NLL + custom)
- [x] Trust score computation (cosine + L2 combined)
- [x] Attack parameter validation
- [x] CSV/JSON export + full experiment report
- [x] Comprehensive documentation

### Known Gaps
- [ ] Custom loss/optimizer plugins not shown in UI dropdowns (backend works)
- [ ] Metrics plugins exist as templates but are never called
- [ ] Training loop hardcodes classification (argmax accuracy)
- [ ] Model plugin interface is CV-biased (input_channels, image_size)
- [ ] No plugin hyperparameter configuration from UI
- [ ] No model-dataset compatibility checking
- [ ] No task type awareness (classification vs generation vs NER)
- [ ] Plugin errors silently swallowed

---

## Feature Roadmap

### Phase 1: Plugin UX & Infrastructure Fixes
*Make existing plugins actually work end-to-end*

#### 1.1 Fix Plugin UI Surface (Trivial)
- [ ] Merge custom optimizer plugins into optimizer dropdown: `{**SUPPORTED_OPTIMIZERS, **get_plugin_choices("optimizers")}`
- [ ] Merge custom loss plugins into loss dropdown: same pattern
- [ ] Merge custom metrics plugins into a new "Metrics" section

#### 1.2 Plugin Discoverability
- [ ] Add a "Plugins" section at the bottom of the sidebar
  ```
  ▶ Plugins (3 loaded)
    Datasets: AG News (4-class)
    Models: TextCNN
    Strategies: (none)
    [How to add plugins]
  ```
- [ ] Surface plugin errors as `st.warning()` instead of swallowing silently
- [ ] Show plugin count badge and help tooltip explaining the custom/ directory

#### 1.3 Declarative Plugin Hyperparameters (`PARAMS` dict)
- [ ] Define convention: plugins MAY export a `PARAMS` dict
  ```python
  PARAMS = {
      "embed_dim": {"type": "int", "default": 128, "min": 32, "max": 512,
                    "help": "Embedding dimension"},
      "dropout": {"type": "float", "default": 0.3, "min": 0.0, "max": 0.9,
                  "help": "Dropout rate"},
  }
  ```
- [ ] When a plugin with PARAMS is selected, render a dynamic expander with Streamlit widgets
- [ ] Pass collected param values as `**kwargs` to `build()` / `load()` / `compute()`
- [ ] Supported types: `int` (slider), `float` (slider), `str` (text_input), `bool` (checkbox), `choice` (selectbox)

#### 1.4 Activate Metrics Plugin Pipeline
- [ ] Runner discovers metrics plugins via `discover_plugins("metrics")`
- [ ] After global model evaluation each round, call each metric's `compute(model, testloader, device)`
- [ ] Store results in new `SimulationResult.custom_metrics: dict[str, list[float]]`
- [ ] Add `RoundEvent.custom_metrics: dict[str, float]`
- [ ] Results tab: render custom metrics in the summary table
- [ ] Analysis tab: add "Custom Metrics" to the chart dropdown, plot per metric's `CHART_TYPE`

---

### Phase 2: Task Type System
*Break free from classification-only assumptions*

#### 2.1 Task Type Selector
- [x] Add top-level radio in sidebar: `Task: Image Classification | Text Classification | Language Modeling | Token Classification (NER) | Regression`
- [x] Store in `SimulationConfig.task_type: str`
- [x] Task type filters:
  - Available models (image models hidden when text task selected)
  - Available datasets (text datasets hidden when image task selected)
  - Available attacks (label flipping hidden for generation tasks)
  - Default loss function (CE for classification, causal LM for generation, MSE for regression)
  - Default metrics (accuracy for classification, perplexity for generation, F1 for NER)

#### 2.2 Generalized Plugin Interface
- [x] Replace CV-biased `build(input_channels, num_classes, image_size)` with:
  ```python
  def build(dataset_info: dict, **kwargs) -> nn.Module:
      """
      dataset_info contains:
        - task_type: str
        - num_classes: int
        - input_channels: int (images) or None
        - image_size: int (images) or None
        - input_size: int (flattened input dim)
        - vocab_size: int (text) or None
        - seq_length: int (text) or None
      """
  ```
- [x] Backward compatible: if plugin's `build()` has old signature, wrap it
- [x] Dataset plugins export a `DATASET_INFO` dict instead of separate attributes
- [x] Add `TASK_TYPE` and `MODALITY` attributes to plugins for filtering

#### 2.3 Pluggable Training Loop
- [x] Model plugins MAY export `train_step(model, batch, optimizer, device, **kwargs) -> dict[str, float]`
- [x] Model plugins MAY export `fit(model, dataloader, optimizer, device, local_epochs, **kwargs) -> dict[str, float]`
- [x] If defined, runner uses these instead of hardcoded classification loop
- [x] If not defined, runner falls back to current classification behavior (backward compatible)
- [x] Built-in training loops for standard task types:
  - `classification`: `loss = criterion(model(x), y)`, metrics = accuracy
  - `language_modeling`: `loss = model(input_ids, labels=input_ids).loss`, metrics = perplexity
  - `ner`: token-level cross entropy, metrics = entity F1
  - `regression`: MSE loss, metrics = MAE, R²

#### 2.4 Generalized Evaluation
- [x] Replace `_evaluate_model` hardcoded accuracy with pluggable evaluation
- [x] Use configured loss function (not hardcoded CrossEntropyLoss)
- [x] Return `dict[str, float]` of metrics instead of `(loss, accuracy)` tuple
- [x] Model plugins MAY export `eval_step(model, batch, device, **kwargs) -> dict[str, float]`
- [x] Model plugins MAY export `evaluate(model, dataloader, device, **kwargs) -> dict[str, float]`
- [x] Backward compatible: `round_losses` and `round_accuracies` remain as convenience accessors

---

### Phase 3: Model-Dataset Compatibility & Validation

#### 3.1 Compatibility Metadata
- [ ] Plugins declare `MODALITY = "image" | "text" | "tabular" | "any"`
- [ ] Plugins declare `COMPATIBLE_TASKS = ["image_classification", "text_classification"]`
- [ ] UI filters model dropdown based on selected dataset's modality
- [ ] Mismatched selections show a warning (not a hard block — power users may experiment)

#### 3.2 Config Validation & Preview
- [ ] Before simulation, show a "Config Preview" expandable:
  ```
  Model: TextCNN (3.35M params)
  Dataset: AG News (120K train, 7.6K test, 4 classes)
  Partition: Non-IID (α=0.5) → ~20K samples/client
  Strategies: FedAvg, Krum, Reputation
  Attack: Label Flipping on 2/6 clients
  Est. runtime: ~3 minutes
  ```
- [ ] Attempt to instantiate model + forward pass with dummy data to catch shape errors
- [ ] Surface parameter count and estimated runtime

#### 3.3 Preset Configurations
- [ ] "Quick Demo" — CNN, CIFAR-10, 3 clients, 5 rounds, FedAvg
- [ ] "Byzantine Robustness" — CNN, CIFAR-10, 10 clients, 15 rounds, 4 strategies, Byzantine attack
- [ ] "Text Classification" — TextCNN, AG News, 6 clients, 10 rounds, Label Flipping
- [ ] Presets populate all sidebar fields via `st.session_state`
- [ ] Accessible from a dropdown or button row at the top of the sidebar

---

### Phase 4: Flower-Parity Controls

#### 4.1 Client Sampling
- [ ] `fraction_fit`: fraction of clients per round (0.0-1.0)
- [ ] `fraction_evaluate`: fraction for evaluation
- [ ] UI: two sliders under Advanced Settings

#### 4.2 Minimum Client Thresholds
- [ ] `min_fit_clients`, `min_evaluate_clients`, `min_available_clients`
- [ ] UI: three number inputs under Advanced Settings

#### 4.3 Evaluation Configuration
- [ ] Server-side vs client-side evaluation toggle
- [ ] Evaluation frequency: every N rounds
- [ ] Data split configuration: train/val/test ratios

#### 4.4 LR Scheduling
- [ ] Built-in: StepLR, CosineAnnealingLR, ExponentialLR
- [ ] Plugin interface: `NAME`, `build(optimizer, **kwargs) -> LRScheduler`
- [ ] UI: scheduler dropdown + params (renders from PARAMS dict)

#### 4.5 Client Resources (Simulation)
- [ ] `client_resources`: CPU/GPU allocation per simulated client
- [ ] Affects Flower simulation backend resource allocation

---

### Phase 5: Advanced Visualization

#### 5.1 Custom Metric Charting
- [x] Any metric from custom plugins appears as a chart option in Results tab
- [x] Respect `CHART_TYPE` hint from plugin (line, bar, scalar)
- [x] Metrics keyed by prefix namespace (e.g., `plugin_name/metric_key`)

#### 5.2 Experiment Comparison
- [x] Save experiment results to local JSON storage (~/.fedsim/experiments/)
- [x] Load and compare past experiments side-by-side
- [x] Config diff view: highlight parameters that differ between experiments

#### 5.3 Delta-from-Baseline View
- [x] Toggle in Results table: show metrics relative to user-selected baseline
- [x] "Pin baseline" mechanism: selectbox to designate one strategy as reference

#### 5.4 Enhanced Anomaly Viz
- [x] Threshold lines on client score distributions
- [x] Anomaly event log: scrollable per-round exclusion events with TP/FP status
- [x] Per-client metric sparklines (trust + reputation trajectories per client)

#### 5.5 Report Export
- [x] Interactive HTML report (single-file, shareable Plotly charts)
- [x] JSON full report export from dashboard

---

### Phase 6: Infrastructure

#### 6.1 Packaging
- [ ] `pyproject.toml` with src/ layout
- [ ] `fedsim` CLI entry point + `python -m fedsim`
- [ ] pip installable from private GitHub
- [ ] .gitignore, README.md

#### 6.2 CI/CD
- [ ] pytest + ruff across Python 3.10-3.12
- [ ] Integration tests for each plugin type
- [ ] Coverage reporting

#### 6.3 HuggingFace Dataset Integration
- [ ] Load any HF dataset by name with auto-partitioning
- [ ] Auto-detect task type from HF dataset metadata
- [ ] UI: text input for HF dataset identifier

#### 6.4 Experiment Scheduler
- [ ] Batch run multiple configs from JSON
- [ ] Hyperparameter sweep (grid/random)
- [ ] Optimal config recommendation after batch runs

---

### Phase 7: AI Integration (Future)

- [ ] Natural language experiment configuration
- [ ] Post-experiment analysis and recommendations
- [ ] Provider-agnostic (OpenAI, Anthropic, Ollama)

---

## Plugin Interface Specifications (v2)

All plugins follow the same pattern: drop a `.py` file in `custom/<type>/`, define `NAME` and the required function. Optional exports (`PARAMS`, `MODALITY`, `TASK_TYPE`, `DESCRIPTION`) enable richer UI integration.

### Common Optional Exports (all plugin types)
```python
NAME = "Display Name"                              # REQUIRED
DESCRIPTION = "One-line description for tooltips"  # Optional
MODALITY = "image" | "text" | "tabular" | "any"    # Optional, for filtering
COMPATIBLE_TASKS = ["image_classification", ...]    # Optional, for filtering
PARAMS = {                                          # Optional, for UI rendering
    "param_name": {
        "type": "int" | "float" | "str" | "bool" | "choice",
        "default": value,
        "min": value,        # for int/float
        "max": value,        # for int/float
        "choices": [...],    # for choice type
        "help": "tooltip",
    },
}
```

### Dataset Plugin (`custom/datasets/`)
```python
NAME = "My Dataset"
NUM_CLASSES = 10            # REQUIRED (for now; will move to DATASET_INFO)
INPUT_CHANNELS = 3          # REQUIRED (for now)
IMAGE_SIZE = 32             # REQUIRED (for now)

# NEW (Phase 2): richer metadata
DATASET_INFO = {
    "num_classes": 10,
    "input_channels": 3,       # None for text
    "image_size": 32,          # None for text
    "vocab_size": None,        # set for text datasets
    "seq_length": None,        # set for text datasets
    "task_type": "image_classification",
}

def load(**kwargs):
    """Return (train_dataset, test_dataset). kwargs from PARAMS."""
```

### Model Plugin (`custom/models/`)
```python
NAME = "My Model"

# Required: generalized build function
def build(dataset_info: dict, **kwargs) -> nn.Module:
    """Return nn.Module.
    dataset_info keys: task_type, num_classes, input_channels, image_size,
                       input_size, vocab_size, seq_length.
    kwargs from PARAMS.
    """

# Optional training overrides (tier 1 = fit, tier 2 = train_step):
def fit(model, dataloader, optimizer, device, local_epochs, **kwargs) -> dict[str, float]:
    """Full epoch control. Must return dict with at least 'loss'."""

def train_step(model, batch, optimizer, device, **kwargs) -> dict[str, float]:
    """Per-step control. Must return dict with at least 'loss'."""

# Optional evaluation overrides (tier 1 = evaluate, tier 2 = eval_step):
def evaluate(model, dataloader, device, **kwargs) -> dict[str, float]:
    """Full eval control. Must return dict with at least 'loss'."""

def eval_step(model, batch, device, **kwargs) -> dict[str, float]:
    """Per-step eval. Must return dict with at least 'loss'."""
```

### Strategy Plugin (`custom/strategies/`)
```python
NAME = "My Strategy"

def build(num_clients, num_malicious=0, **kwargs):
    """Return a Flower Strategy instance. kwargs includes Flower common params + PARAMS."""
```

### Metric Plugin (`custom/metrics/`)
```python
NAME = "F1 Score"
CHART_TYPE = "line"  # "line", "bar", or "scalar"

def compute(model, dataloader, device, **kwargs):
    """Return dict[str, float] of metric values."""
```

### Loss Plugin (`custom/losses/`)
```python
NAME = "Focal Loss"

def build(**kwargs):
    """Return nn.Module loss function."""
```

### Optimizer Plugin (`custom/optimizers/`)
```python
NAME = "AdamW"

def build(model_params, lr=0.001, **kwargs):
    """Return torch.optim.Optimizer."""
```

### LR Scheduler Plugin (`custom/schedulers/`) — Phase 4
```python
NAME = "Cosine Annealing"

def build(optimizer, **kwargs):
    """Return torch.optim.lr_scheduler._LRScheduler."""
```

---

## Implementation Order

Each phase is independent. Within each phase, items can be built in any order.

```
Phase 1: Plugin UX fixes          ← NEXT (unblocks all plugin users)
Phase 2: Task type system          ← NEXT (unblocks NLP/generation)
Phase 3: Compatibility & presets   ← After Phase 2
Phase 4: Flower-parity controls    ← Independent, any time
Phase 5: Advanced visualization    ← After Phase 2 (needs custom metrics)
Phase 6: Infrastructure            ← Last (packaging)
Phase 7: AI integration            ← Future
```

---

## Design Principles

1. **Plugin-first:** Every extension point follows the same pattern. Drop a .py, define NAME + function, it appears in the UI.

2. **Declarative configuration:** Plugins declare what they need (PARAMS, MODALITY, COMPATIBLE_TASKS). The framework renders the UI and enforces constraints.

3. **Backward compatible:** New plugin attributes are always optional. Old plugins work without changes. New SimulationConfig fields always have defaults.

4. **Task-aware:** The task type cascades through the entire system — filtering models, datasets, attacks, metrics, loss functions, and evaluation logic.

5. **Progressive disclosure:** Basic use requires 3-4 selections. Advanced configuration is behind expanders. Plugin hyperparameters only appear when a plugin is selected.

6. **Fail visibly:** Plugin errors surface as warnings in the UI, not silent failures. Config validation catches mismatches before simulation starts.

7. **Flower-native:** All FL orchestration goes through Flower's Strategy interface. FEDSIM adds visualization, anomaly detection, and benchmarking — it never replaces Flower.

8. **Reproducible:** Every experiment exports as JSON with full config, seed, plugin versions, and results. Import a JSON to reproduce any experiment.
