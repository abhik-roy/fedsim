# FEDSIM - A Flexible and Lightweight Federated Learning Simulation  Framework

FEDSIM is a visualization-first platform for simulating, benchmarking, and analyzing federated learning systems under adversarial conditions. It provides a Streamlit dashboard for interactive exploration and a Python scripting API for reproducible experiments. Built for FL researchers, FEDSIM enables systematic comparison of aggregation strategies, attack models, and defense mechanisms, it is meant to be customizable and lightweight.

---

## Features

### Simulation Engine
- **7 aggregation strategies** — FedAvg, Krum, Median, Trimmed Mean, Bulyan, RFA, and Reputation (trust-based client selection with asymmetric temporal updates)
- **6 attack models** — Label flipping, Gaussian noise, token replacement (data poisoning); weight spiking, gradient scaling, Byzantine perturbation (model poisoning)
- **Non-IID data** — Dirichlet-based heterogeneous partitioning with tunable concentration (alpha)
- **Anomaly detection** — TP/FP/TN/FN tracking for client exclusion decisions, removal F1/precision/recall per round
- **Parallel client training** — ThreadPoolExecutor with CUDA stream support for GPU acceleration
- **Mixed precision (AMP)** — Float16 training on tensor core GPUs for ~1.5x speedup

### Dashboard
- **Live training visualization** — Loss and accuracy curves updated in real-time
- **Client activity grid** — Per-client, per-round status heatmap (benign/attacked/idle)
- **Anomaly detection tab** — Removal F1, exclusion timeline, confusion matrices
- **Analysis tab** — 3D accuracy surfaces, trust/reputation landscapes, attack impact matrices, PCA
- **Experiment loading** — Upload scripted experiment configs and run them with full live visualization

### Plugin System
- **Drop-in extensibility** — Add custom datasets, models, strategies, losses, optimizers, metrics, and schedulers
- **Auto-discovery** — Place a `.py` file in `custom/{type}/` and it appears in the UI
- **Strategy training hooks** — Plugins can override client-side training (e.g., FedProx proximal term)
- **Declarative PARAMS** — Plugin hyperparameters rendered as UI controls automatically

### Scripting API
- **`Experiment`** — Named runs with progress reporting and JSON checkpointing
- **`Report`** — Composable PDF builder (convergence plots, accuracy tables, heatmaps, text)
- **Dashboard integration** — Export configs from scripts, load and execute in the dashboard

---

## Quick Start

### Requirements

- Python 3.10+
- GPU recommended (NVIDIA with CUDA support for acceleration)

### Installation

```bash
git clone https://github.com/abhik-roy/fedsim.git
cd fedsim
pip install -r requirements.txt
```

### Launch the Dashboard

```bash
streamlit run app.py
```

1. Select a model (CNN, MLP, ResNet-18, DenseNet-121) and dataset (MNIST, CIFAR-10, Fashion-MNIST, etc.)
2. Choose aggregation strategies (checkboxes)
3. Configure an attack (optional)
4. Click **Run Simulation**
5. Watch live training curves, inspect anomaly detection, and export results

### Run a Scripted Experiment

```python
from api import Experiment, Report
from simulation.runner import SimulationConfig, AttackConfig

exp = Experiment("Byzantine Robustness Benchmark")

exp.add_run("FedAvg (clean)", SimulationConfig(
    model_name="cnn", dataset_name="cifar10",
    num_clients=10, num_rounds=20, local_epochs=3,
    learning_rate=0.01, strategies=["fedavg"],
    attack=AttackConfig(),
))

exp.add_run("Krum (under attack)", SimulationConfig(
    model_name="cnn", dataset_name="cifar10",
    num_clients=10, num_rounds=20, local_epochs=3,
    learning_rate=0.01, strategies=["krum"],
    attack=AttackConfig(attack_type="label_flipping", malicious_fraction=0.3),
))

results = exp.run(checkpoint_path="results/benchmark.json")

# Generate PDF report
report = Report("Byzantine Robustness")
report.add_convergence_plot(results, results.names, title="Strategy Comparison")
report.add_accuracy_table(results)
report.save_pdf("benchmark_report.pdf")
```

---

## Architecture

```
fedsim/
├── app.py                    # Streamlit dashboard
├── fl_core.py                # Core FL types (Strategy, FedAvg, FitRes)
├── plugins.py                # Plugin auto-discovery
├── simulation/
│   └── runner.py             # Simulation engine (1400+ lines)
├── strategies/               # 5 built-in Byzantine-robust strategies
│   ├── krum.py               #   Multi-Krum (Blanchard et al.)
│   ├── trimmed_mean.py       #   Coordinate-wise trimmed mean
│   ├── median.py             #   Coordinate-wise median
│   ├── bulyan.py             #   Two-stage: Krum + trimmed mean
│   └── rfa.py                #   Geometric median (Weiszfeld)
├── attacks/
│   ├── data_poisoning.py     # Label flipping, Gaussian noise, token replacement
│   └── model_poisoning.py    # Weight spiking, gradient scaling, Byzantine
├── anomaly/
│   └── metrics.py            # TP/FP/TN/FN tracking for client exclusion
├── models/                   # CNN, MLP, ResNet-18, DenseNet-121
├── data/                     # Dataset loading + Dirichlet partitioning
├── configs/
│   └── defaults.py           # All default constants and mappings
├── visualization/            # Plotly charts for dashboard
├── api/
│   ├── experiment.py         # Experiment runner with checkpointing
│   └── report.py             # PDF report builder
├── custom/                   # Plugin directory
│   ├── strategies/           # FedProx, Reputation (examples)
│   ├── datasets/             # AG News, California Housing, WikiText-2
│   ├── models/               # TextCNN, RegressionMLP, Small LM
│   ├── losses/               # Template
│   ├── optimizers/           # Template
│   ├── metrics/              # Template
│   └── schedulers/           # Template
└── tests/                    # 30+ test files, 137+ tests
```

---

## Configuration Reference

### SimulationConfig

```python
SimulationConfig(
    # Model & Data
    model_name="cnn",             # cnn, mlp, resnet18, densenet121, custom:*
    dataset_name="cifar10",       # mnist, cifar10, cifar100, fashion_mnist,
                                  # svhn, femnist, pathmnist, dermamnist,
                                  # bloodmnist, organamnist, custom:*
    partition_type="non_iid",     # iid, non_iid
    alpha=0.5,                    # Dirichlet concentration (lower = more heterogeneous)

    # FL Parameters
    num_clients=10,
    num_rounds=20,
    local_epochs=3,
    learning_rate=0.01,
    batch_size=32,
    seed=42,
    fraction_fit=1.0,             # Fraction of clients sampled per round

    # Strategies
    strategies=["fedavg", "krum", "trimmed_mean"],

    # Attack
    attack=AttackConfig(
        attack_type="label_flipping",
        malicious_fraction=0.3,
        attack_params={"snr_db": 10},
    ),

    # Training
    optimizer="sgd",              # sgd, adam, adamw
    loss_function="cross_entropy",

    # GPU Acceleration
    use_amp=False,                # Mixed precision (float16)
    compile_model=False,          # torch.compile
    pin_memory=False,             # Pinned DataLoader memory
    max_parallel_clients=1,       # Parallel client training (1 = sequential)

    # Plugin parameters
    plugin_params={"strategies": {"mu": 0.01}},
)
```

### Strategies

| Strategy | Key | Type | Exclusion Tracking |
|----------|-----|------|:--:|
| FedAvg | `fedavg` | Baseline | |
| Krum | `krum` | Byzantine-robust | Yes |
| Trimmed Mean | `trimmed_mean` | Byzantine-robust | |
| Median | `median` | Byzantine-robust | |
| Bulyan | `bulyan` | Byzantine-robust | Yes |
| RFA | `rfa` | Byzantine-robust | |
| Reputation | `reputation` | Trust-based | Yes |
| FedProx | `custom:FedProx` | Proximal regularization | |

### Attacks

| Attack | Key | Type | Key Parameters |
|--------|-----|------|---------------|
| Label Flipping | `label_flipping` | Data | — |
| Gaussian Noise | `gaussian_noise` | Data | `snr_db` |
| Token Replacement | `token_replacement` | Data | `replace_fraction` |
| Weight Spiking | `weight_spiking` | Model | `spike_fraction`, `magnitude` |
| Gradient Scaling | `gradient_scaling` | Model | `scale_factor` |
| Byzantine Perturbation | `byzantine_perturbation` | Model | `perturbation_scale` |

---

## Writing Plugins

### Custom Strategy

```python
# Save as custom/strategies/my_strategy.py

from fl_core import FedAvg, FitRes, NDArrays

NAME = "My Strategy"
DESCRIPTION = "A custom aggregation strategy"
PARAMS = {
    "beta": {"type": "float", "default": 0.1, "min": 0.0, "max": 1.0,
             "label": "Trimming Parameter"},
}

def build(initial_parameters, num_clients, num_malicious=0, **kwargs):
    beta = kwargs.pop("beta", PARAMS["beta"]["default"])
    return MyStrategy(beta=beta, initial_parameters=initial_parameters, **kwargs)

# Optional: override client training (like FedProx)
def train_step(model, batch, optimizer, device, **kwargs):
    """Called per batch instead of the default training loop."""
    images, labels = batch
    images, labels = images.to(device), labels.to(device)
    optimizer.zero_grad()
    loss = torch.nn.functional.cross_entropy(model(images), labels)
    loss.backward()
    optimizer.step()
    return {"loss": loss.item()}
```

### Custom Dataset

```python
# Save as custom/datasets/my_dataset.py

NAME = "My Dataset"
NUM_CLASSES = 10
INPUT_CHANNELS = 3
IMAGE_SIZE = 32
TASK_TYPE = "image_classification"

def load(**kwargs):
    train_dataset = ...  # torch.utils.data.Dataset
    test_dataset = ...
    return train_dataset, test_dataset
```

### Custom Model

```python
# Save as custom/models/my_model.py

import torch.nn as nn

NAME = "My Model"
COMPATIBLE_TASKS = ["image_classification"]

def build(dataset_info, **kwargs):
    num_classes = dataset_info["num_classes"]
    return MyNet(num_classes)
```

---

## Scripting API

### Experiment

```python
from api import Experiment

exp = Experiment("My Study")
exp.add_run("run_name", SimulationConfig(...))
results = exp.run(checkpoint_path="results/my_study.json")

# Access results
results.final_accuracy("run_name")           # Last accuracy (first strategy)
results.final_accuracy("run_name", strategy_idx=1)  # Second strategy
results.names                                 # All run names
results.items()                               # Iterate (name, results) pairs

# Save/load
exp.save("results/my_study.json")
loaded = Experiment.load("results/my_study.json")

# Export configs for dashboard execution
exp.export_configs("results/my_configs.json")
# Then upload to dashboard via Load Experiment in the sidebar
```

### Report

```python
from api import Report

report = Report("My Report Title")
report.add_text("Introduction", "Description of the experiment...")
report.add_convergence_plot(results, ["run1", "run2"], title="Convergence")
report.add_accuracy_table(results)
report.add_heatmap(data, row_labels, col_labels, title="Heatmap")
report.save_pdf("my_report.pdf")
```

---

## GPU Acceleration

When a CUDA GPU is available, three acceleration options appear in the dashboard sidebar:

| Option | Config Field | Speedup | How It Works |
|--------|-------------|---------|-------------|
| Mixed Precision | `use_amp=True` | ~1.5x | Float16 forward/backward via tensor cores |
| torch.compile | `compile_model=True` | ~1.2x | Graph compilation (graceful fallback if unavailable) |
| Pin Memory | `pin_memory=True` | ~1.05x | Pinned host memory for async CPU-GPU transfers |

---

## Built-in Datasets

| Dataset | Key | Classes | Size | Type |
|---------|-----|---------|------|------|
| MNIST | `mnist` | 10 | 60K/10K | Grayscale 28x28 |
| CIFAR-10 | `cifar10` | 10 | 50K/10K | RGB 32x32 |
| CIFAR-100 | `cifar100` | 100 | 50K/10K | RGB 32x32 |
| Fashion-MNIST | `fashion_mnist` | 10 | 60K/10K | Grayscale 28x28 |
| SVHN | `svhn` | 10 | 73K/26K | RGB 32x32 |
| FEMNIST | `femnist` | 62 | 731K/82K | Grayscale 28x28 |
| PathMNIST | `pathmnist` | 9 | 89K/7K | RGB 28x28 |
| DermaMNIST | `dermamnist` | 7 | 7K/2K | RGB 28x28 |
| BloodMNIST | `bloodmnist` | 8 | 11K/3K | RGB 28x28 |
| OrganAMNIST | `organamnist` | 11 | 34K/8K | Grayscale 28x28 |

**Plugin datasets:** AG News (text, 4-class), California Housing (regression), WikiText-2 (language modeling)

---

