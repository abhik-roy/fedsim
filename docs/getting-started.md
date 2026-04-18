# FEDSIM: Getting Started Guide

Comprehensive documentation for FEDSIM, a Federated Learning simulation and visualization framework.

---

## Table of Contents

1. [What is FEDSIM?](#1-what-is-fedsim)
2. [Installation](#2-installation)
3. [Quick Start](#3-quick-start)
4. [Understanding the Tabs](#4-understanding-the-tabs)
5. [Custom Plugins](#5-custom-plugins)
6. [Configuration Reference](#6-configuration-reference)
7. [Glossary](#7-glossary)
8. [Reproducing Published Results](#8-reproducing-published-results)

---

## 1. What is FEDSIM?

FEDSIM (Federated Learning Simulation & Visualization Framework) is a visualization-first FL simulation platform with a built-in FL core layer. It is designed for ML researchers and graduate students who need to:

- **Benchmark aggregation strategies** side by side under identical conditions (same data partitions, same malicious clients, same random seed).
- **Simulate adversarial attacks** including data poisoning (label flipping, Gaussian noise, token replacement) and model poisoning (weight spiking, gradient scaling, Byzantine perturbation).
- **Detect anomalous clients** by tracking which clients get excluded by robust strategies and measuring removal precision, recall, and F1 against ground-truth labels.
- **Interpret model behavior** through trust/reputation heatmaps, client PCA projections, attack impact matrices, and 3D accuracy surfaces.
- **Extend the framework** with custom models, datasets, and aggregation strategies via a lightweight plugin system.

FEDSIM runs entirely locally. The dashboard is a Streamlit application that orchestrates simulations through a PyTorch training loop and fl_core's aggregation layer, rendering live Plotly visualizations as each round completes.

### Key Design Decisions

- **Deterministic by default.** Every simulation uses a configurable random seed (`42` by default) that controls data partitioning, malicious client selection, and attack noise. This means results are reproducible across runs.
- **Strategy-parallel evaluation.** When you select multiple strategies, FEDSIM trains each one sequentially but against the same data partitions and malicious client assignments, enabling fair comparison.
- **Live streaming.** Loss, accuracy, and client status grids update after every FL round via callback hooks, so you can monitor training in real time.

---

## 2. Installation

### Requirements

- Python 3.10 or later
- CUDA-capable GPU (optional but recommended for larger models like ResNet-18 and DenseNet-121)

### From Source (Recommended for Development)

```bash
git clone git@github.com:<user>/fedsim.git
cd fedsim
pip install -e ".[dev]"
```

### From Git (Direct Install)

```bash
pip install git+ssh://git@github.com/<user>/fedsim.git
```

### Manual Dependency Installation

If you prefer to install dependencies manually:

```bash
pip install -r requirements.txt
```

The core dependencies are:

| Package | Minimum Version | Purpose |
|---|---|---|
| `torch` | 2.2.0 | Model training and inference |
| `torchvision` | 0.17.0 | Built-in datasets (CIFAR, MNIST, SVHN, etc.) |
| `fl_core` | (bundled) | FEDSIM aggregation strategies and FL primitives |
| `streamlit` | 1.30.0 | Dashboard UI |
| `plotly` | 5.18.0 | Interactive visualizations |
| `matplotlib` | 3.8.0 | PDF report generation |
| `numpy` | 1.26.0 | Numerical operations |
| `pandas` | 2.1.0 | Data export (CSV, DataFrames) |
| `medmnist` | 3.0.0 | Medical imaging datasets |
| `scikit-learn` | 1.3.0 | PCA for client embedding visualization |

### Verifying Your Installation

```bash
python -c "from simulation.runner import SimulationConfig; print('FEDSIM ready')"
```

---

## 3. Quick Start

### Launching the Dashboard

```bash
# From the project root:
streamlit run app.py

# Or, if FEDSIM is installed as a package:
fedsim dashboard
```

The dashboard opens in your browser at `http://localhost:8501`.

### Walkthrough: Your First Simulation

Follow these steps in the sidebar to run a basic experiment comparing FedAvg against Krum under a label flipping attack.

#### Step 1: Model & Data

1. Under **Model & Data**, select **Simple CNN** from the model dropdown.
2. Select **CIFAR-10** from the dataset dropdown.
3. Set **Partition** to **Non-IID (Dirichlet)** and leave **Dirichlet alpha** at `0.5` (moderate heterogeneity).

#### Step 2: FL Parameters

1. Set **Clients** to `10`.
2. Set **Rounds** to `20`.
3. Set **Epochs** to `3` (local training epochs per client per round).
4. Set **LR** to `0.01`.

#### Step 3: Strategies

1. In the **Strategies** multiselect, choose **FedAvg** and **Krum**.
   - You can select any combination of strategies. They will all run sequentially against the same data partitions.

#### Step 4: Attack Configuration

1. Under **Attack**, select **Label Flipping**.
2. Set **Malicious %** to `0.30` (30% of clients will be adversarial).
3. Leave **Schedule** on **Static (all rounds)** so the attack is active every round.

#### Step 5: Run

1. Click **Run Simulation**.
2. Watch the **Training** tab for live loss/accuracy curves and the client activity grid.

#### What to Look At

- **Training tab**: Are the loss curves converging? Does Krum maintain higher accuracy than FedAvg under the attack?
- **Anomaly Detection tab**: Check the exclusion timeline. Krum should be excluding malicious clients (red cells) while keeping benign ones (green cells). Look at the Removal F1 metric.
- **Analysis tab**: The trust/reputation landscape shows per-client scores over rounds. The client PCA plot shows how malicious client updates cluster differently in parameter space.
- **Results tab**: Compare final accuracy, best accuracy, and removal F1 across strategies. Export results as CSV or JSON.

---

## 4. Understanding the Tabs

The dashboard has four tabs: **Training**, **Anomaly Detection**, **Analysis**, and **Results**.

### 4.1 Training Tab

This tab displays live metrics during simulation. It updates after every FL round.

#### Loss and Accuracy Curves

Two side-by-side line charts show global model loss (left) and accuracy (right) over rounds. Each strategy gets its own colored line:

| Strategy | Color |
|---|---|
| FedAvg | Red (`#e74c3c`) |
| Trimmed Mean | Green (`#2ecc71`) |
| Krum | Blue (`#3498db`) |
| Median | Purple (`#9b59b6`) |
| Reputation | Orange (`#f39c12`) |

The x-axis spans from round 0 (initial evaluation before any training) through the configured number of rounds.

#### Client Activity Grid

When an attack is active, a heatmap grid appears below the charts. Each row is a client, each column is a round. Cell colors indicate status:

| Color | Status | Meaning |
|---|---|---|
| Green (`#2ecc71`) | `benign` | Client is honest and participated normally |
| Red (`#e74c3c`) | `attacked` | Client is malicious and the attack was applied this round |
| Gray (`#95a5a6`) | `malicious_idle` | Client is malicious but the attack was not active this round (dynamic schedule) |

#### Status Bar

A monospace status bar at the top shows the current strategy, round number, loss, accuracy, and elapsed time.

### 4.2 Anomaly Detection Tab

This tab evaluates how well a strategy identifies and excludes malicious clients. It is only meaningful for strategies that perform client exclusion: **Krum**, **Reputation**, and **Bulyan**. Strategies like FedAvg, Trimmed Mean, Median, and RFA include all clients in aggregation and will show "No anomaly detection data."

#### Strategy Selector

A dropdown at the top lets you pick which strategy's anomaly data to inspect.

#### Removal F1 Over Rounds

A line chart showing per-round removal F1 score. F1 combines precision and recall into a single metric:

- **High F1** means the strategy is correctly excluding malicious clients without excluding benign ones.
- **Low F1** can mean the strategy is either missing malicious clients (low recall) or falsely excluding benign ones (low precision).

#### Exclusion Timeline

A heatmap similar to the client grid, but color-coded by exclusion correctness:

| Outcome | Meaning |
|---|---|
| **TP** (True Positive) | Malicious client was correctly excluded |
| **FP** (False Positive) | Benign client was incorrectly excluded |
| **TN** (True Negative) | Benign client was correctly included |
| **FN** (False Negative) | Malicious client was incorrectly included (missed) |

#### Confusion Summary

A bar chart showing cumulative TP, FP, TN, and FN counts across all rounds.

#### Client Score Distribution

A histogram showing the distribution of strategy-specific scores (e.g., Krum distances, reputation values) for the final round. Malicious and benign clients are shown in different colors, letting you see how well the scoring function separates them.

#### Summary Metrics

Four metric cards at the bottom show cumulative Removal F1, Precision, Recall, and Total Rounds.

### 4.3 Analysis Tab

This tab provides deeper interpretability visualizations. It requires a completed simulation.

#### Accuracy Surface (3D)

A 3D surface plot showing accuracy over rounds across strategies. Useful for comparing convergence speed and final performance.

#### Trust / Reputation Landscape

A heatmap or 3D surface showing per-client trust or reputation scores over rounds. You can toggle between **Trust** and **Reputation** metrics and select a specific strategy.

- **Trust scores** are computed for all strategies. They combine cosine similarity (direction alignment) and L2 distance (magnitude consistency) between each client's update and the coordinate-wise median. Score range: `[0, 1]` where `1` = most trustworthy.
- **Reputation scores** are only available for the **Reputation** strategy. They are stateful scores that persist across rounds, growing linearly for trustworthy clients and decaying exponentially for suspicious ones.

Malicious clients should appear as darker bands (lower scores) over time.

#### Attack Impact Matrix

Available after clicking **Run Full Benchmark**. This runs every attack type against every selected strategy and produces a matrix showing final accuracy for each combination. Useful for systematic robustness evaluation.

#### Client PCA

A 2D scatter plot of client model parameters (from the final round) projected via PCA. Each dot is a client, colored by malicious/benign status. Point size may reflect reputation score. Malicious clients often cluster separately from benign ones, especially under model poisoning attacks.

### 4.4 Results Tab

#### Summary Table

A table with one row per strategy and the following columns:

| Column | Description |
|---|---|
| **Strategy** | Display name of the aggregation strategy |
| **Final Loss** | Cross-entropy loss of the global model after the last round |
| **Final Acc** | Test accuracy of the global model after the last round |
| **Best Acc** | Highest test accuracy achieved during any round |
| **Removal F1** | Cumulative F1 for client exclusion (only meaningful for Krum, Reputation, Bulyan) |
| **Time** | Wall-clock time for the strategy's simulation |

#### Export Options

- **Export CSV**: Per-round loss and accuracy for all strategies.
- **Export Config**: The full `SimulationConfig` as JSON, for reproducibility.
- **Download Full Report (JSON)**: Complete experiment report including config, per-round metrics, anomaly history, and anomaly summary.

---

## 5. Custom Plugins

FEDSIM supports custom datasets, models, and strategies through a plugin system. Place a `.py` file in the appropriate `custom/` subdirectory and it will be auto-discovered and appear in the dashboard dropdowns.

Plugins are discovered by `plugins.py`, which scans `custom/{datasets,models,strategies}/` for `.py` files that do not start with `_`. Each plugin must define a `NAME` attribute (display name) and a specific entry-point function.

### 5.1 Custom Datasets

**Template location:** `custom/datasets/_template.py`

**Interface:**

```python
# custom/datasets/my_dataset.py
NAME = "My Dataset"         # Display name in the dropdown
NUM_CLASSES = 10            # Number of output classes
INPUT_CHANNELS = 3          # 1 for grayscale, 3 for RGB
IMAGE_SIZE = 32             # Height/width of input images (square)

# Optional attributes:
# TASK_TYPE = "image_classification"  # or "text_classification", "regression", etc.
# SEQ_LENGTH = 128                    # For text datasets (defaults to IMAGE_SIZE)
# VOCAB_SIZE = 25000                  # For text datasets

def load():
    """Return (train_dataset, test_dataset).

    Both must be torch.utils.data.Dataset objects that yield (image, label) tuples.
    Images should be tensors of shape (INPUT_CHANNELS, IMAGE_SIZE, IMAGE_SIZE).
    Labels should be integer class indices.
    """
    from torchvision import datasets, transforms
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    train = datasets.ImageFolder("path/to/train", transform=transform)
    test = datasets.ImageFolder("path/to/test", transform=transform)
    return train, test
```

**How it integrates:** The runner calls `load()` to get train/test datasets, then partitions the training set across clients using the configured partitioning scheme (IID or Dirichlet Non-IID). `NUM_CLASSES` is used for label flipping attack logic and model output layer sizing. The `TASK_TYPE` attribute (defaulting to `"image_classification"`) tells the framework which training loop and evaluation logic to use. All dataset attributes are assembled into a `dataset_info` dict that is passed to model plugins.

### 5.2 Custom Models

**Template location:** `custom/models/_template.py`

**Interface:**

```python
# custom/models/my_model.py
NAME = "My Custom Model"    # Display name in the dropdown

def build(dataset_info, **kwargs):
    """Return a torch.nn.Module instance.

    Args:
        dataset_info: Dict containing dataset metadata:
            - task_type (str): e.g. "image_classification", "text_classification"
            - num_classes (int): number of output classes
            - input_channels (int): e.g. 3 for RGB, 1 for grayscale
            - image_size (int): spatial dimension (square images)
            - input_size (int): flattened input dimension
            - vocab_size (int | None): vocabulary size (text datasets)
            - seq_length (int | None): sequence length (text datasets)
        **kwargs: Values from PARAMS dict, populated by the UI.

    Returns:
        A torch.nn.Module ready for training.
    """
    import torch.nn as nn
    num_classes = dataset_info["num_classes"]
    input_channels = dataset_info["input_channels"]
    image_size = dataset_info["image_size"]
    import torchvision.models as models
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(input_channels, 64, 3, 1, 1, bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
```

**Three-tier override hierarchy:** Model plugins can optionally export training and evaluation functions to customize the loop. Tiers are checked in order; the first match wins.

| Tier | Training | Evaluation | Description |
|---|---|---|---|
| **1 (full control)** | `fit(model, dataloader, optimizer, device, local_epochs, **kwargs) -> dict[str, float]` | `evaluate(model, dataloader, device, **kwargs) -> dict[str, float]` | Plugin manages epoch/batch loops entirely |
| **2 (per-step)** | `train_step(model, batch, optimizer, device, **kwargs) -> dict[str, float]` | `eval_step(model, batch, device, **kwargs) -> dict[str, float]` | Plugin handles one batch; runner manages loops |
| **3 (default)** | *(none needed)* | *(none needed)* | Standard classification loop (cross-entropy + argmax accuracy) |

All return dicts must include at least `{"loss": <float>}`.

**Example with train_step override:**

```python
# custom/models/my_text_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

NAME = "My Text Model"
PARAMS = {
    "embed_dim": {"type": "int", "default": 128, "min": 32, "max": 512, "label": "Embedding Dim"},
}

def build(dataset_info, **kwargs):
    embed_dim = kwargs.get("embed_dim", 128)
    vocab_size = dataset_info["vocab_size"]
    num_classes = dataset_info["num_classes"]
    seq_length = dataset_info["seq_length"]
    # ... build and return your model ...

def train_step(model, batch, optimizer, device, **kwargs):
    inputs, labels = batch
    inputs, labels = inputs.to(device), labels.to(device)
    if inputs.dim() == 3:
        inputs = inputs.squeeze(1)
    optimizer.zero_grad()
    output = model(inputs)
    loss = F.cross_entropy(output, labels)
    loss.backward()
    optimizer.step()
    acc = (output.argmax(1) == labels).float().mean().item()
    return {"loss": loss.item(), "accuracy": acc}
```

**Requirements:** The model must be a standard `torch.nn.Module`. Its `state_dict()` is used for parameter extraction, aggregation, and loading. If no training/eval overrides are provided, the model should work with SGD optimization and cross-entropy loss.

### 5.3 Custom Strategies

**Template location:** `custom/strategies/_template.py`

**Interface:**

```python
# custom/strategies/my_strategy.py
NAME = "Weighted Median"    # Display name in the dropdown

def build(initial_parameters, num_clients, num_malicious=0, **kwargs):
    """Return a Flower Strategy instance.

    Args:
        initial_parameters: Initial model weights as list[np.ndarray].
        num_clients: Total number of clients in the simulation.
        num_malicious: Number of malicious clients (if known).
        **kwargs: Common parameters, including:
            - fraction_fit (float): Fraction of clients for training (always 1.0).
            - fraction_evaluate (float): Fraction of clients for evaluation (always 1.0).
            - min_fit_clients (int): Minimum clients for training (equals num_clients).
            - min_evaluate_clients (int): Minimum clients for evaluation.
            - min_available_clients (int): Minimum available clients.
            - initial_parameters: Same as the positional argument.

    Returns:
        A fl_core.Strategy instance. Typically inherits from fl_core.FedAvg.
    """
    from fl_core import FedAvg
    return FedAvg(**kwargs)
```

**Exclusion metadata:** If your strategy performs client exclusion, return metadata in the `aggregate_fit` metrics dict so the anomaly detection layer can track it:

```python
import json

metrics = {
    "included_clients": json.dumps(sorted(included_cids)),   # list of int
    "excluded_clients": json.dumps(sorted(excluded_cids)),   # list of int
    "client_scores": json.dumps({str(k): float(v) for k, v in scores.items()}),
}
return ndarrays_to_parameters(aggregated), metrics
```

### 5.4 Custom Metrics (Coming Soon)

A plugin interface for custom evaluation metrics beyond loss and accuracy is planned. This will allow tracking domain-specific metrics (e.g., AUC-ROC for medical imaging, perplexity for language models) through the same callback system.

### 5.5 Custom Loss Functions (Coming Soon)

A plugin interface for custom loss functions beyond cross-entropy is planned. This will enable experiments with focal loss, contrastive loss, or other domain-specific objectives.

---

## 6. Configuration Reference

### 6.1 SimulationConfig

The main configuration dataclass passed to `run_simulation()`. All fields with their types, defaults, and descriptions:

| Field | Type | Default | Description |
|---|---|---|---|
| `model_name` | `str` | `"resnet18"` | Model architecture identifier. Built-in options: `"cnn"`, `"mlp"`, `"resnet18"`, `"densenet121"`. Custom: `"custom:<PluginName>"`. |
| `dataset_name` | `str` | `"cifar10"` | Dataset identifier. Built-in options: `"cifar10"`, `"cifar100"`, `"mnist"`, `"fashion_mnist"`, `"svhn"`, `"femnist"`, `"medmnist_pathmnist"`, `"medmnist_dermamnist"`, `"medmnist_bloodmnist"`, `"medmnist_organamnist"`. Custom: `"custom:<PluginName>"`. |
| `num_clients` | `int` | `10` | Number of FL clients. Must be >= 2. Dashboard range: 2--20. |
| `num_rounds` | `int` | `20` | Number of FL communication rounds. Must be >= 1. Dashboard range: 1--50. |
| `local_epochs` | `int` | `3` | Local training epochs per client per round. Must be >= 1. Dashboard range: 1--10. |
| `learning_rate` | `float` | `0.01` | SGD learning rate. Must be > 0. Dashboard range: 0.0001--1.0. |
| `partition_type` | `str` | `"non_iid"` | Data partition scheme. `"iid"` for uniform random split, `"non_iid"` for Dirichlet-based heterogeneous split. |
| `alpha` | `float` | `0.5` | Dirichlet concentration parameter for Non-IID partitioning. Lower values = more heterogeneous. Only used when `partition_type == "non_iid"`. Dashboard range: 0.01--10.0. |
| `strategies` | `list[str]` | `["fedavg"]` | List of strategy identifiers to evaluate. Built-in: `"fedavg"`, `"trimmed_mean"`, `"krum"`, `"median"`, `"reputation"`, `"bulyan"`, `"rfa"`. Custom: `"custom:<PluginName>"`. |
| `batch_size` | `int` | `32` | Mini-batch size for local training. |
| `seed` | `int` | `42` | Random seed for reproducibility. Controls data partitioning, malicious client selection, and attack noise generation. |
| `attack` | `AttackConfig` | `AttackConfig()` | Attack configuration (see below). |
| `reputation_truth_threshold` | `float` | `0.7` | Truth threshold for the Reputation strategy. Clients with truth >= this value get linear reputation growth; below it, exponential decay. |
| `reputation_selection_fraction` | `float` | `0.6` | Fraction of clients selected by the Reputation strategy for aggregation each round (top-k by reputation). |
| `reputation_initial_reputation` | `float` | `0.5` | Initial reputation score assigned to all clients at the start of training. |

### 6.2 AttackConfig

Configuration for adversarial attacks. Nested inside `SimulationConfig.attack`.

| Field | Type | Default | Description |
|---|---|---|---|
| `attack_type` | `str` | `"none"` | Attack type identifier. Options: `"none"`, `"label_flipping"`, `"gaussian_noise"`, `"token_replacement"`, `"weight_spiking"`, `"gradient_scaling"`, `"byzantine_perturbation"`. |
| `malicious_fraction` | `float` | `0.2` | Fraction of clients that are malicious. Must be in (0, 1) when an attack is active. Dashboard range: 0.05--0.9. |
| `attack_params` | `dict` | `{}` | Attack-specific parameters (see table below). |
| `schedule_type` | `str` | `"static"` | `"static"`: attack is active every round. `"dynamic"`: attack is active only during specified rounds. |
| `attack_rounds` | `list[int] \| None` | `None` | List of round numbers when the attack is active. Only used when `schedule_type == "dynamic"`. |

#### Attack-Specific Parameters

| Attack | Category | Parameter | Default | Description |
|---|---|---|---|---|
| `label_flipping` | Data | *(none)* | | Applies a bijective derangement to labels (no label maps to itself). |
| `gaussian_noise` | Data | `snr_db` | `20.0` | Signal-to-noise ratio in decibels. Lower = more noise. Must be > 0. |
| | | `attack_fraction` | `1.0` | Fraction of samples to poison. |
| `token_replacement` | Data | `replacement_fraction` | `0.3` | Fraction of input tokens/pixels to replace with random values. |
| `weight_spiking` | Model | `magnitude` | `100.0` | Spike magnitude. Must be > 0. |
| | | `spike_fraction` | `0.1` | Fraction of model parameters to spike. |
| `gradient_scaling` | Model | `scale_factor` | `10.0` | Multiplicative scaling factor for the update delta. Must be > 0. |
| `byzantine_perturbation` | Model | `noise_std` | `1.0` | Standard deviation of Gaussian noise added to model parameters. Must be > 0. |

### 6.3 Supported Models

| Display Name | Key | Architecture | Notes |
|---|---|---|---|
| Simple CNN | `cnn` | 2-conv + 2-FC | Lightweight, suitable for MNIST/CIFAR |
| MLP | `mlp` | Fully connected | Input is flattened; no convolutions |
| ResNet-18 | `resnet18` | 18-layer residual network | Adapted for variable input channels and image sizes |
| DenseNet-121 | `densenet121` | 121-layer dense network | Adapted for variable input channels and image sizes |

### 6.4 Supported Datasets

| Display Name | Key | Channels | Classes | Image Size |
|---|---|---|---|---|
| CIFAR-10 | `cifar10` | 3 | 10 | 32x32 |
| CIFAR-100 | `cifar100` | 3 | 100 | 32x32 |
| MNIST | `mnist` | 1 | 10 | 28x28 |
| Fashion-MNIST | `fashion_mnist` | 1 | 10 | 28x28 |
| SVHN | `svhn` | 3 | 10 | 32x32 |
| FEMNIST (62 classes) | `femnist` | 1 | 62 | 28x28 |
| PathMNIST (Medical) | `medmnist_pathmnist` | 3 | 9 | 28x28 |
| DermaMNIST (Medical) | `medmnist_dermamnist` | 3 | 7 | 28x28 |
| BloodMNIST (Medical) | `medmnist_bloodmnist` | 3 | 8 | 28x28 |
| OrganAMNIST (Medical) | `medmnist_organamnist` | 1 | 11 | 28x28 |

### 6.5 Training Constants

These are internal constants defined in `configs/defaults.py`:

| Constant | Value | Description |
|---|---|---|
| `GRADIENT_CLIP_MAX_NORM` | `10.0` | Maximum gradient norm for gradient clipping during local training. |
| `CLIENT_EVAL_BATCH_LIMIT` | `2` | Number of test batches used for quick per-client accuracy evaluation. |
| `DERANGEMENT_MAX_ATTEMPTS` | `1000` | Maximum iterations for generating a valid label derangement in label flipping. |

---

## 7. Glossary

**Federated Learning (FL)**
A distributed machine learning paradigm where multiple clients collaboratively train a shared global model without exchanging raw data. Each client trains locally on its own data and sends only model updates (parameters or gradients) to a central server for aggregation.

**FedAvg (Federated Averaging)**
The foundational FL aggregation strategy (McMahan et al., 2017). The server averages client model updates weighted by each client's number of training examples. Simple and effective under benign conditions but vulnerable to adversarial attacks.

**Non-IID (Non-Independently and Identically Distributed)**
A data distribution where clients have different label distributions. In real FL deployments, hospitals may see different diseases, phones may have different typing patterns, etc. Non-IID data makes FL harder because client updates point in different directions even without attacks.

**Dirichlet Distribution**
A probability distribution over probability vectors, parameterized by a concentration parameter alpha. In FEDSIM, it controls how heterogeneous client data partitions are. Lower alpha values (e.g., 0.1) create highly skewed partitions where each client has mostly one or two classes; higher values (e.g., 10.0) approach IID; alpha = 0.5 (the default) produces moderate heterogeneity.

**Byzantine Fault**
A failure mode where a participant in a distributed system behaves arbitrarily, including sending intentionally harmful data. In FL, a Byzantine client can send arbitrary model updates to degrade the global model.

**Label Flipping**
A data poisoning attack where a malicious client applies a bijective derangement to its training labels (every label is remapped to a different class, no label maps to itself). This causes the client to learn incorrect associations and send harmful updates.

**Data Poisoning**
A class of attacks where the adversary corrupts the training data itself. In FEDSIM, data poisoning attacks include label flipping, Gaussian noise injection, and token replacement. The poisoned data is used for local training, producing corrupted model updates.

**Model Poisoning**
A class of attacks where the adversary directly manipulates model parameters or gradients after local training. In FEDSIM, model poisoning attacks include weight spiking, gradient scaling, and Byzantine perturbation. These attacks modify the client's update before it is sent to the server.

**Krum**
A Byzantine-robust aggregation strategy (Blanchard et al., 2017). Krum scores each client by the sum of its distances to its nearest neighbors and selects the client with the lowest score. Multi-Krum averages the top-m scoring clients instead of selecting just one. Requires n >= 2f + 3 clients for f Byzantine faults.

**Trimmed Mean**
A robust aggregation strategy that computes the coordinate-wise mean after trimming the highest and lowest beta fraction of values for each parameter. The trim fraction beta should be at least as large as the malicious fraction to remove adversarial outliers.

**Median**
A robust aggregation strategy that computes the coordinate-wise median of all client updates. The median is inherently robust to outliers when fewer than half of clients are malicious.

**Bulyan**
A two-stage meta-aggregation strategy (El Mhamdi et al., 2018). First, iteratively applies Krum to select n - 2f candidate updates. Then applies coordinate-wise trimmed mean to the candidates. Requires n >= 4f + 3 for full Byzantine resilience, providing stronger theoretical guarantees than Krum alone.

**RFA (Robust Federated Averaging / Geometric Median)**
An aggregation strategy that computes the geometric median of client updates using Weiszfeld's iterative algorithm. The geometric median minimizes the sum of Euclidean distances to all points and is naturally robust to outliers.

**Reputation-Based Selection**
An aggregation strategy that maintains per-client reputation scores across rounds. Each round, it computes a "truth value" measuring how close each client's update is to the group centroid, then updates reputations: linear growth for trustworthy clients (truth >= threshold), exponential decay for suspicious ones. Only the top-k clients by reputation are included in weighted FedAvg aggregation.

**Trust Score**
A per-client metric computed each round by FEDSIM (independent of the aggregation strategy). It combines two signals: (1) cosine similarity between the client's update direction and the median direction, and (2) normalized L2 distance from the median update. Both are mapped to [0, 1] and averaged with equal weight. Trust = 1.0 means the client's update closely matches the group consensus.

**Anomaly Detection (in FEDSIM context)**
The process of evaluating whether a strategy's client exclusion decisions align with ground-truth malicious labels. FEDSIM has full visibility into which clients are malicious (since it runs the simulation), so it can compute exact precision, recall, and F1 for the exclusion decisions made by robust strategies.

**Removal F1**
The F1 score for client exclusion decisions, treating exclusion as a binary classification task: "should this client be excluded?" F1 = 2 * precision * recall / (precision + recall). A perfect removal F1 of 1.0 means every malicious client was excluded and no benign client was incorrectly excluded.

**TP / FP / TN / FN (in Client Exclusion Context)**

| Term | Full Name | Meaning |
|---|---|---|
| **TP** | True Positive | A malicious client was correctly excluded from aggregation |
| **FP** | False Positive | A benign client was incorrectly excluded from aggregation |
| **TN** | True Negative | A benign client was correctly included in aggregation |
| **FN** | False Negative | A malicious client was incorrectly included in aggregation (missed) |

---

## 8. Reproducing Published Results

FEDSIM includes a script for reproducing experiments from the capstone report *"Exploring the Robustness of Federated Learning for Image Classification"* (Abhik Roy, RIT).

### Capstone Experiment 2: FedAvg vs Reputation under Label Flipping

This experiment compares FedAvg and Reputation-based aggregation with 1, 2, and 3 poisoned clients out of 6 total.

```bash
python reproduce_capstone.py
```

**Configuration:**

| Parameter | Value |
|---|---|
| Model | Simple CNN |
| Dataset | CIFAR-10 |
| Clients | 6 |
| Rounds | 15 |
| Local Epochs | 3 |
| Learning Rate | 0.01 |
| Partition | IID |
| Attack | Label Flipping |
| Reputation Selection | Top 4 of 6 (67%) |
| Reputation Truth Threshold | 0.7 |
| Initial Reputation | 0.5 |

The script produces a PDF report (`capstone_reproduction.pdf`) with per-client accuracy bar charts, reputation score trajectories, global accuracy comparison, and a summary page.

### Designing Your Own Reproducible Experiments

To set up experiments matching published FL papers, use `SimulationConfig` programmatically:

```python
from simulation.runner import SimulationConfig, AttackConfig, run_simulation

config = SimulationConfig(
    model_name="resnet18",
    dataset_name="cifar10",
    num_clients=20,
    num_rounds=50,
    local_epochs=5,
    learning_rate=0.01,
    partition_type="non_iid",
    alpha=0.1,                  # highly non-IID
    strategies=["fedavg", "krum", "trimmed_mean", "reputation"],
    seed=42,                    # for reproducibility
    attack=AttackConfig(
        attack_type="byzantine_perturbation",
        malicious_fraction=0.3,
        attack_params={"noise_std": 1.0},
    ),
)

results = run_simulation(config)

for r in results:
    print(f"{r.strategy_name}: final_acc={r.round_accuracies[-1]:.4f}, "
          f"removal_f1={r.anomaly_summary.get('cumulative_f1', 'N/A')}")
```

**Tips for matching published experiments:**

1. **Match the seed.** If a paper reports specific results, fixing the seed ensures identical client selection and data partitions.
2. **Use IID partitioning** if the paper assumes homogeneous data, or set alpha to match their Dirichlet parameter.
3. **Match the malicious fraction** exactly. The number of malicious clients is `max(1, int(num_clients * malicious_fraction))`, clamped to at most `num_clients - 1`.
4. **Use callbacks** (`round_callback`, `client_callback`) to capture per-round and per-client metrics for detailed analysis.
5. **Export the config JSON** from the Results tab to document exactly what was run.
