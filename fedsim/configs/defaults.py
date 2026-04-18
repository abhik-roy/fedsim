SUPPORTED_MODELS = {
    "ResNet-18": "resnet18",
    "DenseNet-121": "densenet121",
    "Simple CNN": "cnn",
    "MLP": "mlp",
}

SUPPORTED_DATASETS = {
    "CIFAR-100": "cifar100",
    "CIFAR-10": "cifar10",
    "PathMNIST (Medical)": "medmnist_pathmnist",
    "DermaMNIST (Medical)": "medmnist_dermamnist",
    "BloodMNIST (Medical)": "medmnist_bloodmnist",
    "OrganAMNIST (Medical)": "medmnist_organamnist",
    "Fashion-MNIST": "fashion_mnist",
    "SVHN": "svhn",
    "MNIST": "mnist",
    "FEMNIST (62 classes)": "femnist",
}

SUPPORTED_STRATEGIES = {
    "FedAvg": "fedavg",
    "Trimmed Mean": "trimmed_mean",
    "Krum": "krum",
    "Median": "median",
    "Bulyan": "bulyan",
    "RFA (Geometric Median)": "rfa",
    # Reputation is a plugin (custom/strategies/reputation.py) — auto-discovered
}

PARTITION_TYPES = ["IID", "Non-IID (Dirichlet)"]

DEFAULT_MODEL = "resnet18"
DEFAULT_DATASET = "cifar10"
DEFAULT_PARTITION_TYPE = "non_iid"
DEFAULT_STRATEGIES = ["fedavg", "custom:Reputation"]
DEFAULT_ATTACK = "label_flipping"

DEFAULT_NUM_CLIENTS = 10
DEFAULT_NUM_ROUNDS = 20
DEFAULT_LOCAL_EPOCHS = 3
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_ALPHA = 0.5
DEFAULT_PARALLEL_CLIENTS = 1

DATASET_INFO = {
    "mnist": {"input_channels": 1, "num_classes": 10, "input_size": 784, "image_size": 28, "task_type": "image_classification"},
    "fashion_mnist": {"input_channels": 1, "num_classes": 10, "input_size": 784, "image_size": 28, "task_type": "image_classification"},
    "cifar10": {"input_channels": 3, "num_classes": 10, "input_size": 3072, "image_size": 32, "task_type": "image_classification"},
    "cifar100": {"input_channels": 3, "num_classes": 100, "input_size": 3072, "image_size": 32, "task_type": "image_classification"},
    "svhn": {"input_channels": 3, "num_classes": 10, "input_size": 3072, "image_size": 32, "task_type": "image_classification"},
    "femnist": {"input_channels": 1, "num_classes": 62, "input_size": 784, "image_size": 28, "task_type": "image_classification"},
    # MedMNIST datasets — 28x28 medical images
    "medmnist_pathmnist": {"input_channels": 3, "num_classes": 9, "input_size": 2352, "image_size": 28, "task_type": "image_classification"},
    "medmnist_dermamnist": {"input_channels": 3, "num_classes": 7, "input_size": 2352, "image_size": 28, "task_type": "image_classification"},
    "medmnist_bloodmnist": {"input_channels": 3, "num_classes": 8, "input_size": 2352, "image_size": 28, "task_type": "image_classification"},
    "medmnist_organamnist": {"input_channels": 1, "num_classes": 11, "input_size": 784, "image_size": 28, "task_type": "image_classification"},
}

# --- Attack Configuration ---
SUPPORTED_ATTACKS = {
    "None": "none",
    "Label Flipping": "label_flipping",
    "Gaussian Noise": "gaussian_noise",
    "Token Replacement": "token_replacement",
    "Weight Spiking": "weight_spiking",
    "Gradient Scaling": "gradient_scaling",
    "Byzantine Perturbation": "byzantine_perturbation",
}

ATTACK_CATEGORIES = {
    "none": None,
    "label_flipping": "data",
    "gaussian_noise": "data",
    "token_replacement": "data",
    "weight_spiking": "model",
    "gradient_scaling": "model",
    "byzantine_perturbation": "model",
}

DEFAULT_ATTACK_PARAMS = {
    "label_flipping": {},
    "gaussian_noise": {"snr_db": 20.0, "attack_fraction": 1.0},
    "token_replacement": {"replacement_fraction": 0.3},
    "weight_spiking": {"magnitude": 100.0, "spike_fraction": 0.1},
    "gradient_scaling": {"scale_factor": 10.0},
    "byzantine_perturbation": {"noise_std": 1.0},
}

ATTACK_SCHEDULE_TYPES = ["Static (all rounds)", "Dynamic (select rounds)"]

DEFAULT_MALICIOUS_FRACTION = 0.3

# ── Training Constants ────────────────────────────────────────────
GRADIENT_CLIP_MAX_NORM = 10.0       # max gradient norm for clipping
CLIENT_EVAL_BATCH_LIMIT = 2         # number of test batches for quick per-client eval
DERANGEMENT_MAX_ATTEMPTS = 1000     # max iterations for label flipping derangement

# ── Client Sampling Defaults ────────────────────────────────────
DEFAULT_FRACTION_FIT = 1.0
DEFAULT_FRACTION_EVALUATE = 1.0
DEFAULT_MIN_FIT_CLIENTS = 2
DEFAULT_MIN_EVALUATE_CLIENTS = 2
DEFAULT_MIN_AVAILABLE_CLIENTS = 2

# ── Optimizer & Loss Configuration ────────────────────────────────
SUPPORTED_OPTIMIZERS = {
    "SGD": "sgd",
    "Adam": "adam",
    "AdamW": "adamw",
}

SUPPORTED_LOSSES = {
    "Cross Entropy": "cross_entropy",
    "Binary Cross Entropy": "bce_with_logits",
    "Negative Log Likelihood": "nll",
}

DEFAULT_OPTIMIZER = "sgd"
DEFAULT_LOSS = "cross_entropy"

# ── LR Scheduler Configuration ──────────────────────────────────
SUPPORTED_SCHEDULERS = {
    "None": "none",
    "StepLR": "step_lr",
    "Cosine Annealing": "cosine_annealing",
    "Exponential": "exponential",
}

DEFAULT_LR_SCHEDULER = "none"

DEFAULT_LR_SCHEDULER_PARAMS = {
    "step_lr": {"step_size": 5, "gamma": 0.1},
    "cosine_annealing": {"T_max": 3},
    "exponential": {"gamma": 0.95},
}

# ── Data Split Configuration ─────────────────────────────────────
DEFAULT_VAL_SPLIT = 0.0             # 0.0 = no validation split (all training)
DEFAULT_TEST_EVAL_FREQUENCY = 1     # evaluate on test set every N rounds

# ── Compatibility Metadata ──────────────────────────────────────
MODEL_COMPATIBLE_TASKS = {
    "cnn": ["image_classification"],
    "mlp": ["image_classification", "regression"],
    "resnet18": ["image_classification"],
    "densenet121": ["image_classification"],
}

ATTACK_COMPATIBLE_TASKS = {
    "label_flipping": ["image_classification", "text_classification"],
    "token_replacement": ["text_classification", "language_modeling", "token_classification"],
}

# ── Preset Configurations ───────────────────────────────────────
PRESETS = {
    "Quick Demo": {
        "model": "cnn", "dataset": "mnist",
        "num_clients": 5, "num_rounds": 5, "local_epochs": 2,
        "learning_rate": 0.01, "partition_type": "iid",
        "strategies": ["fedavg", "krum"],
        "attack_type": "label_flipping", "malicious_fraction": 0.2,
    },
    "Byzantine Robustness": {
        "model": "resnet18", "dataset": "cifar10",
        "num_clients": 15, "num_rounds": 15, "local_epochs": 3,
        "learning_rate": 0.01, "partition_type": "non_iid", "alpha": 0.5,
        "strategies": ["fedavg", "trimmed_mean", "krum", "median",
                        "custom:Reputation", "bulyan", "rfa"],
        "attack_type": "byzantine_perturbation", "malicious_fraction": 0.2,
    },
    "Text Classification": {
        "model": "custom:TextCNN", "dataset": "custom:AG News (4-class)",
        "num_clients": 5, "num_rounds": 10, "local_epochs": 2,
        "learning_rate": 0.01, "partition_type": "iid",
        "strategies": ["fedavg", "custom:Reputation"],
        "attack_type": "label_flipping", "malicious_fraction": 0.2,
    },
    "Regression": {
        "model": "custom:RegressionMLP", "dataset": "custom:California Housing",
        "num_clients": 5, "num_rounds": 10, "local_epochs": 3,
        "learning_rate": 0.001, "partition_type": "iid",
        "strategies": ["fedavg", "trimmed_mean"],
        "attack_type": "gaussian_noise", "malicious_fraction": 0.2,
    },
    "Language Modeling": {
        "model": "custom:Small LM", "dataset": "custom:WikiText-2",
        "num_clients": 3, "num_rounds": 5, "local_epochs": 2,
        "learning_rate": 0.001, "partition_type": "iid",
        "strategies": ["fedavg"],
        "attack_type": "none", "malicious_fraction": 0.0,
    },
}
