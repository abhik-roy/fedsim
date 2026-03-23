import os
from pathlib import Path

import torchvision
import torchvision.transforms as transforms


def _default_data_dir() -> str:
    """Resolve the default data directory.

    Priority: FEDSIM_DATA_DIR env var > ~/.fedsim/data > ./data/raw
    """
    env_dir = os.environ.get("FEDSIM_DATA_DIR")
    if env_dir:
        return env_dir
    home_dir = Path.home() / ".fedsim" / "data"
    if home_dir.exists():
        return str(home_dir)
    return str(Path.home() / ".fedsim" / "data")


def get_dataset(dataset_name: str, data_dir: str | None = None, **kwargs):
    """Load a dataset by name.

    Args:
        dataset_name: One of the supported dataset keys (e.g., 'cifar10', 'mnist').
        data_dir: Directory to store/load dataset files. Defaults to ~/.fedsim/data
                  or the FEDSIM_DATA_DIR environment variable.

    Returns:
        Tuple of (train_dataset, test_dataset).
    """
    if data_dir is None:
        data_dir = _default_data_dir()
    if dataset_name == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        train = torchvision.datasets.MNIST(data_dir, train=True, download=True, transform=transform)
        test = torchvision.datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    elif dataset_name == "fashion_mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
        ])
        train = torchvision.datasets.FashionMNIST(data_dir, train=True, download=True, transform=transform)
        test = torchvision.datasets.FashionMNIST(data_dir, train=False, download=True, transform=transform)

    elif dataset_name == "cifar10":
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        train = torchvision.datasets.CIFAR10(data_dir, train=True, download=True, transform=train_transform)
        test = torchvision.datasets.CIFAR10(data_dir, train=False, download=True, transform=test_transform)

    elif dataset_name == "cifar100":
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        train = torchvision.datasets.CIFAR100(data_dir, train=True, download=True, transform=train_transform)
        test = torchvision.datasets.CIFAR100(data_dir, train=False, download=True, transform=test_transform)

    elif dataset_name == "svhn":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
        ])
        train = torchvision.datasets.SVHN(data_dir, split="train", download=True, transform=transform)
        test = torchvision.datasets.SVHN(data_dir, split="test", download=True, transform=transform)

    elif dataset_name == "femnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1736,), (0.3317,)),
        ])
        train = torchvision.datasets.EMNIST(data_dir, split="byclass", train=True, download=True, transform=transform)
        test = torchvision.datasets.EMNIST(data_dir, split="byclass", train=False, download=True, transform=transform)

    elif dataset_name.startswith("medmnist_"):
        train, test = _load_medmnist(dataset_name, data_dir)

    elif dataset_name.startswith("custom:"):
        from plugins import discover_plugins
        plugin_name = dataset_name.replace("custom:", "")
        plugins = discover_plugins("datasets")
        for name, mod in plugins.items():
            if name == plugin_name and mod is not None:
                return mod.load(**kwargs)
        raise ValueError(f"Custom dataset plugin not found: {plugin_name}")

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return train, test


def _load_medmnist(dataset_name: str, data_dir: str):
    """Load a MedMNIST dataset. dataset_name format: medmnist_<subset>"""
    import medmnist
    from medmnist import INFO

    subset = dataset_name.replace("medmnist_", "")
    info = INFO[subset]
    num_channels = info["n_channels"]

    # MedMNIST images are 28x28, values in [0,1] when using transforms.ToTensor()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5] * num_channels, [0.5] * num_channels),
    ])

    DataClass = getattr(medmnist, info["python_class"])
    train = DataClass(split="train", transform=transform, download=True, root=data_dir)
    test = DataClass(split="test", transform=transform, download=True, root=data_dir)

    # MedMNIST returns labels as 2D arrays (N,1), wrap to return 1D labels
    train = _MedMNISTWrapper(train)
    test = _MedMNISTWrapper(test)

    return train, test


class _MedMNISTWrapper:
    """Wraps MedMNIST dataset to return scalar labels instead of (N,1) arrays."""

    def __init__(self, dataset):
        self.dataset = dataset
        # Pre-extract targets for fast non-IID partitioning
        import numpy as _np
        if hasattr(dataset, 'labels') and dataset.labels is not None:
            self.targets = _np.array(dataset.labels).flatten()
        else:
            self.targets = _np.array([int(dataset[i][1].squeeze()) for i in range(len(dataset))])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        return img, int(label.squeeze())
