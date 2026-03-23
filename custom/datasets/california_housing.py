"""California Housing dataset plugin for regression tasks.

Median house value prediction from 8 features (location, rooms, income, etc.).
20,640 samples. Target: median house value (continuous, normalized to [0,1]).
"""

import numpy as np
import torch
from torch.utils.data import Dataset

NAME = "California Housing"
TASK_TYPE = "regression"
NUM_CLASSES = 1
INPUT_CHANNELS = 1
IMAGE_SIZE = 8
INPUT_SIZE = 8  # 8 features, override default calculation
SEQ_LENGTH = None
DESCRIPTION = "California Housing regression dataset (8 features, continuous target)"

PARAMS = {
    "normalize": {"type": "bool", "default": True, "label": "Normalize Features"},
}


class _RegressionDataset(Dataset):
    """Wrapper that provides .targets for non-IID partitioning."""

    def __init__(self, features, targets, target_bins=None):
        self.features = features
        self.target_values = targets
        # For non-IID partitioning: bin continuous targets into integer classes
        if target_bins is not None:
            self.targets = target_bins
        else:
            self.targets = np.zeros(len(targets), dtype=np.int64)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.target_values[idx]


def load(**kwargs):
    """Load California Housing dataset.

    Returns:
        (train_dataset, test_dataset) both with .targets attribute for partitioning.
    """
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split

    data = fetch_california_housing()
    X = data.data.astype(np.float32)
    y = data.target.astype(np.float32)

    # Normalize features to [0, 1]
    if kwargs.get("normalize", True):
        X_min = X.min(axis=0)
        X_max = X.max(axis=0)
        X = (X - X_min) / (X_max - X_min + 1e-8)

    # Normalize targets to [0, 1]
    y_min, y_max = y.min(), y.max()
    y = (y - y_min) / (y_max - y_min + 1e-8)

    # Split 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Bin targets for non-IID partitioning (5 quantile buckets)
    quantiles = np.quantile(y_train, [0.2, 0.4, 0.6, 0.8])
    train_bins = np.digitize(y_train, quantiles).astype(np.int64)
    test_bins = np.digitize(y_test, quantiles).astype(np.int64)

    train_ds = _RegressionDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train), train_bins
    )
    test_ds = _RegressionDataset(
        torch.from_numpy(X_test), torch.from_numpy(y_test), test_bins
    )

    return train_ds, test_ds
