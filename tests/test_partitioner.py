import warnings

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from data.partitioner import partition_dataset


def _make_dataset(num_samples, num_classes=10):
    """Create a simple synthetic dataset with balanced classes."""
    data = torch.randn(num_samples, 1, 8, 8)
    targets = torch.arange(num_samples) % num_classes
    ds = TensorDataset(data, targets)
    # Attach .targets so the partitioner can use the fast path
    ds.targets = targets
    return ds


class TestIIDPartition:
    def test_iid_partition_equal_sizes(self):
        ds = _make_dataset(1000, num_classes=10)
        subsets = partition_dataset(ds, num_clients=10, partition_type="iid", seed=42)
        assert len(subsets) == 10
        sizes = [len(s) for s in subsets]
        assert all(s == 100 for s in sizes)

    def test_iid_partition_uneven(self):
        ds = _make_dataset(105, num_classes=5)
        subsets = partition_dataset(ds, num_clients=10, partition_type="iid", seed=42)
        assert len(subsets) == 10
        sizes = [len(s) for s in subsets]
        assert sum(sizes) == 105
        # np.array_split distributes remainder: some get 11, rest get 10
        assert all(s in (10, 11) for s in sizes)


class TestNonIIDPartition:
    def test_non_iid_dirichlet_produces_heterogeneous_splits(self):
        ds = _make_dataset(1000, num_classes=10)
        subsets = partition_dataset(ds, num_clients=10, partition_type="non_iid",
                                    alpha=0.1, seed=42)
        assert len(subsets) == 10
        # With alpha=0.1 the class distributions should vary across clients
        class_counts = []
        for subset in subsets:
            labels = [ds[i][1].item() for i in subset.indices]
            counts = np.bincount(labels, minlength=10)
            class_counts.append(counts)
        class_counts = np.array(class_counts)
        # Check that not all rows are identical (heterogeneous)
        assert not np.all(class_counts == class_counts[0])

    def test_non_iid_large_alpha_approaches_iid(self):
        ds = _make_dataset(1000, num_classes=10)
        subsets = partition_dataset(ds, num_clients=10, partition_type="non_iid",
                                    alpha=1000.0, seed=42)
        sizes = [len(s) for s in subsets]
        # With very large alpha, sizes should be relatively uniform
        assert max(sizes) - min(sizes) < 60  # generous tolerance for integer rounding

    def test_non_iid_alpha_zero_raises(self):
        ds = _make_dataset(100, num_classes=5)
        with pytest.raises(ValueError, match="alpha must be > 0"):
            partition_dataset(ds, num_clients=5, partition_type="non_iid", alpha=0)

    def test_non_iid_alpha_negative_raises(self):
        ds = _make_dataset(100, num_classes=5)
        with pytest.raises(ValueError, match="alpha must be > 0"):
            partition_dataset(ds, num_clients=5, partition_type="non_iid", alpha=-1)

    def test_non_iid_empty_partition_guard(self):
        """With very small alpha and many clients, the steal-one-sample fallback
        should ensure no client ends up with 0 samples."""
        ds = _make_dataset(200, num_classes=10)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            subsets = partition_dataset(ds, num_clients=20, partition_type="non_iid",
                                        alpha=0.01, seed=42)
        for i, subset in enumerate(subsets):
            assert len(subset) > 0, f"Client {i} has 0 samples"


class TestPartitionErrors:
    def test_unknown_partition_type_raises(self):
        ds = _make_dataset(100, num_classes=5)
        with pytest.raises(ValueError, match="Unknown partition type"):
            partition_dataset(ds, num_clients=5, partition_type="unknown")
