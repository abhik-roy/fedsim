import numpy as np
from torch.utils.data import Subset


def partition_dataset(dataset, num_clients: int, partition_type: str = "iid", alpha: float = 0.5, seed: int = 42):
    """Partition a dataset into per-client subsets for federated simulation.

    Args:
        dataset: PyTorch Dataset to partition.
        num_clients: Number of FL clients.
        partition_type: 'iid' for uniform random split, 'non_iid' for Dirichlet-based.
        alpha: Dirichlet concentration parameter (lower = more heterogeneous). Only used
               when partition_type='non_iid'.
        seed: Random seed for reproducibility.

    Returns:
        List of torch.utils.data.Subset, one per client.
    """
    rng = np.random.default_rng(seed)
    num_samples = len(dataset)

    if partition_type == "iid":
        indices = rng.permutation(num_samples)
        splits = np.array_split(indices, num_clients)
        return [Subset(dataset, split.tolist()) for split in splits]

    elif partition_type == "non_iid":
        if alpha <= 0:
            raise ValueError(f"Dirichlet alpha must be > 0, got {alpha}")
        if alpha < 0.01:
            import warnings
            warnings.warn(f"Dirichlet alpha={alpha} is very small; most clients will have near-zero samples")

        # Use .targets attribute directly when available to avoid applying transforms
        if hasattr(dataset, 'targets'):
            raw = dataset.targets
            targets = np.array(raw.numpy() if hasattr(raw, 'numpy') else raw)
        elif hasattr(dataset, 'labels'):
            raw = dataset.labels
            targets = np.array(raw.numpy() if hasattr(raw, 'numpy') else raw).flatten()
        else:
            targets = np.array([dataset[i][1] for i in range(num_samples)])

        # Guard: non-IID requires integer class labels
        if targets.dtype.kind == 'f':  # float targets (regression, LM)
            import warnings
            warnings.warn("Non-IID partitioning requires integer class labels. "
                          "Falling back to IID partitioning for this dataset.")
            rng = np.random.default_rng(seed)
            indices = rng.permutation(num_samples)
            splits = np.array_split(indices, num_clients)
            return [Subset(dataset, split.tolist()) for split in splits]

        classes = np.unique(targets)
        num_classes = len(classes)

        class_indices = {c: np.where(targets == c)[0] for c in classes}

        proportions = rng.dirichlet(np.repeat(alpha, num_clients), size=num_classes)

        client_indices = [[] for _ in range(num_clients)]

        for c_idx, c in enumerate(classes):
            c_inds = class_indices[c]
            rng.shuffle(c_inds)
            splits = (proportions[c_idx] * len(c_inds)).astype(int)
            # Clamp to avoid over-allocation from rounding
            cumsum = np.cumsum(splits)
            cumsum = np.minimum(cumsum, len(c_inds))
            splits = np.diff(np.concatenate([[0], cumsum]))
            # Assign remainder to last client
            splits[-1] = len(c_inds) - splits[:-1].sum()

            start = 0
            for client_id in range(num_clients):
                end = start + max(0, splits[client_id])
                client_indices[client_id].extend(c_inds[start:end].tolist())
                start = end

        # Ensure every client has at least 1 sample
        for i in range(num_clients):
            if len(client_indices[i]) == 0:
                # Steal one sample from the largest partition
                largest = max(range(num_clients), key=lambda x: len(client_indices[x]))
                client_indices[i].append(client_indices[largest].pop())

        # Warn about very small partitions that may hurt training quality
        import warnings
        min_samples = min(len(ci) for ci in client_indices)
        if min_samples < 10:
            warnings.warn(
                f"Non-IID partitioning created a client with only {min_samples} sample(s). "
                f"Consider increasing alpha (currently {alpha}) or reducing num_clients "
                f"(currently {num_clients}) for more balanced partitions.",
                stacklevel=2,
            )

        return [Subset(dataset, indices) for indices in client_indices]

    else:
        raise ValueError(f"Unknown partition type: {partition_type}")
