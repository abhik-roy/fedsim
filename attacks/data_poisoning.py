import numpy as np
import torch
from torch.utils.data import Dataset


class LabelFlippedDataset(Dataset):
    """Wraps a dataset and applies a bijective permutation to all labels."""

    def __init__(self, dataset, num_classes: int, seed: int = 42):
        self.dataset = dataset
        if num_classes < 2:
            raise ValueError("Label flipping requires at least 2 classes")
        rng = np.random.default_rng(seed)
        # Create a bijective permutation where no class maps to itself (derangement)
        if num_classes == 2:
            perm = np.array([1, 0])
        else:
            perm = np.arange(num_classes)
            identity = np.arange(num_classes)
            from configs.defaults import DERANGEMENT_MAX_ATTEMPTS
            for _ in range(DERANGEMENT_MAX_ATTEMPTS):
                rng.shuffle(perm)
                if not np.any(perm == identity):
                    break
            else:
                # Deterministic fallback: cyclic shift by 1
                perm = np.roll(identity, 1)
        self.label_map = {int(i): int(perm[i]) for i in range(num_classes)}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data, label = self.dataset[idx]
        return data, self.label_map.get(int(label), int(label))


class GaussianNoiseDataset(Dataset):
    """Wraps a dataset and injects additive Gaussian noise to input samples."""

    def __init__(self, dataset, snr_db: float = 20.0, attack_fraction: float = 1.0, seed: int = 42):
        self.dataset = dataset
        self.snr_db = snr_db
        self.attack_fraction = attack_fraction
        self.rng = np.random.default_rng(seed)
        self._torch_gen = torch.Generator().manual_seed(seed)
        n = len(dataset)
        self.attacked_indices = set(
            self.rng.choice(n, size=int(n * attack_fraction), replace=False)
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data, label = self.dataset[idx]
        if idx in self.attacked_indices and data.is_floating_point():
            signal_power = data.pow(2).mean().item()
            snr_linear = 10 ** (self.snr_db / 10)
            noise_power = signal_power / max(snr_linear, 1e-10)
            noise_std = max(noise_power ** 0.5, 1e-4)  # Minimum noise floor
            # Use index-seeded generator for reproducible per-sample noise
            gen = torch.Generator().manual_seed(self._torch_gen.initial_seed() + idx)
            noise = torch.randn(data.shape, generator=gen) * noise_std
            data = data + noise
        return data, label


class TokenReplacedDataset(Dataset):
    """Wraps a dataset and randomly replaces input features to simulate token replacement.

    For image datasets, this replaces a rectangular region (sized by
    replacement_fraction) with content from a donor sample. For 1D data,
    replaces a contiguous segment of that fraction.

    Args:
        replacement_fraction: Fraction of the spatial area / feature length to
            replace in each attacked sample (default 0.3 = 30%).
    """

    def __init__(self, dataset, replacement_fraction: float = 0.3, seed: int = 42):
        self.dataset = dataset
        self.replacement_fraction = replacement_fraction
        self.n = len(dataset)
        # Pre-select attacked indices and donor indices for determinism
        rng = np.random.default_rng(seed)
        self.attacked_indices = set(rng.choice(self.n, size=self.n, replace=False))
        self._donor_indices = rng.integers(0, self.n, size=self.n)
        # Avoid self-donation
        for i in range(self.n):
            if self._donor_indices[i] == i:
                self._donor_indices[i] = (i + 1) % self.n
        self._seed = seed

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data, label = self.dataset[idx]
        if idx in self.attacked_indices:
            donor_idx = int(self._donor_indices[idx])
            donor_data, _ = self.dataset[donor_idx]
            # Use index-seeded RNG for reproducible per-sample patch location
            rng = np.random.default_rng(self._seed + idx)
            frac = self.replacement_fraction
            if data.dim() >= 2:
                h, w = data.shape[-2], data.shape[-1]
                rh = max(1, int(h * frac ** 0.5))
                rw = max(1, int(w * frac ** 0.5))
                y0 = rng.integers(0, max(1, h - rh))
                x0 = rng.integers(0, max(1, w - rw))
                data = data.clone()
                data[..., y0:y0 + rh, x0:x0 + rw] = donor_data[..., y0:y0 + rh, x0:x0 + rw]
            else:
                # 1D data: replace a contiguous segment
                n_features = data.shape[0]
                seg_len = max(1, int(n_features * frac))
                start = rng.integers(0, max(1, n_features - seg_len))
                data = data.clone()
                data[start:start + seg_len] = donor_data[start:start + seg_len]
        return data, label


def apply_label_flipping(dataset, num_classes: int, seed: int = 42):
    """Apply label flipping attack to a dataset partition."""
    return LabelFlippedDataset(dataset, num_classes, seed)


def apply_gaussian_noise(dataset, snr_db: float = 20.0, attack_fraction: float = 1.0, seed: int = 42):
    """Apply Gaussian noise injection to a dataset partition."""
    return GaussianNoiseDataset(dataset, snr_db, attack_fraction, seed)


def apply_token_replacement(dataset, replacement_fraction: float = 0.3, seed: int = 42):
    """Apply token replacement attack to a dataset partition."""
    return TokenReplacedDataset(dataset, replacement_fraction, seed)
