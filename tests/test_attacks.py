import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from attacks.data_poisoning import LabelFlippedDataset, GaussianNoiseDataset
from attacks.model_poisoning import (
    apply_weight_spiking,
    apply_gradient_scaling,
    apply_byzantine_perturbation,
)


def _make_dataset(num_samples=50, num_classes=10):
    data = torch.randn(num_samples, 1, 8, 8)
    targets = torch.arange(num_samples) % num_classes
    return TensorDataset(data, targets)


# ── Label Flipping Tests ─────────────────────────────────────────

class TestLabelFlipping:
    def test_label_flip_derangement(self):
        """Every class should map to a different class (derangement property)."""
        ds = _make_dataset(100, num_classes=10)
        flipped = LabelFlippedDataset(ds, num_classes=10, seed=42)
        for original_label, new_label in flipped.label_map.items():
            assert original_label != new_label, (
                f"Label {original_label} maps to itself"
            )

    def test_label_flip_deterministic(self):
        """Same seed should produce the same permutation."""
        ds = _make_dataset(100, num_classes=10)
        flipped1 = LabelFlippedDataset(ds, num_classes=10, seed=99)
        flipped2 = LabelFlippedDataset(ds, num_classes=10, seed=99)
        assert flipped1.label_map == flipped2.label_map

    def test_label_flip_different_seeds(self):
        """Different seeds should (very likely) produce different permutations."""
        ds = _make_dataset(100, num_classes=10)
        flipped1 = LabelFlippedDataset(ds, num_classes=10, seed=1)
        flipped2 = LabelFlippedDataset(ds, num_classes=10, seed=2)
        # With 10 classes, chance of same derangement from different seeds is tiny
        assert flipped1.label_map != flipped2.label_map

    def test_label_flip_preserves_data(self):
        """Data should be unchanged; only labels differ."""
        ds = _make_dataset(20, num_classes=5)
        flipped = LabelFlippedDataset(ds, num_classes=5, seed=42)
        for i in range(len(ds)):
            orig_data, _ = ds[i]
            flip_data, _ = flipped[i]
            assert torch.equal(orig_data, flip_data)


# ── Gaussian Noise Tests ─────────────────────────────────────────

class TestGaussianNoise:
    def test_gaussian_noise_changes_data(self):
        """Noised data should differ from original."""
        ds = _make_dataset(20, num_classes=5)
        noised = GaussianNoiseDataset(ds, snr_db=20.0, attack_fraction=1.0, seed=42)
        changed = 0
        for i in range(len(ds)):
            orig_data, _ = ds[i]
            noised_data, _ = noised[i]
            if not torch.equal(orig_data, noised_data):
                changed += 1
        assert changed > 0, "No samples were modified by noise"

    def test_gaussian_noise_zero_power_still_adds_noise(self):
        """Zero-valued data should still get noise thanks to the 1e-4 noise floor."""
        zero_data = torch.zeros(10, 1, 4, 4)
        targets = torch.zeros(10, dtype=torch.long)
        ds = TensorDataset(zero_data, targets)
        noised = GaussianNoiseDataset(ds, snr_db=20.0, attack_fraction=1.0, seed=42)
        data_noised, _ = noised[0]
        assert not torch.equal(data_noised, zero_data[0]), (
            "Zero-valued data was not modified — noise floor may be broken"
        )

    def test_gaussian_noise_preserves_labels(self):
        """Labels should remain unchanged."""
        ds = _make_dataset(20, num_classes=5)
        noised = GaussianNoiseDataset(ds, snr_db=20.0, attack_fraction=1.0, seed=42)
        for i in range(len(ds)):
            _, orig_label = ds[i]
            _, noised_label = noised[i]
            assert orig_label == noised_label


# ── Model Poisoning Tests ────────────────────────────────────────

def _make_params(dim=100, seed=42):
    """Create simple float32 parameter arrays mimicking a small model."""
    rng = np.random.default_rng(seed)
    return [rng.normal(size=(dim,)).astype(np.float32)]


class TestWeightSpiking:
    def test_weight_spiking_changes_params(self):
        params = _make_params()
        spiked = apply_weight_spiking(params, magnitude=100.0, spike_fraction=0.5, seed=42)
        assert not np.array_equal(params[0], spiked[0])

    def test_weight_spiking_preserves_shape(self):
        params = _make_params(dim=200)
        spiked = apply_weight_spiking(params, magnitude=10.0, spike_fraction=0.1)
        assert spiked[0].shape == params[0].shape


class TestGradientScaling:
    def test_gradient_scaling_multiplies(self):
        params = _make_params()
        scaled = apply_gradient_scaling(params, scale_factor=2.0)
        # Without global params, it should just multiply by scale_factor
        np.testing.assert_allclose(scaled[0], params[0] * 2.0, rtol=1e-6)

    def test_gradient_scaling_with_global_params(self):
        local = _make_params(seed=1)
        glob = _make_params(seed=2)
        scaled = apply_gradient_scaling(local, scale_factor=3.0, global_parameters=glob)
        expected = glob[0] + 3.0 * (local[0] - glob[0])
        np.testing.assert_allclose(scaled[0], expected, rtol=1e-5)


class TestByzantinePerturbation:
    def test_byzantine_perturbation_changes_params(self):
        params = _make_params()
        perturbed = apply_byzantine_perturbation(params, noise_std=1.0, seed=42)
        assert not np.array_equal(params[0], perturbed[0])

    def test_byzantine_perturbation_preserves_shape(self):
        params = _make_params(dim=200)
        perturbed = apply_byzantine_perturbation(params, noise_std=1.0, seed=42)
        assert perturbed[0].shape == params[0].shape
