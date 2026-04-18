import math

import pytest

from anomaly.metrics import AnomalyMetrics


class TestAnomalyMetricsNaN:
    def test_no_malicious_no_exclusion_returns_nan(self):
        """When there are no malicious clients and no exclusions,
        precision/recall/f1 should be NaN (vacuous case)."""
        m = AnomalyMetrics()
        result = m.compute_round(
            malicious_clients=set(),
            excluded_clients=set(),
            all_clients=set(range(5)),
        )
        assert result["tp"] == 0
        assert result["fp"] == 0
        assert result["fn"] == 0
        assert result["tn"] == 5
        assert math.isnan(result["precision"])
        assert math.isnan(result["recall"])
        assert math.isnan(result["f1"])

    def test_malicious_with_exclusion_returns_numbers(self):
        """When there are malicious clients and exclusions, metrics
        should be valid floats (not NaN)."""
        m = AnomalyMetrics()
        result = m.compute_round(
            malicious_clients={0, 1},
            excluded_clients={0, 1, 2},
            all_clients=set(range(10)),
        )
        assert result["tp"] == 2
        assert result["fp"] == 1
        assert result["fn"] == 0
        assert result["tn"] == 7
        assert not math.isnan(result["precision"])
        assert not math.isnan(result["recall"])
        assert not math.isnan(result["f1"])
        assert result["precision"] == pytest.approx(2 / 3)
        assert result["recall"] == 1.0
        assert result["f1"] == pytest.approx(2 * (2 / 3) * 1.0 / ((2 / 3) + 1.0))

    def test_summary_nan_when_all_rounds_vacuous(self):
        """Summary should also return NaN if all rounds were vacuous."""
        m = AnomalyMetrics()
        m.compute_round(set(), set(), set(range(5)))
        m.compute_round(set(), set(), set(range(5)))
        summary = m.summary()
        assert math.isnan(summary["cumulative_precision"])
        assert math.isnan(summary["cumulative_recall"])
        assert math.isnan(summary["cumulative_f1"])

    def test_summary_valid_when_malicious_present(self):
        """Summary should return valid floats when malicious clients exist."""
        m = AnomalyMetrics()
        m.compute_round({0, 1}, {0, 1}, set(range(5)))
        summary = m.summary()
        assert not math.isnan(summary["cumulative_precision"])
        assert summary["cumulative_precision"] == 1.0
        assert summary["cumulative_recall"] == 1.0
