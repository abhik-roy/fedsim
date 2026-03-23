import pytest
from anomaly.metrics import AnomalyMetrics

def test_perfect_detection():
    """All malicious excluded, all benign included."""
    m = AnomalyMetrics()
    malicious = {0, 1}
    excluded = {0, 1}
    all_clients = set(range(5))
    result = m.compute_round(malicious, excluded, all_clients)
    assert result["tp"] == 2
    assert result["fp"] == 0
    assert result["tn"] == 3
    assert result["fn"] == 0
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1"] == 1.0

def test_no_detection():
    """No clients excluded — all malicious are false negatives."""
    m = AnomalyMetrics()
    malicious = {0, 1}
    excluded = set()
    all_clients = set(range(5))
    result = m.compute_round(malicious, excluded, all_clients)
    assert result["tp"] == 0
    assert result["fp"] == 0
    assert result["tn"] == 3
    assert result["fn"] == 2
    assert result["precision"] == 0.0
    assert result["recall"] == 0.0
    assert result["f1"] == 0.0

def test_false_positives():
    """Benign clients incorrectly excluded."""
    m = AnomalyMetrics()
    malicious = {0}
    excluded = {0, 2, 3}
    all_clients = set(range(5))
    result = m.compute_round(malicious, excluded, all_clients)
    assert result["tp"] == 1
    assert result["fp"] == 2
    assert result["tn"] == 2
    assert result["fn"] == 0
    assert result["precision"] == pytest.approx(1/3)
    assert result["recall"] == 1.0

def test_no_malicious_clients():
    """Baseline scenario with no attack — vacuous case returns NaN."""
    import math
    m = AnomalyMetrics()
    malicious = set()
    excluded = set()
    all_clients = set(range(5))
    result = m.compute_round(malicious, excluded, all_clients)
    assert result["tp"] == 0
    assert result["fp"] == 0
    assert result["tn"] == 5
    assert result["fn"] == 0
    assert math.isnan(result["precision"])
    assert math.isnan(result["recall"])
    assert math.isnan(result["f1"])

def test_accumulate_rounds():
    """Test multi-round accumulation."""
    m = AnomalyMetrics()
    malicious = {0, 1}
    all_clients = set(range(5))
    m.compute_round(malicious, {0, 1}, all_clients)
    m.compute_round(malicious, {0}, all_clients)
    m.compute_round(malicious, {0, 1, 2}, all_clients)
    summary = m.summary()
    assert summary["total_rounds"] == 3
    assert summary["cumulative_tp"] == 5
    assert summary["cumulative_fp"] == 1
    assert summary["cumulative_fn"] == 1
    assert len(m.rounds) == 3
