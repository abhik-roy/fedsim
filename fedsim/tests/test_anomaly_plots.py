import plotly.graph_objects as go
from visualization.anomaly_plots import (
    plot_removal_f1_over_rounds,
    plot_exclusion_timeline,
    plot_confusion_summary,
    plot_client_score_distribution,
)

def test_removal_f1_plot_returns_figure():
    anomaly_history = [
        {"precision": 1.0, "recall": 0.5, "f1": 0.67, "tp": 1, "fp": 0, "tn": 3, "fn": 1},
        {"precision": 0.5, "recall": 1.0, "f1": 0.67, "tp": 2, "fp": 2, "tn": 1, "fn": 0},
    ]
    fig = plot_removal_f1_over_rounds(anomaly_history, num_rounds=2)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 3  # Precision, Recall, F1 traces

def test_exclusion_timeline_returns_figure():
    anomaly_history = [
        {"excluded": [0, 1], "malicious": [0, 1]},
        {"excluded": [0], "malicious": [0, 1]},
    ]
    fig = plot_exclusion_timeline(anomaly_history, num_clients=5, num_rounds=2, malicious_clients={0, 1})
    assert isinstance(fig, go.Figure)

def test_confusion_summary_returns_figure():
    summary = {"cumulative_tp": 10, "cumulative_fp": 2, "cumulative_tn": 30, "cumulative_fn": 3}
    fig = plot_confusion_summary(summary)
    assert isinstance(fig, go.Figure)
    # The z matrix should be 2x2 with non-negative values
    z = fig.data[0].z
    assert len(z) == 2 and len(z[0]) == 2
    for row in z:
        for val in row:
            assert val >= 0

def test_score_distribution_returns_figure():
    client_scores = {0: 0.2, 1: 0.9, 2: 0.85, 3: 0.15, 4: 0.92}
    fig = plot_client_score_distribution(client_scores, malicious_clients={0, 3})
    assert isinstance(fig, go.Figure)
