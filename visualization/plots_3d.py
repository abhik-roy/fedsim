"""Plotly visualizations for the FEDSIM Analysis tab."""

import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA

from visualization import (
    COLOR_BENIGN,
    STRATEGY_COLORS,
    THEME,
)

COLOR_MALICIOUS = "#e74c3c"

# Fallback ordered color list for index-based lookups
_STRATEGY_COLOR_LIST = list(STRATEGY_COLORS.values())


def _strategy_color(name: str, index: int) -> str:
    """Return the canonical color for a strategy, falling back to index."""
    key = name.lower().replace(" ", "_")
    return STRATEGY_COLORS.get(key, _STRATEGY_COLOR_LIST[index % len(_STRATEGY_COLOR_LIST)])


def plot_accuracy_surface(results, num_rounds: int, metric_name: str = "Accuracy") -> go.Figure:
    """2D line chart comparing strategy accuracy (or primary metric) over rounds.

    Args:
        results: list of SimulationResult objects (need .strategy_name, .round_accuracies)
        num_rounds: total number of rounds
        metric_name: label for the y-axis metric (e.g. "Accuracy", "MAE", "Perplexity")
    """
    fig = go.Figure()

    for i, r in enumerate(results):
        # Use custom metric values instead of round_accuracies for non-classification
        y_values = r.round_accuracies
        if metric_name != "Accuracy" and r.custom_metrics:
            for mk, vals in r.custom_metrics.items():
                if mk.startswith("eval/") and mk != "eval/loss" and isinstance(vals, list):
                    y_values = vals
                    break
        rounds = list(range(len(y_values)))
        fig.add_trace(go.Scatter(
            x=rounds,
            y=y_values,
            mode="lines+markers",
            name=r.strategy_name,
            line=dict(color=_strategy_color(r.strategy_name, i), width=2),
            marker=dict(size=5),
        ))

    layout_kwargs = dict(
        title=f"Strategy {metric_name} Comparison",
        xaxis_title="Round",
        yaxis_title=metric_name,
        template=THEME,
        height=420,
        margin=dict(t=40, b=60, l=60, r=20),
    )
    if metric_name == "Accuracy":
        layout_kwargs["yaxis"] = dict(range=[0, 1.05])
    fig.update_layout(**layout_kwargs)
    return fig


def plot_trust_reputation_landscape(
    score_history: dict[int, list[float]],
    malicious_clients: set[int],
    num_rounds: int,
    score_type: str = "Trust",
) -> go.Figure:
    """2D heatmap of trust/reputation scores (client x round).

    Args:
        score_history: {client_id: [score_per_round]}
        malicious_clients: set of malicious client IDs
        num_rounds: total rounds
        score_type: "Trust" or "Reputation" (for labeling)
    """
    if not score_history:
        fig = go.Figure()
        fig.add_annotation(text="No data available", xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False, font=dict(size=16))
        fig.update_layout(height=500, template=THEME)
        return fig

    cids = sorted(score_history.keys())
    max_rounds = max(len(scores) for scores in score_history.values())

    # Build Z matrix: rows=clients, cols=rounds
    z_matrix = np.zeros((len(cids), max_rounds))
    for i, cid in enumerate(cids):
        scores = score_history[cid]
        z_matrix[i, :len(scores)] = scores

    client_labels = [
        f"C{cid}*" if cid in malicious_clients else f"C{cid}"
        for cid in cids
    ]

    colorscale = [
        [0.0, "#e74c3c"],
        [0.3, "#f39c12"],
        [0.7, "#f1c40f"],
        [1.0, "#2ecc71"],
    ]

    fig = go.Figure(data=[go.Heatmap(
        x=list(range(1, max_rounds + 1)),
        y=client_labels,
        z=z_matrix,
        colorscale=colorscale,
        colorbar=dict(title=score_type),
        zmin=0,
        zmax=1,
    )])

    fig.update_layout(
        title=f"{score_type} Scores by Client \u00d7 Round",
        xaxis_title="Round",
        yaxis_title="Client",
        template=THEME,
        height=420,
        margin=dict(t=40, b=60, l=60, r=20),
    )
    return fig


def plot_attack_impact(benchmark_results: dict, strategy_names: list[str],
                       attack_names: list[str]) -> go.Figure:
    """2D annotated heatmap of attack impact (attack x strategy).

    Args:
        benchmark_results: {(attack_key, strategy_key): final_accuracy}
        strategy_names: list of strategy display names
        attack_names: list of attack display names
    """
    if not benchmark_results:
        fig = go.Figure()
        fig.add_annotation(
            text="Click 'Run Full Benchmark' to populate",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=16, color="#888"),
        )
        fig.update_layout(height=420, template=THEME)
        return fig

    # Build Z matrix: rows=attacks, cols=strategies
    z_matrix = np.zeros((len(attack_names), len(strategy_names)))
    for i, atk in enumerate(attack_names):
        for j, strat in enumerate(strategy_names):
            z_matrix[i, j] = benchmark_results.get((atk, strat), 0.0)

    text_matrix = [[f"{z_matrix[i][j]:.3f}" for j in range(len(strategy_names))]
                   for i in range(len(attack_names))]

    fig = go.Figure(data=[go.Heatmap(
        x=strategy_names,
        y=attack_names,
        z=z_matrix,
        text=text_matrix,
        texttemplate="%{text}",
        colorscale="RdYlGn",
        zmin=0, zmax=1,
        colorbar=dict(title="Accuracy"),
    )])

    fig.update_layout(
        title="Attack Impact Matrix",
        xaxis_title="Strategy",
        yaxis_title="Attack",
        template=THEME,
        height=420,
        margin=dict(t=40, b=60, l=60, r=20),
    )
    return fig


def plot_client_pca(
    client_params: dict[int, list[np.ndarray]],
    malicious_clients: set[int],
    reputation_scores: dict[int, float] | None = None,
) -> go.Figure:
    """2D PCA scatter of client parameter vectors.

    Args:
        client_params: {client_id: [param_arrays]} — final round parameters
        malicious_clients: set of malicious client IDs
        reputation_scores: {client_id: reputation} for sizing dots
    """
    if not client_params or len(client_params) < 2:
        fig = go.Figure()
        fig.add_annotation(text="Need at least 2 clients for PCA",
                          xref="paper", yref="paper", x=0.5, y=0.5,
                          showarrow=False, font=dict(size=16))
        fig.update_layout(height=420, template=THEME)
        return fig

    cids = sorted(client_params.keys())

    # Flatten parameters
    flat_params = []
    for cid in cids:
        flat_params.append(np.concatenate([p.flatten() for p in client_params[cid]]))
    flat_matrix = np.array(flat_params)

    # PCA to 2 components
    n_components = min(2, len(cids), flat_matrix.shape[1])
    pca = PCA(n_components=n_components)
    projected = pca.fit_transform(flat_matrix)

    # Pad to 2D if needed
    while projected.shape[1] < 2:
        projected = np.column_stack([projected, np.zeros(len(cids))])

    # Split clients into benign and malicious groups
    benign_idx = [i for i, cid in enumerate(cids) if cid not in malicious_clients]
    malicious_idx = [i for i, cid in enumerate(cids) if cid in malicious_clients]

    if reputation_scores:
        sizes = [max(8, reputation_scores.get(cid, 0.5) * 25) for cid in cids]
    else:
        sizes = [12] * len(cids)

    fig = go.Figure()

    # Benign trace
    if benign_idx:
        fig.add_trace(go.Scatter(
            x=projected[benign_idx, 0],
            y=projected[benign_idx, 1],
            mode="markers+text",
            name="Benign",
            text=[f"C{cids[i]}" for i in benign_idx],
            textposition="top center",
            marker=dict(
                size=[sizes[i] for i in benign_idx],
                color=COLOR_BENIGN,
                opacity=0.85,
                line=dict(width=1, color="#333"),
            ),
            hovertext=[f"Client {cids[i]}" for i in benign_idx],
            hoverinfo="text",
        ))

    # Malicious trace
    if malicious_idx:
        fig.add_trace(go.Scatter(
            x=projected[malicious_idx, 0],
            y=projected[malicious_idx, 1],
            mode="markers+text",
            name="Malicious",
            text=[f"C{cids[i]}" for i in malicious_idx],
            textposition="top center",
            marker=dict(
                size=[sizes[i] for i in malicious_idx],
                color=COLOR_MALICIOUS,
                opacity=0.85,
                line=dict(width=1, color="#333"),
            ),
            hovertext=[f"Client {cids[i]} (malicious)" for i in malicious_idx],
            hoverinfo="text",
        ))

    explained = pca.explained_variance_ratio_
    fig.update_layout(
        title="Client Parameter PCA",
        xaxis_title=f"PC1 ({explained[0]:.1%})" if len(explained) > 0 else "PC1",
        yaxis_title=f"PC2 ({explained[1]:.1%})" if len(explained) > 1 else "PC2",
        template=THEME,
        height=420,
        margin=dict(t=40, b=60, l=60, r=20),
    )
    return fig
