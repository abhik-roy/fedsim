import plotly.graph_objects as go
import numpy as np

from visualization import (
    STRATEGY_COLORS,
    COLOR_BENIGN,
    COLOR_ATTACKED,
    COLOR_MALICIOUS_IDLE,
    COLOR_EMPTY,
    COLOR_ACCENT,
    COLOR_TEXT,
    COLOR_TEXT_MUTED,
    COLOR_BG_DARK,
    COLOR_BG_SURFACE,
    fedsim_layout_defaults,
)


STRATEGY_DISPLAY_NAMES = {
    "fedavg": "FedAvg",
    "trimmed_mean": "Trimmed Mean",
    "krum": "Krum",
    "median": "Median",
    "reputation": "Reputation",
    "custom:Reputation": "Reputation",
    "bulyan": "Bulyan",
    "rfa": "RFA",
}


def plot_live_loss(strategy_losses: dict[str, list[float]], num_rounds: int,
                   loss_name: str = "Loss") -> go.Figure:
    """Create a live-updating loss chart."""
    fig = go.Figure()
    for strat_name, losses in strategy_losses.items():
        name = STRATEGY_DISPLAY_NAMES.get(strat_name, strat_name)
        color = STRATEGY_COLORS.get(strat_name, "#888")
        fig.add_trace(go.Scatter(
            x=list(range(len(losses))),
            y=losses,
            mode="lines+markers",
            name=name,
            line=dict(width=2, color=color),
            marker=dict(size=4),
        ))
    fig.update_layout(
        **fedsim_layout_defaults(),
        title=f"{loss_name} (Live)",
        xaxis_title="Round (0 = pre-training)",
        yaxis_title=loss_name,
        template="plotly_dark",
        height=280,
        margin=dict(t=40, b=40, l=50, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(range=[0, num_rounds])
    return fig


def plot_live_accuracy(strategy_accuracies: dict[str, list[float]], num_rounds: int,
                       metric_name: str = "Accuracy") -> go.Figure:
    """Create a live-updating accuracy/primary-metric chart."""
    fig = go.Figure()
    for strat_name, accs in strategy_accuracies.items():
        name = STRATEGY_DISPLAY_NAMES.get(strat_name, strat_name)
        color = STRATEGY_COLORS.get(strat_name, "#888")
        fig.add_trace(go.Scatter(
            x=list(range(len(accs))),
            y=accs,
            mode="lines+markers",
            name=name,
            line=dict(width=2, color=color),
            marker=dict(size=4),
        ))
    fig.update_layout(
        **fedsim_layout_defaults(),
        title=f"{metric_name} (Live)",
        xaxis_title="Round (0 = pre-training)",
        yaxis_title=metric_name,
        template="plotly_dark",
        height=280,
        margin=dict(t=40, b=40, l=50, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(range=[0, num_rounds])
    if metric_name == "Accuracy":
        fig.update_yaxes(range=[0, 1.05])
    return fig


def plot_custom_metric(results, metric_key, chart_type="line"):
    """Plot a custom metric across strategies."""
    display_name = metric_key.split("/")[-1] if "/" in metric_key else metric_key
    fig = go.Figure()

    if chart_type == "bar":
        for r in results:
            vals = (r.custom_metrics or {}).get(metric_key, [])
            if vals:
                sname = STRATEGY_DISPLAY_NAMES.get(r.strategy_name, r.strategy_name)
                fig.add_trace(go.Bar(name=sname, x=[sname], y=[vals[-1]],
                    marker_color=STRATEGY_COLORS.get(r.strategy_name, "#888")))
    else:
        for r in results:
            vals = (r.custom_metrics or {}).get(metric_key, [])
            if vals:
                fig.add_trace(go.Scatter(
                    x=list(range(len(vals))), y=vals, mode="lines+markers",
                    name=STRATEGY_DISPLAY_NAMES.get(r.strategy_name, r.strategy_name),
                    line=dict(color=STRATEGY_COLORS.get(r.strategy_name, "#888"))))

    fig.update_layout(
        **fedsim_layout_defaults(),
        title=display_name.replace("_", " ").title(),
        xaxis_title="Round (0 = pre-training)", yaxis_title=display_name,
        template="plotly_dark", margin=dict(t=40, b=60, l=60, r=20),
        legend=dict(orientation="h", y=-0.15))
    return fig


def plot_client_sparklines(trust_history, reputation_history, malicious_ids, num_rounds):
    """Create a grid of per-client trust/reputation sparklines.

    Args:
        trust_history: dict[int, list[float]] — per-client trust values over rounds
        reputation_history: dict[int, list[float]] — per-client reputation values over rounds
        malicious_ids: set of malicious client IDs
        num_rounds: total rounds in simulation

    Returns:
        go.Figure with one row per client, two sparkline columns (trust + reputation)
    """
    from plotly.subplots import make_subplots

    client_ids = sorted(trust_history.keys())
    n_clients = len(client_ids)

    if n_clients == 0:
        fig = go.Figure()
        fig.update_layout(
            **fedsim_layout_defaults(),
            template="plotly_dark", height=100,
            annotations=[dict(text="No client data", showarrow=False,
                              xref="paper", yref="paper", x=0.5, y=0.5,
                              font=dict(color=COLOR_TEXT_MUTED))])
        return fig

    has_reputation = bool(reputation_history)
    n_cols = 2 if has_reputation else 1
    col_titles = ["Trust"] + (["Reputation"] if has_reputation else [])

    fig = make_subplots(
        rows=n_clients, cols=n_cols,
        shared_xaxes=True,
        vertical_spacing=max(0.01, 0.4 / n_clients),
        horizontal_spacing=0.05,
        subplot_titles=col_titles if n_clients == 1 else None,
    )

    rounds_x = list(range(1, num_rounds + 1))

    for row_idx, cid in enumerate(client_ids, start=1):
        is_mal = cid in malicious_ids
        color = COLOR_ATTACKED if is_mal else COLOR_BENIGN
        label = f"C{cid}{'*' if is_mal else ''}"

        # Trust sparkline
        trust_vals = trust_history.get(cid, [])
        fig.add_trace(go.Scatter(
            x=rounds_x[:len(trust_vals)], y=trust_vals,
            mode="lines", name=label, showlegend=(row_idx == 1),
            line=dict(width=1.5, color=color),
            hovertemplate=f"{label} Trust R%{{x}}: %{{y:.3f}}<extra></extra>",
        ), row=row_idx, col=1)

        # Reputation sparkline
        if has_reputation:
            rep_vals = reputation_history.get(cid, [])
            fig.add_trace(go.Scatter(
                x=rounds_x[:len(rep_vals)], y=rep_vals,
                mode="lines", name=label, showlegend=False,
                line=dict(width=1.5, color=color),
                hovertemplate=f"{label} Rep R%{{x}}: %{{y:.3f}}<extra></extra>",
            ), row=row_idx, col=2)

        # Add client label as y-axis title
        fig.update_yaxes(title_text=label, title_font=dict(size=9, color=COLOR_TEXT_MUTED),
                         row=row_idx, col=1,
                         range=[0, 1.05], showticklabels=False, title_standoff=2)
        if has_reputation:
            fig.update_yaxes(range=[0, 1.05], showticklabels=False, row=row_idx, col=2)

    # Column headers
    fig.add_annotation(text="<b>Trust</b>", x=0.25 if has_reputation else 0.5, y=1.02,
                       xref="paper", yref="paper", showarrow=False,
                       font=dict(size=12, color=COLOR_TEXT))
    if has_reputation:
        fig.add_annotation(text="<b>Reputation</b>", x=0.75, y=1.02,
                           xref="paper", yref="paper", showarrow=False,
                           font=dict(size=12, color=COLOR_TEXT))

    fig.update_xaxes(showticklabels=False)
    # Show x-axis labels only on the bottom row
    for col in range(1, n_cols + 1):
        fig.update_xaxes(showticklabels=True, title_text="Round",
                         title_font=dict(size=10, color=COLOR_TEXT_MUTED), row=n_clients, col=col)

    row_height = max(35, min(55, 400 // n_clients))
    fig.update_layout(
        **fedsim_layout_defaults(),
        template="plotly_dark",
        height=max(200, row_height * n_clients + 80),
        margin=dict(t=40, b=40, l=60, r=20),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1,
                    font=dict(size=9, color=COLOR_TEXT)),
    )
    return fig


def plot_client_grid(grid_data: list[list[str]], num_clients: int, num_rounds: int,
                     current_round: int) -> go.Figure:
    """Create a clients x rounds heatmap showing attack status.

    grid_data[round_idx] is a list of status strings per client for that round.
    Statuses: "benign", "attacked", "malicious_idle", "empty"
    """
    # Build the numeric matrix and text matrix for the heatmap
    # Rows = clients (bottom to top), Columns = rounds
    # 6 states: benign(0), attacked(1), malicious_idle(2), idle/empty(3), excluded(4), false_positive(5)
    from visualization import COLOR_BORDER
    _COLOR_EXCLUDED = "#D4A76A"      # warm sand — attacked but caught by strategy
    _COLOR_FALSE_POS = "#B088C4"     # muted lavender — benign but wrongly excluded
    status_to_num = {
        "benign": 0, "attacked": 1, "malicious_idle": 2,
        "empty": 3, "idle": 3,
        "excluded": 4, "false_positive": 5,
    }
    status_labels = {
        "benign": "Benign", "attacked": "ATTACKED",
        "malicious_idle": "Malicious (idle)", "idle": "Not selected", "empty": "",
        "excluded": "EXCLUDED", "false_positive": "Excluded (FP)",
    }
    z = []
    text = []

    for cid in range(num_clients):
        row = []
        text_row = []
        for rnd in range(num_rounds + 1):  # round 0 = initial
            if rnd < len(grid_data):
                status = grid_data[rnd][cid] if cid < len(grid_data[rnd]) else "empty"
            else:
                status = "empty"
            row.append(status_to_num.get(status, 3))
            text_row.append(status_labels.get(status, ""))
        z.append(row)
        text.append(text_row)

    colorscale = [
        [0.0, COLOR_BENIGN],           # 0 = benign
        [0.2, COLOR_BENIGN],
        [0.2, COLOR_ATTACKED],         # 1 = attacked (included in aggregation)
        [0.4, COLOR_ATTACKED],
        [0.4, COLOR_MALICIOUS_IDLE],   # 2 = malicious but idle
        [0.6, COLOR_MALICIOUS_IDLE],
        [0.6, COLOR_EMPTY],            # 3 = idle / not selected
        [0.7, COLOR_EMPTY],
        [0.7, _COLOR_EXCLUDED],        # 4 = attacked AND excluded by strategy (TP)
        [0.85, _COLOR_EXCLUDED],
        [0.85, _COLOR_FALSE_POS],      # 5 = benign but excluded (FP)
        [1.0, _COLOR_FALSE_POS],
    ]
    _zmax = 5

    fig = go.Figure(data=go.Heatmap(
        z=z,
        text=text,
        texttemplate="%{text}",
        textfont=dict(size=8, color=COLOR_TEXT),
        x=[f"R{r}" for r in range(num_rounds + 1)],
        y=[f"C{c}" for c in range(num_clients)],
        colorscale=colorscale,
        zmin=0,
        zmax=_zmax,
        showscale=False,
        xgap=2,
        ygap=2,
    ))

    # Add a vertical line at the current round
    if current_round > 0:
        fig.add_vline(
            x=current_round - 0.5, line_width=2, line_dash="dot", line_color=COLOR_ACCENT,
            annotation_text=f"Round {current_round}", annotation_position="top",
        )

    fig.update_layout(
        **fedsim_layout_defaults(),
        title="Client Activity Grid",
        xaxis_title="Round",
        yaxis_title="Client",
        template="plotly_dark",
        height=max(200, 35 * num_clients + 80),
        margin=dict(t=50, b=40, l=70, r=20),
    )
    return fig
