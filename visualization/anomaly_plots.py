"""Anomaly detection visualization functions for FEDSIM dashboard."""
import numpy as np
import plotly.graph_objects as go

from visualization import (
    COLOR_BENIGN,
    COLOR_ATTACKED,
    COLOR_TEXT,
    COLOR_TEXT_MUTED,
    COLOR_BG_DARK,
    COLOR_BG_SURFACE,
    COLOR_BORDER,
    fedsim_layout_defaults,
)

# Muted tones for anomaly detection metrics
_COLOR_F1 = "#7FB5A0"       # sage/mint (F1)
_COLOR_PRECISION = "#6A9FD4" # slate blue (Precision)
_COLOR_RECALL = "#D4726A"    # muted coral (Recall)

# Confusion matrix colors (muted)
_COLOR_TN = "#3A3F4D"       # dark blue-gray
_COLOR_TP = "#5A9E87"       # deeper mint
_COLOR_FP = "#D4A76A"       # warm sand
_COLOR_FN = "#D4726A"       # muted coral

# Threshold / reference line
_COLOR_THRESHOLD = "#D4A76A"  # warm sand


def plot_removal_f1_over_rounds(anomaly_history, num_rounds):
    """Line chart: Precision, Recall, F1 per round with improved styling."""
    rounds = list(range(1, len(anomaly_history) + 1))
    precisions = [r["precision"] for r in anomaly_history]
    recalls = [r["recall"] for r in anomaly_history]
    f1s = [r["f1"] for r in anomaly_history]

    fig = go.Figure()

    # F1 with area fill (plotted first so it's behind)
    fig.add_trace(go.Scatter(
        x=rounds, y=f1s, mode="lines+markers",
        name="F1 Score", line=dict(color=_COLOR_F1, width=3),
        marker=dict(size=8, symbol="star"),
        fill="tozeroy", fillcolor="rgba(127, 181, 160, 0.1)",
    ))
    fig.add_trace(go.Scatter(
        x=rounds, y=precisions, mode="lines+markers",
        name="Precision", line=dict(color=_COLOR_PRECISION, width=2),
        marker=dict(size=7, symbol="circle"),
    ))
    fig.add_trace(go.Scatter(
        x=rounds, y=recalls, mode="lines+markers",
        name="Recall", line=dict(color=_COLOR_RECALL, width=2),
        marker=dict(size=7, symbol="diamond"),
    ))

    # Reference line at 0.5
    fig.add_hline(y=0.5, line_dash="dot", line_color=COLOR_BORDER,
                  annotation_text="Chance", annotation_position="bottom right",
                  annotation_font_color=COLOR_TEXT_MUTED)

    fig.update_layout(
        **fedsim_layout_defaults(),
        title="Anomaly Removal Performance",
        xaxis_title="Round", yaxis_title="Score",
        template="plotly_dark",
        height=350,
        margin=dict(t=40, b=50, l=60, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(range=[0, 1.05])
    return fig


def plot_exclusion_timeline(anomaly_history, num_clients, num_rounds, malicious_clients):
    """Heatmap: clients x rounds with TP/FP/TN/FN color coding and legend."""
    z = np.full((num_clients, len(anomaly_history)), 0, dtype=int)
    text = np.full((num_clients, len(anomaly_history)), "", dtype=object)

    for rnd_idx, rdata in enumerate(anomaly_history):
        excluded = set(rdata.get("excluded", []))
        for cid in range(num_clients):
            is_mal = cid in malicious_clients
            is_excl = cid in excluded
            if is_mal and is_excl:
                z[cid, rnd_idx] = 1; text[cid, rnd_idx] = "TP"
            elif not is_mal and is_excl:
                z[cid, rnd_idx] = 2; text[cid, rnd_idx] = "FP"
            elif is_mal and not is_excl:
                z[cid, rnd_idx] = 3; text[cid, rnd_idx] = "FN"
            else:
                z[cid, rnd_idx] = 0; text[cid, rnd_idx] = ""

    colorscale = [
        [0.0, _COLOR_TN], [0.25, _COLOR_TN],    # TN dark blue-gray
        [0.25, _COLOR_TP], [0.50, _COLOR_TP],    # TP deeper mint
        [0.50, _COLOR_FP], [0.75, _COLOR_FP],    # FP warm sand
        [0.75, _COLOR_FN], [1.0, _COLOR_FN],     # FN muted coral
    ]
    z_normalized = z / 3.0

    fig = go.Figure(data=go.Heatmap(
        z=z_normalized, colorscale=colorscale, zmin=0, zmax=1, showscale=False,
        x=[f"R{r+1}" for r in range(len(anomaly_history))],
        y=[f"C{c}{'*' if c in malicious_clients else ''}" for c in range(num_clients)],
        text=text.tolist(),
        texttemplate="%{text}",
        textfont=dict(size=10, color=COLOR_TEXT),
        xgap=2, ygap=2,
        hovertemplate="Client: %{y}<br>Round: %{x}<br>%{text}<extra></extra>",
    ))

    fig.update_layout(
        **fedsim_layout_defaults(),
        title="Client Exclusion Timeline",
        xaxis_title="Round", yaxis_title="Client",
        yaxis=dict(autorange="reversed", gridcolor="rgba(45,49,64,0.6)",
                   title_font=dict(color=COLOR_TEXT_MUTED),
                   tickfont=dict(color=COLOR_TEXT_MUTED)),
        template="plotly_dark",
        height=max(250, 40 * num_clients + 100),
        margin=dict(t=50, b=80, l=70, r=20),
        # Color legend as annotations
        annotations=[
            dict(text="<b>TN</b> Benign+Included", x=0.0, y=-0.15, xref="paper", yref="paper",
                 showarrow=False, font=dict(color=_COLOR_TN, size=11)),
            dict(text="<b>TP</b> Malicious+Excluded", x=0.3, y=-0.15, xref="paper", yref="paper",
                 showarrow=False, font=dict(color=_COLOR_TP, size=11)),
            dict(text="<b>FP</b> Benign+Excluded", x=0.6, y=-0.15, xref="paper", yref="paper",
                 showarrow=False, font=dict(color=_COLOR_FP, size=11)),
            dict(text="<b>FN</b> Malicious+Included", x=0.9, y=-0.15, xref="paper", yref="paper",
                 showarrow=False, font=dict(color=_COLOR_FN, size=11)),
        ],
    )
    return fig


def plot_confusion_summary(anomaly_summary):
    """Professional 2x2 confusion matrix with color-coded cells."""
    tp = anomaly_summary.get("cumulative_tp", 0)
    fp = anomaly_summary.get("cumulative_fp", 0)
    tn = anomaly_summary.get("cumulative_tn", 0)
    fn = anomaly_summary.get("cumulative_fn", 0)
    total = tp + fp + tn + fn or 1

    # Color: TP=mint, TN=mint, FP=sand, FN=coral
    # Use a custom z for coloring: 1=correct (TP/TN), 0=error (FP/FN)
    # Rows ordered bottom-to-top (Plotly default): row 0=Benign (bottom), row 1=Malicious (top)
    z_color = [[0, 1], [1, 0]]

    text = [
        [f"<b>FP</b><br>{fp}<br>({fp/total:.0%})", f"<b>TN</b><br>{tn}<br>({tn/total:.0%})"],
        [f"<b>TP</b><br>{tp}<br>({tp/total:.0%})", f"<b>FN</b><br>{fn}<br>({fn/total:.0%})"],
    ]

    fig = go.Figure(data=go.Heatmap(
        z=z_color, text=text, texttemplate="%{text}",
        textfont=dict(size=16, color=COLOR_TEXT),
        x=["Excluded", "Included"], y=["Benign", "Malicious"],
        colorscale=[[0, _COLOR_FN], [0.5, _COLOR_FP], [1, _COLOR_TP]],
        showscale=False, zmin=0, zmax=1,
        xgap=4, ygap=4,
        hoverinfo="skip",
    ))
    fig.update_layout(
        **fedsim_layout_defaults(),
        title="Cumulative Confusion Matrix",
        template="plotly_dark",
        height=350,
        margin=dict(t=50, b=40, l=80, r=20),
        xaxis=dict(side="bottom", gridcolor="rgba(45,49,64,0.6)",
                   title_font=dict(color=COLOR_TEXT_MUTED),
                   tickfont=dict(color=COLOR_TEXT_MUTED)),
    )
    return fig


def plot_client_score_distribution(client_scores, malicious_clients, threshold=None):
    """Bar chart of per-client strategy scores with improved styling."""
    if not client_scores:
        fig = go.Figure()
        fig.add_annotation(text="No client scores available", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False,
                           font=dict(size=14, color=COLOR_TEXT_MUTED))
        fig.update_layout(**fedsim_layout_defaults(), height=300, template="plotly_dark")
        return fig

    cids = sorted(client_scores.keys())
    scores = [client_scores[c] for c in cids]
    colors = [COLOR_ATTACKED if c in malicious_clients else COLOR_BENIGN for c in cids]
    labels = [f"C{c}{'*' if c in malicious_clients else ''}" for c in cids]
    status = ["Malicious" if c in malicious_clients else "Benign" for c in cids]

    fig = go.Figure(data=go.Bar(
        x=labels, y=scores,
        marker=dict(color=colors, opacity=0.85, line=dict(width=1, color=COLOR_BORDER)),
        hovertemplate="Client %{x}<br>Score: %{y:.4f}<br>Status: %{customdata}<extra></extra>",
        customdata=status,
    ))

    if threshold is not None:
        fig.add_hline(y=threshold, line_dash="dash", line_color=_COLOR_THRESHOLD,
                      annotation_text=f"Threshold: {threshold:.2f}",
                      annotation_position="top right",
                      annotation_font_color=_COLOR_THRESHOLD)

    fig.update_layout(
        **fedsim_layout_defaults(),
        title="Client Strategy Scores",
        xaxis_title="Client", yaxis_title="Score",
        template="plotly_dark",
        height=300,
        margin=dict(t=40, b=50, l=60, r=20),
    )
    return fig
