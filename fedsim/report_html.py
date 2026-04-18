"""Generate self-contained interactive HTML reports from FEDSIM simulation results."""
import html as _html_mod
import json
import dataclasses
from datetime import datetime

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio


def _serialize(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, set):
        return sorted(obj)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    if isinstance(obj, float) and (obj != obj or obj == float('inf') or obj == float('-inf')):
        return None
    raise TypeError(f"Not serializable: {type(obj)}")


def generate_html_report(config, results, title=None):
    """Generate a self-contained HTML report with embedded Plotly charts.

    Args:
        config: SimulationConfig dataclass
        results: list of SimulationResult objects
        title: Optional report title

    Returns:
        HTML string
    """
    if title is None:
        title = f"FEDSIM Report — {config.model_name} | {config.dataset_name}"

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Build charts
    charts_html = []

    # 1. Loss chart
    loss_fig = go.Figure()
    for r in results:
        loss_fig.add_trace(go.Scatter(
            x=list(range(len(r.round_losses))),
            y=r.round_losses,
            mode="lines+markers",
            name=r.strategy_name,
        ))
    loss_fig.update_layout(
        title="Aggregated Loss per Round",
        xaxis_title="Round", yaxis_title="Loss",
        template="plotly_dark", height=400,
    )
    charts_html.append(pio.to_html(loss_fig, full_html=False, include_plotlyjs=False))

    # 2. Accuracy/Metric chart
    acc_fig = go.Figure()
    for r in results:
        acc_fig.add_trace(go.Scatter(
            x=list(range(len(r.round_accuracies))),
            y=r.round_accuracies,
            mode="lines+markers",
            name=r.strategy_name,
        ))
    acc_fig.update_layout(
        title="Model Performance per Round",
        xaxis_title="Round", yaxis_title="Metric",
        template="plotly_dark", height=400,
    )
    charts_html.append(pio.to_html(acc_fig, full_html=False, include_plotlyjs=False))

    # 3. Custom metrics charts
    all_custom_keys = set()
    for r in results:
        all_custom_keys.update((r.custom_metrics or {}).keys())

    for mk in sorted(all_custom_keys):
        metric_fig = go.Figure()
        for r in results:
            vals = (r.custom_metrics or {}).get(mk, [])
            if vals:
                metric_fig.add_trace(go.Scatter(
                    x=list(range(len(vals))), y=vals,
                    mode="lines+markers", name=r.strategy_name,
                ))
        display_name = mk.split("/")[-1] if "/" in mk else mk
        metric_fig.update_layout(
            title=display_name.replace("_", " ").title(),
            xaxis_title="Round", yaxis_title=display_name,
            template="plotly_dark", height=350,
        )
        charts_html.append(pio.to_html(metric_fig, full_html=False, include_plotlyjs=False))

    # 4. Anomaly summary chart (if any strategy has anomaly data)
    for r in results:
        if r.anomaly_history:
            anomaly_fig = go.Figure()
            rounds = list(range(1, len(r.anomaly_history) + 1))
            anomaly_fig.add_trace(go.Scatter(
                x=rounds, y=[h["f1"] for h in r.anomaly_history],
                mode="lines+markers", name="F1",
                line=dict(color="#7FB5A0", width=3),
            ))
            anomaly_fig.add_trace(go.Scatter(
                x=rounds, y=[h["precision"] for h in r.anomaly_history],
                mode="lines+markers", name="Precision",
                line=dict(color="#6A9FD4"),
            ))
            anomaly_fig.add_trace(go.Scatter(
                x=rounds, y=[h["recall"] for h in r.anomaly_history],
                mode="lines+markers", name="Recall",
                line=dict(color="#D4726A"),
            ))
            anomaly_fig.update_layout(
                title=f"Anomaly Detection — {_html_mod.escape(r.strategy_name)}",
                xaxis_title="Round", yaxis_title="Score",
                yaxis=dict(range=[0, 1.05]),
                template="plotly_dark", height=350,
            )
            charts_html.append(pio.to_html(anomaly_fig, full_html=False, include_plotlyjs=False))

    # Build config table
    config_dict = dataclasses.asdict(config)
    config_rows = ""
    for k, v in config_dict.items():
        if k in ("plugin_params", "lr_scheduler_params") and not v:
            continue
        val_str = json.dumps(v, default=_serialize) if isinstance(v, (dict, list)) else str(v)
        config_rows += f"<tr><td>{_html_mod.escape(str(k))}</td><td>{_html_mod.escape(val_str)}</td></tr>\n"

    # Build results summary table
    import math as _math_mod

    def _fmt(v):
        if isinstance(v, (int, float)) and _math_mod.isfinite(v):
            return f"{v:.4f}"
        return "N/A"

    results_rows = ""
    for r in results:
        final_loss = r.round_losses[-1] if r.round_losses else 0
        final_acc = r.round_accuracies[-1] if r.round_accuracies else 0
        best_acc = max(r.round_accuracies) if r.round_accuracies else 0
        f1 = r.anomaly_summary.get("cumulative_f1", "N/A") if r.anomaly_summary else "N/A"

        results_rows += f"""<tr>
            <td>{_html_mod.escape(r.strategy_name)}</td>
            <td>{_fmt(final_loss)}</td>
            <td>{_fmt(final_acc)}</td>
            <td>{_fmt(best_acc)}</td>
            <td>{f1 if isinstance(f1, str) else _fmt(f1)}</td>
            <td>{r.total_time:.1f}s</td>
        </tr>\n"""

    # Assemble HTML
    charts_section = "\n<hr>\n".join(charts_html)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{_html_mod.escape(title)}</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: #0E1117; color: #FAFAFA; padding: 2rem;
            max-width: 1200px; margin: 0 auto;
        }}
        h1 {{ color: #7FB5A0; margin-bottom: 0.5rem; font-size: 1.8rem; }}
        h2 {{ color: #FAFAFA; margin: 2rem 0 1rem; font-size: 1.3rem; border-bottom: 1px solid #333; padding-bottom: 0.5rem; }}
        .meta {{ color: #888; font-size: 0.85rem; margin-bottom: 2rem; }}
        table {{ width: 100%; border-collapse: collapse; margin: 1rem 0; }}
        th, td {{ padding: 0.6rem 1rem; text-align: left; border-bottom: 1px solid #333; }}
        th {{ background: #1C1F26; color: #7FB5A0; font-weight: 600; }}
        td {{ color: #CCC; font-size: 0.9rem; }}
        tr:hover {{ background: #1C1F26; }}
        .chart {{ margin: 1.5rem 0; }}
        .footer {{ margin-top: 3rem; padding-top: 1rem; border-top: 1px solid #333; color: #666; font-size: 0.8rem; }}
        details {{ margin: 1rem 0; }}
        summary {{ cursor: pointer; color: #7FB5A0; font-weight: 600; }}
    </style>
</head>
<body>
    <h1>{_html_mod.escape(title)}</h1>
    <div class="meta">Generated: {timestamp} | Seed: {config.seed}</div>

    <h2>Results Summary</h2>
    <table>
        <thead>
            <tr><th>Strategy</th><th>Final Loss</th><th>Final Metric</th><th>Best Metric</th><th>Removal F1</th><th>Time</th></tr>
        </thead>
        <tbody>
            {results_rows}
        </tbody>
    </table>

    <h2>Charts</h2>
    <div class="chart">
        {charts_section}
    </div>

    <h2>Configuration</h2>
    <details>
        <summary>Full Configuration</summary>
        <table>
            <thead><tr><th>Parameter</th><th>Value</th></tr></thead>
            <tbody>{config_rows}</tbody>
        </table>
    </details>

    <div class="footer">
        Generated by FEDSIM — Federated Learning Simulation Framework
    </div>
</body>
</html>"""

    return html
