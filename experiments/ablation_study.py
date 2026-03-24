#!/usr/bin/env python3
"""Ablation Study: Finding Conditions That Differentiate FedAvg vs Reputation vs ReputationV2.

Systematically varies:
  - Data distribution: IID vs Non-IID (alpha=0.5)
  - Attack type: None, Label Flipping, Gradient Scaling (scale=5)
  - Malicious fraction: 28.6% (4/14) vs 42.8% (6/14)

Generates a deep-dive HTML report with per-config analysis, cross-config
comparison heatmaps, and identification of conditions that best separate
the three strategies.

Estimated runtime: ~3 hours (20 strategy runs × ~10 min each)
"""
import os
import sys
import time
import json
import traceback
from dataclasses import dataclass
from datetime import datetime

_FEDSIM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _FEDSIM_ROOT not in sys.path:
    sys.path.insert(0, _FEDSIM_ROOT)

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from simulation.runner import (
    SimulationConfig, AttackConfig, SimulationResult, RoundEvent, run_simulation,
)
from visualization import (
    STRATEGY_COLORS, COLOR_TEXT, COLOR_TEXT_MUTED, COLOR_BG_DARK,
    COLOR_BG_SURFACE, COLOR_BORDER, COLOR_BENIGN, COLOR_ATTACKED,
    fedsim_layout_defaults,
)
from visualization.plots import (
    plot_live_loss, plot_live_accuracy, plot_client_grid,
    STRATEGY_DISPLAY_NAMES,
)
from visualization.anomaly_plots import plot_exclusion_timeline

# Register V2
STRATEGY_DISPLAY_NAMES["custom:ReputationV2"] = "ReputationV2"
STRATEGY_COLORS["custom:ReputationV2"] = "#6A9FD4"

# ── Constants ─────────────────────────────────────────────────────────
NUM_CLIENTS = 14
NUM_ROUNDS = 10
LOCAL_EPOCHS = 2
LEARNING_RATE = 0.011
WARMUP_ROUNDS = 3
SEED = 42
BATCH_SIZE = 32

ALL_STRATEGIES = ["fedavg", "custom:Reputation", "custom:ReputationV2"]
BASELINE_STRATEGIES = ["fedavg"]  # for no-attack baselines

OUTPUT_DIR = os.path.join(_FEDSIM_ROOT, "experiments", "results")
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "ablation_checkpoint.json")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SLABEL = {
    "fedavg": "FedAvg",
    "custom:Reputation": "Rep V1",
    "custom:ReputationV2": "Rep V2",
}


# ── Experiment Configs ────────────────────────────────────────────────

@dataclass
class RunConfig:
    name: str
    partition_type: str
    alpha: float
    attack_type: str
    attack_params: dict
    malicious_fraction: float
    strategies: list[str]
    description: str


CONFIGS = [
    # ── Clean baselines (FedAvg only — ceiling for each data condition)
    RunConfig("IID_clean", "iid", 0.5,
              "none", {}, 0.0, BASELINE_STRATEGIES,
              "IID, no attack (ceiling)"),
    RunConfig("NonIID_clean", "non_iid", 0.5,
              "none", {}, 0.0, BASELINE_STRATEGIES,
              "Non-IID alpha=0.5, no attack (ceiling)"),

    # ── Label Flipping: IID vs Non-IID × fraction
    RunConfig("IID_LF_29", "iid", 0.5,
              "label_flipping", {}, 4/14, ALL_STRATEGIES,
              "IID, label flip, 4/14 malicious"),
    RunConfig("NonIID_LF_29", "non_iid", 0.5,
              "label_flipping", {}, 4/14, ALL_STRATEGIES,
              "Non-IID alpha=0.5, label flip, 4/14 malicious"),
    RunConfig("IID_LF_43", "iid", 0.5,
              "label_flipping", {}, 6/14, ALL_STRATEGIES,
              "IID, label flip, 6/14 malicious"),
    RunConfig("NonIID_LF_43", "non_iid", 0.5,
              "label_flipping", {}, 6/14, ALL_STRATEGIES,
              "Non-IID alpha=0.5, label flip, 6/14 malicious"),

    # ── Gradient Scaling: amplified model poisoning
    RunConfig("IID_GS5_29", "iid", 0.5,
              "gradient_scaling", {"scale_factor": 5.0}, 4/14, ALL_STRATEGIES,
              "IID, gradient scaling (5x), 4/14 malicious"),
    RunConfig("NonIID_GS5_29", "non_iid", 0.5,
              "gradient_scaling", {"scale_factor": 5.0}, 4/14, ALL_STRATEGIES,
              "Non-IID alpha=0.5, gradient scaling (5x), 4/14 malicious"),

    # ── Gradient Scaling: higher fraction
    RunConfig("IID_GS5_43", "iid", 0.5,
              "gradient_scaling", {"scale_factor": 5.0}, 6/14, ALL_STRATEGIES,
              "IID, gradient scaling (5x), 6/14 malicious"),
    RunConfig("NonIID_GS5_43", "non_iid", 0.5,
              "gradient_scaling", {"scale_factor": 5.0}, 6/14, ALL_STRATEGIES,
              "Non-IID alpha=0.5, gradient scaling (5x), 6/14 malicious"),
]


# ── Run Management ────────────────────────────────────────────────────

def _build_sim_config(cfg: RunConfig) -> SimulationConfig:
    return SimulationConfig(
        model_name="resnet18", dataset_name="cifar10",
        num_clients=NUM_CLIENTS, num_rounds=NUM_ROUNDS,
        local_epochs=LOCAL_EPOCHS, learning_rate=LEARNING_RATE,
        partition_type=cfg.partition_type, alpha=cfg.alpha,
        val_split=0.1, strategies=cfg.strategies,
        batch_size=BATCH_SIZE, seed=SEED,
        optimizer="sgd", loss_function="cross_entropy",
        attack=AttackConfig(
            attack_type=cfg.attack_type,
            malicious_fraction=cfg.malicious_fraction,
            attack_params=cfg.attack_params,
        ),
        reputation_truth_threshold=0.5,
        reputation_trust_exclusion_threshold=0.15,
        reputation_initial_reputation=0.5,
        reputation_warmup_rounds=WARMUP_ROUNDS,
        reputation_smoothing_beta=0.85,
        fraction_fit=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
    )


def _extract_malicious(result):
    mal = set()
    for rnd_st in result.client_statuses_history:
        for cid, status in rnd_st.items():
            if status in ("attacked", "malicious_idle"):
                mal.add(cid)
    return mal


def _avg_post_warmup(anomaly_hist, metric):
    vals = [r[metric] for i, r in enumerate(anomaly_hist) if i + 1 > WARMUP_ROUNDS]
    return float(np.mean(vals)) if vals else 0.0


# ── Checkpoint helpers ────────────────────────────────────────────────

def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        v = float(obj)
        return None if (np.isnan(v) or np.isinf(v)) else v
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, set):
        return list(obj)
    if hasattr(obj, '__dataclass_fields__'):
        from dataclasses import asdict
        return asdict(obj)
    raise TypeError("Not serializable: {}".format(type(obj)))


def _save_checkpoint(all_results, path):
    from dataclasses import asdict
    data = {}
    for cfg_name, results in all_results.items():
        data[cfg_name] = [asdict(r) for r in results]
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, default=_json_default, indent=1)
    os.replace(tmp, path)


def _load_checkpoint(path):
    if not os.path.exists(path):
        return {}
    from dataclasses import fields
    with open(path) as f:
        data = json.load(f)
    out = {}
    for cfg_name, results_list in data.items():
        sim_results = []
        for r in results_list:
            kwargs = {}
            valid = {f.name for f in fields(SimulationResult)}
            for k, v in r.items():
                if k in valid:
                    kwargs[k] = v
            sim_results.append(SimulationResult(**kwargs))
        out[cfg_name] = sim_results
    return out


# ── Run Experiment ────────────────────────────────────────────────────

def run_all():
    all_results = _load_checkpoint(CHECKPOINT_PATH)
    cached = len(all_results)
    total_configs = len(CONFIGS)
    pending = [c for c in CONFIGS if c.name not in all_results]

    print("=" * 65)
    print("  FEDSIM Ablation Study")
    print("  {} configs total, {} cached, {} to run".format(
        total_configs, cached, len(pending)))
    print("=" * 65)

    for idx, cfg in enumerate(pending):
        n_strats = len(cfg.strategies)
        print("\n[{}/{}] {}: {} ({} strategies)".format(
            cached + idx + 1, total_configs, cfg.name, cfg.description, n_strats))

        sim_config = _build_sim_config(cfg)
        t0 = time.time()

        def _round_cb(event):
            s = SLABEL.get(event.strategy_name, event.strategy_name)
            n_excl = len(event.client_excluded)
            excl_s = " excl={}".format(len(event.client_excluded)) if n_excl else ""
            print("  [{}] R{}/{} acc={:.4f}{}".format(
                s, event.round_num, event.num_rounds, event.accuracy, excl_s))

        try:
            results = run_simulation(sim_config, round_callback=_round_cb)
            elapsed = time.time() - t0
            print("  Done in {:.0f}s".format(elapsed))

            for r in results:
                s = SLABEL.get(r.strategy_name, r.strategy_name)
                acc = r.round_accuracies[-1] if r.round_accuracies else 0
                print("    {}: final_acc={:.4f}".format(s, acc))

            all_results[cfg.name] = results
            _save_checkpoint(all_results, CHECKPOINT_PATH)

        except Exception as e:
            print("  FAILED: {}".format(e))
            traceback.print_exc()
            continue

    return all_results


# ── Report Generation ─────────────────────────────────────────────────

def _fig_html(fig):
    return pio.to_html(fig, full_html=False, include_plotlyjs=False)


def _build_grid_data(result, num_clients, malicious_ids):
    grid = []
    for rnd_idx, rnd_statuses in enumerate(result.client_statuses_history):
        excluded = set()
        if rnd_idx < len(result.anomaly_history):
            excluded = set(result.anomaly_history[rnd_idx].get("excluded", []))
        row = []
        for cid in range(num_clients):
            st = rnd_statuses.get(cid, "idle")
            if st == "attacked" and cid in excluded:
                st = "excluded"
            elif st == "benign" and cid in excluded:
                st = "false_positive"
            row.append(st)
        grid.append(row)
    return grid


def generate_report(all_results, total_elapsed):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── Build summary table data ──────────────────────────────────
    # For each config: final acc per strategy, attack degradation, detection metrics
    summary_rows = []
    for cfg in CONFIGS:
        if cfg.name not in all_results:
            continue
        results = all_results[cfg.name]
        row = {"config": cfg, "strategies": {}}
        for r in results:
            sname = r.strategy_name
            acc = r.round_accuracies[-1] if r.round_accuracies else 0
            peak = max(r.round_accuracies) if r.round_accuracies else 0
            loss = r.round_losses[-1] if r.round_losses else 0
            mal = _extract_malicious(r)

            avg_prec = _avg_post_warmup(r.anomaly_history, "precision") if r.anomaly_history else 0
            avg_rec = _avg_post_warmup(r.anomaly_history, "recall") if r.anomaly_history else 0
            avg_f1 = _avg_post_warmup(r.anomaly_history, "f1") if r.anomaly_history else 0
            cum_tp = r.anomaly_summary.get("cumulative_tp", 0) if r.anomaly_summary else 0
            cum_fp = r.anomaly_summary.get("cumulative_fp", 0) if r.anomaly_summary else 0

            row["strategies"][sname] = {
                "acc": acc, "peak": peak, "loss": loss,
                "avg_prec": avg_prec, "avg_rec": avg_rec, "avg_f1": avg_f1,
                "cum_tp": cum_tp, "cum_fp": cum_fp, "time": r.total_time,
                "malicious": mal,
            }
        summary_rows.append(row)

    # ── Compute ceilings from clean baselines ─────────────────────
    ceilings = {}
    for row in summary_rows:
        cfg = row["config"]
        if cfg.attack_type == "none" and "fedavg" in row["strategies"]:
            key = cfg.partition_type
            ceilings[key] = row["strategies"]["fedavg"]["acc"]

    # ── Summary heatmap data ──────────────────────────────────────
    attack_configs = [r for r in summary_rows if r["config"].attack_type != "none"]

    # ── Build per-config detail sections ──────────────────────────
    config_sections = []
    for row in attack_configs:
        cfg = row["config"]
        results = all_results[cfg.name]
        mal = set()
        for r in results:
            mal |= _extract_malicious(r)

        ceiling = ceilings.get(cfg.partition_type, 0)
        fa_acc = row["strategies"].get("fedavg", {}).get("acc", 0)
        degradation = ceiling - fa_acc if ceiling else 0

        # Loss/accuracy curves
        s_losses = {r.strategy_name: r.round_losses for r in results}
        s_accs = {r.strategy_name: r.round_accuracies for r in results}
        fig_loss = plot_live_loss(s_losses, NUM_ROUNDS)
        fig_loss.update_layout(height=300, title=None)
        fig_acc = plot_live_accuracy(s_accs, NUM_ROUNDS)
        fig_acc.update_layout(height=300, title=None)

        # Client grids for trust strategies
        grid_htmls = []
        for r in results:
            if r.strategy_name != "fedavg":
                g = _build_grid_data(r, NUM_CLIENTS, mal)
                gfig = plot_client_grid(g, NUM_CLIENTS, NUM_ROUNDS, NUM_ROUNDS)
                gfig.update_layout(title=SLABEL.get(r.strategy_name, r.strategy_name),
                                   height=max(200, 30 * NUM_CLIENTS + 60))
                grid_htmls.append(_fig_html(gfig))

        # Exclusion timelines
        excl_htmls = []
        for r in results:
            if r.anomaly_history:
                efig = plot_exclusion_timeline(r.anomaly_history, NUM_CLIENTS, NUM_ROUNDS, mal)
                efig.update_layout(title=SLABEL.get(r.strategy_name, r.strategy_name),
                                   height=max(200, 30 * NUM_CLIENTS + 60))
                excl_htmls.append(_fig_html(efig))

        # Strategy comparison table
        strat_table = '<table class="data-table"><thead><tr><th>Strategy</th>'
        strat_table += '<th>Final Acc</th><th>Peak Acc</th><th>Final Loss</th>'
        strat_table += '<th>Avg Prec</th><th>Avg Recall</th><th>Avg F1</th>'
        strat_table += '<th>Cum TP</th><th>Cum FP</th><th>Time</th></tr></thead><tbody>'
        for sname in ALL_STRATEGIES:
            if sname not in row["strategies"]:
                continue
            d = row["strategies"][sname]
            strat_table += '<tr><td>{}</td><td>{:.4f}</td><td>{:.4f}</td><td>{:.4f}</td>'.format(
                SLABEL.get(sname, sname), d["acc"], d["peak"], d["loss"])
            strat_table += '<td>{:.2f}</td><td>{:.2f}</td><td>{:.2f}</td>'.format(
                d["avg_prec"], d["avg_rec"], d["avg_f1"])
            strat_table += '<td>{}</td><td>{}</td><td>{:.0f}s</td></tr>'.format(
                d["cum_tp"], d["cum_fp"], d["time"])
        strat_table += '</tbody></table>'

        # V2 vs V1 accuracy delta
        v1_acc = row["strategies"].get("custom:Reputation", {}).get("acc", 0)
        v2_acc = row["strategies"].get("custom:ReputationV2", {}).get("acc", 0)
        v2v1_delta = v2_acc - v1_acc
        v2_fa_delta = v2_acc - fa_acc

        section = """
        <div class="config-section" id="{id}">
          <h3>{name}: {desc}</h3>
          <div class="config-meta">
            Clean ceiling: {ceiling:.2%} | Attack degradation: {degrad:.2%} |
            Malicious clients: {mal} |
            V2 vs FedAvg: <b>{v2fa:+.2%}</b> |
            V2 vs V1: <b style="color:{v2v1_color}">{v2v1:+.2%}</b>
          </div>
          {strat_table}
          <div class="chart-row">
            <div class="chart-section">{loss}</div>
            <div class="chart-section">{acc}</div>
          </div>
          {grids}
          {excls}
        </div>
        """.format(
            id=cfg.name, name=cfg.name, desc=cfg.description,
            ceiling=ceiling, degrad=degradation,
            mal=sorted(mal) if mal else "none",
            v2fa=v2_fa_delta, v2v1=v2v1_delta,
            v2v1_color="#7FB5A0" if v2v1_delta > 0.005 else "#D4726A" if v2v1_delta < -0.005 else COLOR_TEXT_MUTED,
            strat_table=strat_table,
            loss=_fig_html(fig_loss), acc=_fig_html(fig_acc),
            grids='<div class="chart-row">{}</div>'.format(
                "".join('<div class="chart-section">{}</div>'.format(h) for h in grid_htmls)
            ) if grid_htmls else "",
            excls='<div class="chart-row">{}</div>'.format(
                "".join('<div class="chart-section">{}</div>'.format(h) for h in excl_htmls)
            ) if excl_htmls else "",
        )
        config_sections.append(section)

    # ── Master comparison heatmap ─────────────────────────────────
    # Rows = configs, Cols = strategies, Values = final accuracy
    cfg_names = [r["config"].name for r in attack_configs]
    cfg_descs = [r["config"].description for r in attack_configs]
    strat_names = ["fedavg", "custom:Reputation", "custom:ReputationV2"]
    strat_labels = [SLABEL[s] for s in strat_names]

    acc_matrix = []
    for row in attack_configs:
        acc_row = []
        for s in strat_names:
            acc_row.append(row["strategies"].get(s, {}).get("acc", 0))
        acc_matrix.append(acc_row)
    acc_matrix = np.array(acc_matrix)

    fig_heatmap = go.Figure(data=go.Heatmap(
        z=acc_matrix, x=strat_labels, y=cfg_descs,
        text=[["{:.2%}".format(v) for v in row] for row in acc_matrix],
        texttemplate="%{text}", textfont=dict(size=11, color=COLOR_TEXT),
        colorscale=[[0, "#D4726A"], [0.5, "#D4A76A"], [1, "#7FB5A0"]],
        showscale=True, colorbar=dict(title="Accuracy", tickformat=".0%"),
        xgap=3, ygap=3,
    ))
    fig_heatmap.update_layout(
        **fedsim_layout_defaults(), template="plotly_dark",
        title="Final Accuracy: All Configs x All Strategies",
        height=max(350, 50 * len(cfg_names) + 100),
        margin=dict(t=50, b=40, l=300, r=80),
    )

    # ── Delta heatmap (V2 - FedAvg) ──────────────────────────────
    delta_matrix = []
    for row in attack_configs:
        fa = row["strategies"].get("fedavg", {}).get("acc", 0)
        delta_row = []
        for s in strat_names:
            delta_row.append(row["strategies"].get(s, {}).get("acc", 0) - fa)
        delta_matrix.append(delta_row)
    delta_matrix = np.array(delta_matrix)

    fig_delta = go.Figure(data=go.Heatmap(
        z=delta_matrix, x=strat_labels, y=cfg_descs,
        text=[["{:+.2%}".format(v) for v in row] for row in delta_matrix],
        texttemplate="%{text}", textfont=dict(size=11, color=COLOR_TEXT),
        colorscale=[[0, "#D4726A"], [0.5, COLOR_BG_SURFACE], [1, "#7FB5A0"]],
        zmid=0, showscale=True,
        colorbar=dict(title="Delta vs FedAvg", tickformat="+.0%"),
        xgap=3, ygap=3,
    ))
    fig_delta.update_layout(
        **fedsim_layout_defaults(), template="plotly_dark",
        title="Accuracy Gain over FedAvg",
        height=max(350, 50 * len(cfg_names) + 100),
        margin=dict(t=50, b=40, l=300, r=80),
    )

    # ── Detection heatmap (avg recall) ────────────────────────────
    det_strats = ["custom:Reputation", "custom:ReputationV2"]
    det_labels = [SLABEL[s] for s in det_strats]
    recall_matrix = []
    for row in attack_configs:
        recall_row = []
        for s in det_strats:
            recall_row.append(row["strategies"].get(s, {}).get("avg_rec", 0))
        recall_matrix.append(recall_row)
    recall_matrix = np.array(recall_matrix)

    fig_recall = go.Figure(data=go.Heatmap(
        z=recall_matrix, x=det_labels, y=cfg_descs,
        text=[["{:.0%}".format(v) for v in row] for row in recall_matrix],
        texttemplate="%{text}", textfont=dict(size=12, color=COLOR_TEXT),
        colorscale=[[0, "#D4726A"], [0.5, "#D4A76A"], [1, "#7FB5A0"]],
        showscale=True, colorbar=dict(title="Avg Recall", tickformat=".0%"),
        xgap=3, ygap=3,
    ))
    fig_recall.update_layout(
        **fedsim_layout_defaults(), template="plotly_dark",
        title="Malicious Client Detection: Average Post-Warmup Recall",
        height=max(350, 50 * len(cfg_names) + 100),
        margin=dict(t=50, b=40, l=300, r=80),
    )

    # ── Attack degradation bar chart ──────────────────────────────
    degrad_names = []
    degrad_vals = []
    for row in attack_configs:
        cfg = row["config"]
        ceiling = ceilings.get(cfg.partition_type, 0)
        fa_acc = row["strategies"].get("fedavg", {}).get("acc", 0)
        degrad_names.append(cfg.description)
        degrad_vals.append(ceiling - fa_acc)

    fig_degrad = go.Figure(data=go.Bar(
        x=degrad_vals, y=degrad_names, orientation="h",
        text=["{:.1%}".format(v) for v in degrad_vals],
        textposition="outside",
        marker=dict(color=[COLOR_ATTACKED if v > 0.1 else "#D4A76A" for v in degrad_vals]),
    ))
    fig_degrad.update_layout(
        **fedsim_layout_defaults(), template="plotly_dark",
        title="Attack Degradation (Clean Ceiling - FedAvg under Attack)",
        height=max(300, 40 * len(degrad_names) + 100),
        margin=dict(t=50, b=40, l=300, r=80),
        xaxis=dict(tickformat=".0%"),
    )

    # ── Best separation table ─────────────────────────────────────
    best_rows = []
    for row in attack_configs:
        cfg = row["config"]
        fa = row["strategies"].get("fedavg", {}).get("acc", 0)
        v1 = row["strategies"].get("custom:Reputation", {}).get("acc", 0)
        v2 = row["strategies"].get("custom:ReputationV2", {}).get("acc", 0)
        ceiling = ceilings.get(cfg.partition_type, 0)
        degrad = ceiling - fa
        v2_vs_fa = v2 - fa
        v2_vs_v1 = v2 - v1
        spread = max(fa, v1, v2) - min(fa, v1, v2)
        best_rows.append((cfg, degrad, v2_vs_fa, v2_vs_v1, spread, fa, v1, v2))

    best_rows.sort(key=lambda x: x[4], reverse=True)  # sort by spread

    separation_table = '<table class="data-table"><thead><tr>'
    separation_table += '<th>Config</th><th>Attack Degrad</th>'
    separation_table += '<th>FedAvg</th><th>Rep V1</th><th>Rep V2</th>'
    separation_table += '<th>V2-FedAvg</th><th>V2-V1</th><th>Spread</th></tr></thead><tbody>'
    for cfg, degrad, v2fa, v2v1, spread, fa, v1, v2 in best_rows:
        separation_table += '<tr><td>{}</td><td>{:.2%}</td>'.format(cfg.description, degrad)
        separation_table += '<td>{:.2%}</td><td>{:.2%}</td><td>{:.2%}</td>'.format(fa, v1, v2)
        separation_table += '<td>{:+.2%}</td><td style="color:{}">{:+.2%}</td>'.format(
            v2fa, "#7FB5A0" if v2v1 > 0.005 else "#D4726A", v2v1)
        separation_table += '<td><b>{:.2%}</b></td></tr>'.format(spread)
    separation_table += '</tbody></table>'

    # ── Assemble HTML ─────────────────────────────────────────────
    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>FEDSIM Ablation Study — Strategy Differentiation</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
  :root {{ --bg-dark: {bg_dark}; --bg-surface: {bg_surface}; --border: {border};
    --text: {text}; --text-muted: {text_muted}; --benign: {benign}; --attacked: {attacked}; }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ background: var(--bg-dark); color: var(--text);
    font-family: 'Inter', -apple-system, sans-serif; line-height: 1.6; padding: 2rem; }}
  .container {{ max-width: 1400px; margin: 0 auto; }}
  h1 {{ font-size: 1.8rem; margin-bottom: 0.25rem; color: var(--benign); }}
  h2 {{ font-size: 1.4rem; margin: 2.5rem 0 1rem; padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border); color: var(--benign); }}
  h3 {{ font-size: 1.1rem; margin: 1.5rem 0 0.5rem; color: var(--text); }}
  p, li {{ margin-bottom: 0.5rem; }}
  ul {{ padding-left: 1.5rem; }}
  .subtitle {{ color: var(--text-muted); font-size: 0.9rem; margin-bottom: 2rem; }}
  .chart-section {{ background: var(--bg-surface); border: 1px solid var(--border);
    border-radius: 8px; padding: 1rem; margin: 1rem 0; }}
  .chart-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }}
  @media (max-width: 1000px) {{ .chart-row {{ grid-template-columns: 1fr; }} }}
  .data-table {{ width: 100%; border-collapse: collapse; margin: 1rem 0; font-size: 0.82rem; }}
  .data-table th, .data-table td {{ padding: 0.4rem 0.6rem; border: 1px solid var(--border); text-align: left; }}
  .data-table th {{ background: var(--bg-surface); color: var(--text-muted); font-weight: 600; }}
  .data-table tr:nth-child(even) {{ background: rgba(35,39,48,0.5); }}
  .highlight {{ background: rgba(127,181,160,0.1); border-left: 3px solid var(--benign);
    padding: 0.8rem 1rem; margin: 1rem 0; border-radius: 0 4px 4px 0; }}
  .config-section {{ margin: 2rem 0; padding: 1.5rem; background: var(--bg-surface);
    border: 1px solid var(--border); border-radius: 8px; }}
  .config-meta {{ color: var(--text-muted); font-size: 0.85rem; margin-bottom: 1rem; }}
  .footer {{ margin-top: 3rem; padding-top: 1rem; border-top: 1px solid var(--border);
    color: var(--text-muted); font-size: 0.8rem; text-align: center; }}
  .toc a {{ color: var(--benign); text-decoration: none; }}
  .toc a:hover {{ text-decoration: underline; }}
</style>
</head>
<body>
<div class="container">

<h1>Ablation Study: Finding Conditions That Differentiate FL Defense Strategies</h1>
<p class="subtitle">
  {timestamp} &mdash; CIFAR-10 / ResNet-18 / {nc} clients / {nr} rounds / {ne} epochs / LR {lr} / Seed {seed}<br>
  {n_configs} configs &times; up to 3 strategies = {n_runs} runs &mdash; Total: {elapsed:.0f}s ({elapsed_min:.1f} min)
</p>

<h2>1. Study Design</h2>
<p>This study systematically varies <b>data distribution</b> (IID vs Non-IID &alpha;=0.5),
   <b>attack type</b> (label flipping vs gradient scaling 5x), and <b>malicious fraction</b>
   (28.6% vs 42.8%) to identify conditions that best differentiate FedAvg, Reputation V1,
   and ReputationV2.</p>

<h3>Table of Contents</h3>
<div class="toc"><ul>
  <li><a href="#heatmaps">2. Cross-Config Comparison Heatmaps</a></li>
  <li><a href="#separation">3. Strategy Separation Ranking</a></li>
  <li><a href="#degradation">4. Attack Degradation Analysis</a></li>
  <li><a href="#configs">5. Per-Config Detailed Results</a></li>
  <li><a href="#findings">6. Key Findings</a></li>
</ul></div>

<h2 id="heatmaps">2. Cross-Config Comparison</h2>

<h3>2.1 Absolute Accuracy</h3>
<div class="chart-section">{heatmap}</div>

<h3>2.2 Accuracy Gain over FedAvg</h3>
<div class="chart-section">{delta_heatmap}</div>

<h3>2.3 Malicious Client Detection (Average Recall)</h3>
<div class="chart-section">{recall_heatmap}</div>

<h2 id="separation">3. Strategy Separation Ranking</h2>
<p>Configs ranked by <b>spread</b> (max accuracy - min accuracy across the 3 strategies).
   Higher spread = better differentiation.</p>
{separation_table}

<h2 id="degradation">4. Attack Degradation</h2>
<p>How much accuracy each attack configuration costs vs the clean baseline.</p>
<div class="chart-section">{degrad_chart}</div>

<h3>Clean Baselines</h3>
<table class="data-table">
  <thead><tr><th>Condition</th><th>Clean Accuracy</th></tr></thead>
  <tbody>{ceiling_rows}</tbody>
</table>

<h2 id="configs">5. Per-Config Detailed Results</h2>
{"".join(config_sections)}

<h2 id="findings">6. Key Findings</h2>
<div class="highlight">
  <p><b>Analysis will be added after reviewing the data.</b> Open this report in a browser
  and examine the heatmaps and per-config sections to identify:</p>
  <ul>
    <li>Which attack type creates the most strategy differentiation?</li>
    <li>Does IID vs Non-IID compress or amplify the differences?</li>
    <li>At what malicious fraction does V2 clearly outperform V1?</li>
    <li>Where does V2's better detection translate to better accuracy?</li>
  </ul>
</div>

<div class="footer">
  Generated by FEDSIM &mdash; {timestamp}
</div>

</div>
</body>
</html>""".format(
        bg_dark=COLOR_BG_DARK, bg_surface=COLOR_BG_SURFACE, border=COLOR_BORDER,
        text=COLOR_TEXT, text_muted=COLOR_TEXT_MUTED, benign=COLOR_BENIGN, attacked=COLOR_ATTACKED,
        timestamp=timestamp,
        nc=NUM_CLIENTS, nr=NUM_ROUNDS, ne=LOCAL_EPOCHS, lr=LEARNING_RATE, seed=SEED,
        n_configs=len(CONFIGS), n_runs=sum(len(c.strategies) for c in CONFIGS),
        elapsed=total_elapsed, elapsed_min=total_elapsed / 60,
        heatmap=_fig_html(fig_heatmap),
        delta_heatmap=_fig_html(fig_delta),
        recall_heatmap=_fig_html(fig_recall),
        separation_table=separation_table,
        degrad_chart=_fig_html(fig_degrad),
        ceiling_rows="".join(
            "<tr><td>{}</td><td>{:.4f}</td></tr>".format(k, v)
            for k, v in ceilings.items()),
    )
    return html


# ── Main ──────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    all_results = run_all()
    total_elapsed = time.time() - t0

    print("\n" + "=" * 65)
    print("  All runs complete. Generating report...")
    print("=" * 65)

    html = generate_report(all_results, total_elapsed)
    report_path = os.path.join(OUTPUT_DIR, "ablation_study.html")
    with open(report_path, "w") as f:
        f.write(html)
    print("Report saved to: {}".format(report_path))
    return report_path


if __name__ == "__main__":
    main()
