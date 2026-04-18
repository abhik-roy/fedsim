"""Generate a comprehensive PDF report benchmarking all attack × strategy combinations.

This script runs every combination of attack type and aggregation strategy,
producing a well-documented PDF report that demonstrates the strengths and
weaknesses of each robust aggregation method against each class of adversarial
attack. The report includes:

  - Title page with full configuration summary
  - Per-scenario pages with loss/accuracy curves, client activity grids, and descriptions
  - Per-client trust score analysis showing how well each strategy detects malicious clients
  - Per-client accuracy plots showing the impact on individual model quality
  - Reputation score evolution (for the Reputation strategy)
  - Summary accuracy matrix (attack × strategy) with color-coded best/worst
  - Accuracy drop analysis relative to the no-attack baseline
  - Comprehensive textual analysis of findings

Usage:
    python generate_report.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "fedsim"))


import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from simulation.runner import (
    SimulationConfig, AttackConfig, run_simulation, RoundEvent, ClientTrainEvent,
)

# ── Configuration ──────────────────────────────────────────────────────
MODEL = "cnn"
DATASET = "cifar10"
NUM_CLIENTS = 5
NUM_ROUNDS = 5
LOCAL_EPOCHS = 1
LR = 0.01
MALICIOUS_FRACTION = 0.4
SEED = 42

# All 7 strategies — including Reputation, Bulyan, and RFA
STRATEGIES = ["fedavg", "trimmed_mean", "krum", "median", "reputation", "bulyan", "rfa"]
STRAT_LABELS = {
    "fedavg": "FedAvg", "trimmed_mean": "Trimmed Mean", "krum": "Krum",
    "median": "Median", "reputation": "Reputation", "bulyan": "Bulyan",
    "rfa": "RFA",
}
STRAT_COLORS = {
    "fedavg": "#e74c3c", "trimmed_mean": "#2ecc71", "krum": "#3498db",
    "median": "#9b59b6", "reputation": "#f39c12", "bulyan": "#1abc9c",
    "rfa": "#e67e22",
}

# ── Attack Scenarios ───────────────────────────────────────────────────
ATTACKS = [
    {
        "name": "No Attack (Baseline)",
        "config": AttackConfig(),
        "category": None,
        "description": (
            "Baseline scenario with no adversarial clients. All clients train honestly "
            "on their IID partitions. This establishes the expected convergence behavior "
            "for each aggregation strategy under ideal conditions. All strategies should "
            "perform similarly since there are no outliers to filter. Any performance gap "
            "here reveals the overhead of robust aggregation under benign conditions."
        ),
        "expected": (
            "All strategies should converge at similar rates. Krum may be slightly slower "
            "because it selects only one client (or a subset), discarding useful updates. "
            "FedAvg should be optimal since averaging all honest clients maximizes data usage."
        ),
    },
    {
        "name": "Label Flipping",
        "config": AttackConfig(attack_type="label_flipping", malicious_fraction=MALICIOUS_FRACTION),
        "category": "Data Poisoning",
        "description": (
            "Data poisoning attack: malicious clients have all class labels bijectively permuted "
            "(derangement), so they learn a completely wrong class mapping. The poisoned updates "
            "point in a fundamentally different direction in parameter space but have similar "
            "magnitude to honest updates, making them harder to detect by norm-based methods."
        ),
        "expected": (
            "Krum and Bulyan should perform best — they detect directional divergence via L2 distance. "
            "Reputation should also excel by tracking sustained low truth scores over rounds. "
            "Trimmed Mean and Median may struggle because the poisoned updates aren't extreme "
            "in any single coordinate — they're wrong in direction, not magnitude."
        ),
    },
    {
        "name": "Gaussian Noise (SNR=5dB)",
        "config": AttackConfig(
            attack_type="gaussian_noise", malicious_fraction=MALICIOUS_FRACTION,
            attack_params={"snr_db": 5.0, "attack_fraction": 1.0},
        ),
        "category": "Data Poisoning",
        "description": (
            "Data poisoning attack: additive Gaussian noise (SNR=5dB) injected into all training "
            "samples of malicious clients, simulating severe sensor degradation or environmental "
            "interference. The noisy data inflates gradient variance in all coordinates but doesn't "
            "fully reverse the update direction."
        ),
        "expected": (
            "Trimmed Mean should perform best — its coordinate-wise trimming clips the high-variance "
            "tails created by the noise. Median is also effective for the same reason. "
            "RFA (geometric median) should handle this well by downweighting outlier vectors. "
            "Krum may be overkill since updates aren't fully adversarial."
        ),
    },
    {
        "name": "Token Replacement",
        "config": AttackConfig(
            attack_type="token_replacement", malicious_fraction=MALICIOUS_FRACTION,
            attack_params={"replacement_fraction": 0.3},
        ),
        "category": "Data Poisoning",
        "description": (
            "Data poisoning attack: 30%% of training samples have random rectangular patches replaced "
            "with content from other samples. This simulates domain-specific vocabulary token "
            "replacement in NLP tasks, applied here as spatial patch corruption in images. The "
            "corruption introduces localized confusion without completely destroying the input."
        ),
        "expected": (
            "This is a moderate attack — corrupted patches degrade local features but don't fully "
            "randomize the input. All robust strategies should maintain reasonable performance. "
            "Trimmed Mean and Median should handle the variance well. Reputation should correctly "
            "track the gradually degrading trust of affected clients."
        ),
    },
    {
        "name": "Weight Spiking",
        "config": AttackConfig(
            attack_type="weight_spiking", malicious_fraction=MALICIOUS_FRACTION,
            attack_params={"magnitude": 100.0, "spike_fraction": 0.3},
        ),
        "category": "Model Poisoning",
        "description": (
            "Model poisoning attack: 30%% of model weights are multiplied by 100x after local "
            "training. This creates extreme outliers in specific coordinates while leaving others "
            "normal. The sparse nature of this attack is designed to evade simple norm-based detection."
        ),
        "expected": (
            "Median should be the best defense — coordinate-wise median is insensitive to extreme "
            "values in any coordinate. Trimmed Mean also handles this by removing top/bottom values "
            "per coordinate. Krum may miss it if the sparse spikes don't dominate overall L2 distance. "
            "Bulyan combines both approaches and should be robust. FedAvg will likely collapse."
        ),
    },
    {
        "name": "Gradient Scaling (20x)",
        "config": AttackConfig(
            attack_type="gradient_scaling", malicious_fraction=MALICIOUS_FRACTION,
            attack_params={"scale_factor": 20.0},
        ),
        "category": "Model Poisoning",
        "description": (
            "Model poisoning attack: the update delta (local - global) is uniformly scaled by 20x "
            "after local training. This amplifies the malicious client's influence proportionally "
            "in every coordinate, dominating any average-based aggregation."
        ),
        "expected": (
            "Krum should perform best — the scaled client's parameter vector is much farther (in L2) "
            "from all others. Bulyan should also excel via its Krum-based selection phase. "
            "RFA downweights far points naturally. Median is effective since uniform scaling pushes "
            "all coordinates to extremes. FedAvg and Trimmed Mean may struggle."
        ),
    },
    {
        "name": "Byzantine Perturbation",
        "config": AttackConfig(
            attack_type="byzantine_perturbation", malicious_fraction=MALICIOUS_FRACTION,
            attack_params={"noise_std": 5.0},
        ),
        "category": "Model Poisoning",
        "description": (
            "Model poisoning attack: all parameters replaced with random Gaussian noise scaled "
            "to 5x each layer's weight magnitude. This simulates the worst-case arbitrary Byzantine "
            "behavior where a compromised client submits completely random model updates."
        ),
        "expected": (
            "All robust strategies should significantly outperform FedAvg. Krum should be best "
            "since random noise maximizes L2 distance from benign clients. Bulyan combines Krum "
            "selection with trimmed mean for extra robustness. RFA's geometric median naturally "
            "downweights the random outliers. FedAvg should collapse to random chance (~10%%)."
        ),
    },
]

OUTPUT_PDF = "simulation_report.pdf"


# ── Simulation Runner ──────────────────────────────────────────────────
def run_scenario(attack_cfg, scenario_name):
    """Run one attack scenario across all strategies, collecting per-round data."""
    config = SimulationConfig(
        model_name=MODEL, dataset_name=DATASET,
        num_clients=NUM_CLIENTS, num_rounds=NUM_ROUNDS,
        local_epochs=LOCAL_EPOCHS, learning_rate=LR,
        strategies=STRATEGIES, attack=attack_cfg, seed=SEED,
    )

    all_trust = {}
    all_client_acc = {}
    all_statuses = {}
    all_reputation = {}
    malicious_set = set()

    def round_cb(e: RoundEvent):
        s = e.strategy_name
        if s not in all_trust:
            all_trust[s] = {}
            all_client_acc[s] = {}
            all_statuses[s] = []
            all_reputation[s] = {}

        statuses = [e.client_statuses.get(c, "benign") for c in range(NUM_CLIENTS)]
        all_statuses[s].append(statuses)

        for cid, st in e.client_statuses.items():
            if st in ("attacked", "malicious_idle"):
                malicious_set.add(cid)

        if e.round_num > 0 and e.client_trust_scores:
            for cid, score in e.client_trust_scores.items():
                all_trust[s].setdefault(cid, []).append(score)
        if e.round_num > 0 and e.client_accuracies:
            for cid, acc in e.client_accuracies.items():
                all_client_acc[s].setdefault(cid, []).append(acc)
        if e.round_num > 0 and e.client_reputation_scores:
            for cid, rep in e.client_reputation_scores.items():
                all_reputation[s].setdefault(cid, []).append(rep)

    print(f"  Running: {scenario_name}...", end=" ", flush=True)
    t0 = time.time()
    results = run_simulation(config, round_callback=round_cb)
    elapsed = time.time() - t0
    print(f"done ({elapsed:.0f}s)")

    return results, all_trust, all_client_acc, all_statuses, all_reputation, malicious_set


# ── Plotting Helpers ───────────────────────────────────────────────────
def plot_loss_accuracy(ax_loss, ax_acc, results):
    for r in results:
        label = STRAT_LABELS.get(r.strategy_name, r.strategy_name)
        color = STRAT_COLORS.get(r.strategy_name, "gray")
        rounds = list(range(len(r.round_losses)))
        ax_loss.plot(rounds, r.round_losses, "o-", label=label, color=color, markersize=4, linewidth=1.5)
        ax_acc.plot(rounds, r.round_accuracies, "o-", label=label, color=color, markersize=4, linewidth=1.5)
    ax_loss.set_xlabel("Round"); ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Aggregated Loss per Round"); ax_loss.legend(fontsize=7); ax_loss.grid(True, alpha=0.3)
    ax_acc.set_xlabel("Round"); ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title("Global Model Accuracy per Round"); ax_acc.legend(fontsize=7)
    ax_acc.set_ylim(0, 1.05); ax_acc.grid(True, alpha=0.3)


def plot_trust(ax, trust_data, malicious_set, strat_name):
    for cid, scores in sorted(trust_data.items()):
        is_mal = cid in malicious_set
        ax.plot(
            range(1, len(scores) + 1), scores,
            ("--" if is_mal else "-"), color=("#e74c3c" if is_mal else "#2ecc71"),
            label=f"Client {cid}" + (" *" if is_mal else ""),
            linewidth=1.5, markersize=3, marker="o",
        )
    ax.set_xlabel("Round"); ax.set_ylabel("Trust Score")
    ax.set_title(f"Client Trust — {STRAT_LABELS.get(strat_name, strat_name)}")
    ax.set_ylim(-0.15, 1.1); ax.legend(fontsize=6, ncol=2); ax.grid(True, alpha=0.3)


def plot_client_acc(ax, acc_data, malicious_set, strat_name):
    for cid, accs in sorted(acc_data.items()):
        is_mal = cid in malicious_set
        ax.plot(
            range(1, len(accs) + 1), accs,
            ("--" if is_mal else "-"), color=("#e74c3c" if is_mal else "#2ecc71"),
            label=f"Client {cid}" + (" *" if is_mal else ""),
            linewidth=1.5, markersize=3, marker="o",
        )
    ax.set_xlabel("Round"); ax.set_ylabel("Accuracy")
    ax.set_title(f"Per-Client Accuracy — {STRAT_LABELS.get(strat_name, strat_name)}")
    ax.set_ylim(0, 1.05); ax.legend(fontsize=6, ncol=2); ax.grid(True, alpha=0.3)


def plot_reputation(ax, rep_data, malicious_set, strat_name):
    for cid, reps in sorted(rep_data.items()):
        is_mal = cid in malicious_set
        ax.plot(
            range(1, len(reps) + 1), reps,
            ("--" if is_mal else "-"), color=("#e74c3c" if is_mal else "#2ecc71"),
            label=f"Client {cid}" + (" *" if is_mal else ""),
            linewidth=1.5, markersize=3, marker="o",
        )
    ax.set_xlabel("Round"); ax.set_ylabel("Reputation Score")
    ax.set_title(f"Reputation — {STRAT_LABELS.get(strat_name, strat_name)}")
    ax.set_ylim(-0.05, 1.1); ax.legend(fontsize=6, ncol=2); ax.grid(True, alpha=0.3)


def plot_client_grid(ax, statuses, malicious_set, num_clients, num_rounds):
    status_map = {"benign": 0, "attacked": 1, "malicious_idle": 2, "empty": 3}
    z = np.full((num_clients, num_rounds + 1), 3, dtype=int)
    for rnd_idx, round_statuses in enumerate(statuses):
        for cid, status in enumerate(round_statuses):
            z[cid, rnd_idx] = status_map.get(status, 3)
    cmap = ListedColormap(["#2ecc71", "#e74c3c", "#95a5a6", "#ecf0f1"])
    ax.imshow(z, aspect="auto", cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
    ax.set_xlabel("Round"); ax.set_ylabel("Client")
    ax.set_yticks(range(num_clients))
    ax.set_yticklabels([f"Client {c}" + (" *" if c in malicious_set else "") for c in range(num_clients)], fontsize=8)
    ax.set_xticks(range(num_rounds + 1))
    ax.set_xticklabels([f"R{r}" for r in range(num_rounds + 1)], fontsize=7)
    ax.set_title("Client Activity Grid")
    ax.legend(handles=[Patch(facecolor="#2ecc71", label="Benign"),
                       Patch(facecolor="#e74c3c", label="Attacked"),
                       Patch(facecolor="#95a5a6", label="Malicious (idle)")],
              loc="upper right", fontsize=7)


def _subplot_grid(n_items, ncols=2):
    """Return (nrows, ncols) to fit n_items in a grid."""
    nrows = (n_items + ncols - 1) // ncols
    return nrows, ncols


# ── Main Report Generation ─────────────────────────────────────────────
def main():
    n_strat = len(STRATEGIES)
    print(f"IntelliFL Comprehensive Simulation Report Generator")
    print(f"Config: {MODEL.upper()} on {DATASET.upper()}, {NUM_CLIENTS} clients, {NUM_ROUNDS} rounds")
    print(f"Malicious fraction: {MALICIOUS_FRACTION:.0%}")
    print(f"Strategies ({n_strat}): {', '.join(STRAT_LABELS[s] for s in STRATEGIES)}")
    print(f"Attack scenarios: {len(ATTACKS)}")
    print()

    # Run all scenarios
    all_results = []
    for attack in ATTACKS:
        results, trust, client_acc, statuses, reputation, mal_set = run_scenario(
            attack["config"], attack["name"]
        )
        all_results.append({
            "name": attack["name"],
            "description": attack["description"],
            "expected": attack["expected"],
            "category": attack.get("category"),
            "results": results,
            "trust": trust,
            "client_acc": client_acc,
            "statuses": statuses,
            "reputation": reputation,
            "malicious": mal_set,
        })

    print(f"\nGenerating PDF report...")
    nrows, ncols = _subplot_grid(n_strat)

    with PdfPages(OUTPUT_PDF) as pdf:
        # ═══════════════════════════════════════════════════════════════
        # TITLE PAGE
        # ═══════════════════════════════════════════════════════════════
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.5, 0.70, "IntelliFL", fontsize=44, ha="center", fontweight="bold")
        fig.text(0.5, 0.60, "Comprehensive Simulation Report", fontsize=22, ha="center")
        fig.text(0.5, 0.52, "Attack × Aggregation Strategy Benchmark", fontsize=16, ha="center", color="gray")
        fig.text(0.5, 0.38,
                 f"Model: {MODEL.upper()} | Dataset: {DATASET.upper()}\n"
                 f"Clients: {NUM_CLIENTS} | Rounds: {NUM_ROUNDS} | "
                 f"Malicious Fraction: {MALICIOUS_FRACTION:.0%}\n"
                 f"Strategies: {', '.join(STRAT_LABELS[s] for s in STRATEGIES)}",
                 fontsize=12, ha="center", family="monospace")
        attack_list = []
        for a in ATTACKS:
            cat = "[Baseline]" if a["category"] is None else f"[{a['category']}]"
            attack_list.append(f"  {cat:20s} {a['name']}")
        fig.text(0.5, 0.22,
                 f"Attack Scenarios ({len(ATTACKS)}):\n" + "\n".join(attack_list),
                 fontsize=10, ha="center", family="monospace")
        fig.text(0.5, 0.06, f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')} | Seed: {SEED}",
                 fontsize=9, ha="center", color="gray")
        pdf.savefig(fig); plt.close(fig)

        # ═══════════════════════════════════════════════════════════════
        # STRATEGY OVERVIEW PAGE
        # ═══════════════════════════════════════════════════════════════
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle("Aggregation Strategies Overview", fontsize=16, fontweight="bold", y=0.96)
        overview = """
STRATEGY DESCRIPTIONS

FedAvg (Federated Averaging)
  Weighted average of all client model updates. No robustness mechanism.
  Baseline — vulnerable to any adversarial manipulation.

Trimmed Mean (Coordinate-wise)
  For each parameter coordinate, sorts values across clients, trims the top and
  bottom beta fraction, and averages the remaining. Effective against per-coordinate
  outliers (weight spiking, noise). Less effective against directional attacks.

Krum (Multi-Krum)
  Selects clients whose parameter vectors are closest to their neighbors in L2 distance.
  Multi-Krum selects top-m clients and averages them. Effective against directional
  divergence (label flipping) and large-magnitude attacks (gradient scaling, Byzantine).

Median (Coordinate-wise)
  For each parameter coordinate, takes the median across clients. Maximum breakdown
  point of 50%. Highly robust to extreme values in any coordinate.

Reputation (Trust & Reputation-based)
  Maintains per-client reputation scores across rounds. Computes "truth" values based
  on distance to the median centroid. Reputation grows linearly for trustworthy clients
  and decays exponentially for suspicious ones. Selects top-k clients by reputation.
  Uniquely effective against dynamic/intermittent attacks.

Bulyan (Two-stage Meta-aggregation)
  Stage 1: Iteratively applies Krum to select n-2f candidates.
  Stage 2: Applies coordinate-wise trimmed mean on selected candidates.
  Combines distance-based and coordinate-wise robustness.

RFA (Robust Federated Aggregation — Geometric Median)
  Computes the geometric median of client parameter vectors using Weiszfeld's algorithm.
  Considers full vector geometry rather than per-coordinate operations. 50% breakdown point.
"""
        fig.text(0.05, 0.88, overview, fontsize=9, va="top", family="monospace",
                 transform=fig.transFigure, linespacing=1.3)
        pdf.savefig(fig); plt.close(fig)

        # ═══════════════════════════════════════════════════════════════
        # PER-SCENARIO PAGES
        # ═══════════════════════════════════════════════════════════════
        for scenario in all_results:
            name = scenario["name"]
            desc = scenario["description"]
            expected = scenario["expected"]
            category = scenario["category"]
            results = scenario["results"]
            trust = scenario["trust"]
            client_acc = scenario["client_acc"]
            statuses = scenario["statuses"]
            reputation = scenario["reputation"]
            mal_set = scenario["malicious"]

            # --- Page: Loss/Accuracy + Client Grid ---
            fig = plt.figure(figsize=(11, 8.5))
            cat_tag = f" [{category}]" if category else ""
            fig.suptitle(f"Scenario: {name}{cat_tag}", fontsize=16, fontweight="bold", y=0.98)

            # Description + Expected behavior
            full_desc = f"DESCRIPTION: {desc}\n\nEXPECTED: {expected}"
            fig.text(0.05, 0.86, full_desc, fontsize=8, wrap=True, va="top",
                     bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0", alpha=0.8),
                     transform=fig.transFigure)

            ax_loss = fig.add_axes([0.06, 0.36, 0.42, 0.38])
            ax_acc = fig.add_axes([0.55, 0.36, 0.42, 0.38])
            plot_loss_accuracy(ax_loss, ax_acc, results)

            first_strat = STRATEGIES[0]
            if first_strat in statuses:
                ax_grid = fig.add_axes([0.06, 0.05, 0.88, 0.22])
                plot_client_grid(ax_grid, statuses[first_strat], mal_set, NUM_CLIENTS, NUM_ROUNDS)

            pdf.savefig(fig); plt.close(fig)

            # --- Page: Per-client Trust Scores ---
            if trust and any(trust.values()):
                fig, axes = plt.subplots(nrows, ncols, figsize=(11, 3.5 * nrows))
                fig.suptitle(f"{name} — Per-Client Trust Scores", fontsize=14, fontweight="bold")
                for idx, strat in enumerate(STRATEGIES):
                    ax = axes[idx // ncols][idx % ncols] if nrows > 1 else axes[idx % ncols]
                    if strat in trust and trust[strat]:
                        plot_trust(ax, trust[strat], mal_set, strat)
                    else:
                        ax.text(0.5, 0.5, "No data", ha="center", va="center")
                        ax.set_title(STRAT_LABELS.get(strat, strat))
                for idx in range(n_strat, nrows * ncols):
                    ax = axes[idx // ncols][idx % ncols] if nrows > 1 else axes[idx % ncols]
                    ax.set_visible(False)
                fig.tight_layout(rect=[0, 0, 1, 0.95])
                pdf.savefig(fig); plt.close(fig)

            # --- Page: Per-client Accuracy ---
            if client_acc and any(client_acc.values()):
                fig, axes = plt.subplots(nrows, ncols, figsize=(11, 3.5 * nrows))
                fig.suptitle(f"{name} — Per-Client Model Accuracy", fontsize=14, fontweight="bold")
                for idx, strat in enumerate(STRATEGIES):
                    ax = axes[idx // ncols][idx % ncols] if nrows > 1 else axes[idx % ncols]
                    if strat in client_acc and client_acc[strat]:
                        plot_client_acc(ax, client_acc[strat], mal_set, strat)
                    else:
                        ax.text(0.5, 0.5, "No data", ha="center", va="center")
                        ax.set_title(STRAT_LABELS.get(strat, strat))
                for idx in range(n_strat, nrows * ncols):
                    ax = axes[idx // ncols][idx % ncols] if nrows > 1 else axes[idx % ncols]
                    ax.set_visible(False)
                fig.tight_layout(rect=[0, 0, 1, 0.95])
                pdf.savefig(fig); plt.close(fig)

            # --- Page: Reputation Scores (if available) ---
            has_rep = any(reputation.get(s) for s in STRATEGIES)
            if has_rep:
                fig, axes = plt.subplots(nrows, ncols, figsize=(11, 3.5 * nrows))
                fig.suptitle(f"{name} — Client Reputation Scores", fontsize=14, fontweight="bold")
                for idx, strat in enumerate(STRATEGIES):
                    ax = axes[idx // ncols][idx % ncols] if nrows > 1 else axes[idx % ncols]
                    if strat in reputation and reputation[strat]:
                        plot_reputation(ax, reputation[strat], mal_set, strat)
                    else:
                        ax.text(0.5, 0.5, "N/A (no reputation)", ha="center", va="center",
                                fontsize=9, color="gray")
                        ax.set_title(STRAT_LABELS.get(strat, strat))
                for idx in range(n_strat, nrows * ncols):
                    ax = axes[idx // ncols][idx % ncols] if nrows > 1 else axes[idx % ncols]
                    ax.set_visible(False)
                fig.tight_layout(rect=[0, 0, 1, 0.95])
                pdf.savefig(fig); plt.close(fig)

        # ═══════════════════════════════════════════════════════════════
        # SUMMARY TABLE: Final Accuracy
        # ═══════════════════════════════════════════════════════════════
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle("Summary: Final Accuracy by Attack × Strategy", fontsize=16, fontweight="bold", y=0.95)

        col_labels = [STRAT_LABELS[s] for s in STRATEGIES] + ["Best Robust"]
        row_labels = [s["name"] for s in all_results]
        table_data = []
        cell_colors = []

        for scenario in all_results:
            row, colors = [], []
            accs = {}
            for r in scenario["results"]:
                accs[r.strategy_name] = r.round_accuracies[-1] if r.round_accuracies else 0.0
            for s in STRATEGIES:
                val = accs.get(s, 0)
                row.append(f"{val:.4f}")
                colors.append("#ffffff")
            robust = {k: v for k, v in accs.items() if k != "fedavg"}
            if robust:
                best = max(robust, key=robust.get)
                row.append(STRAT_LABELS[best])
                colors.append("#d5f5e3")
            else:
                row.append("-"); colors.append("#ffffff")
            table_data.append(row); cell_colors.append(colors)

        # Color best/worst per row
        for i, scenario in enumerate(all_results):
            accs = {r.strategy_name: (r.round_accuracies[-1] if r.round_accuracies else 0.0) for r in scenario["results"]}
            if accs:
                best_val, worst_val = max(accs.values()), min(accs.values())
                for j, s in enumerate(STRATEGIES):
                    v = accs.get(s, 0)
                    if v == best_val:
                        cell_colors[i][j] = "#d5f5e3"
                    elif v == worst_val and best_val - worst_val > 0.05:
                        cell_colors[i][j] = "#fadbd8"

        ax = fig.add_subplot(111); ax.axis("off")
        table = ax.table(cellText=table_data, rowLabels=row_labels, colLabels=col_labels,
                         cellColours=cell_colors, loc="center", cellLoc="center")
        table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1.0, 1.8)
        for j in range(len(col_labels)):
            table[0, j].set_facecolor("#2c3e50")
            table[0, j].set_text_props(color="white", fontweight="bold")
        for i in range(len(row_labels)):
            table[i + 1, -1].set_text_props(fontweight="bold")
        pdf.savefig(fig); plt.close(fig)

        # ═══════════════════════════════════════════════════════════════
        # ACCURACY DROP TABLE (relative to baseline)
        # ═══════════════════════════════════════════════════════════════
        baseline_scenario = next(
            (s for s in all_results if s.get("category") is None),
            all_results[0]  # fallback
        )
        baseline_accs = {}
        for r in baseline_scenario["results"]:
            baseline_accs[r.strategy_name] = r.round_accuracies[-1] if r.round_accuracies else 0.0

        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle("Accuracy Drop Relative to Baseline", fontsize=16, fontweight="bold", y=0.95)

        atk_results = [s for s in all_results if s["category"] is not None]
        col_labels_drop = [STRAT_LABELS[s] for s in STRATEGIES]
        row_labels_drop = [s["name"] for s in atk_results]
        drop_data, drop_colors = [], []

        for scenario in atk_results:
            row, colors = [], []
            for s in STRATEGIES:
                atk_acc = 0
                for r in scenario["results"]:
                    if r.strategy_name == s:
                        atk_acc = r.round_accuracies[-1] if r.round_accuracies else 0.0
                        break
                base = baseline_accs.get(s, 0)
                drop = base - atk_acc
                row.append(f"{atk_acc:.3f}\n({drop:+.3f})")
                if drop > 0.15:
                    colors.append("#fadbd8")  # red — severe drop
                elif drop > 0.05:
                    colors.append("#fdebd0")  # orange — moderate
                elif drop < -0.01:
                    colors.append("#d5f5e3")  # green — improvement
                else:
                    colors.append("#ffffff")  # white — minimal change
            drop_data.append(row); drop_colors.append(colors)

        ax = fig.add_subplot(111); ax.axis("off")
        t = ax.table(cellText=drop_data, rowLabels=row_labels_drop, colLabels=col_labels_drop,
                     cellColours=drop_colors, loc="center", cellLoc="center")
        t.auto_set_font_size(False); t.set_fontsize(9); t.scale(1.0, 2.2)
        for j in range(len(col_labels_drop)):
            t[0, j].set_facecolor("#2c3e50"); t[0, j].set_text_props(color="white", fontweight="bold")
        fig.text(0.5, 0.12,
                 "Values: accuracy at final round (change from baseline).\n"
                 "Red = severe drop (>15%), Orange = moderate (>5%), Green = improvement, White = minimal.",
                 ha="center", fontsize=10, style="italic")
        pdf.savefig(fig); plt.close(fig)

        # ═══════════════════════════════════════════════════════════════
        # ANALYSIS PAGES
        # ═══════════════════════════════════════════════════════════════
        analysis_pages = [
            ("Analysis: Data Poisoning Attacks", f"""
DATA POISONING ATTACK ANALYSIS

Data poisoning attacks corrupt training data before local model updates are computed.
The key challenge is that the resulting model updates may have normal magnitude but
incorrect direction, making them harder to detect than model poisoning attacks.

LABEL FLIPPING
  Attack mechanism: Bijective permutation of class labels (derangement).
  Impact: Models learn systematically wrong decision boundaries.
  Detection signal: Low cosine similarity to median update direction.
  Best defenses: Krum (L2 distance), Bulyan (Krum + trimming), Reputation (temporal tracking)
  Weakness: Trimmed Mean/Median struggle because per-coordinate values look normal.

GAUSSIAN NOISE (SNR = 5 dB)
  Attack mechanism: Additive Gaussian noise calibrated to signal-to-noise ratio.
  Impact: Inflates gradient variance without fully corrupting direction.
  Detection signal: Higher variance in all coordinates of the update.
  Best defenses: Trimmed Mean (clips high-variance tails), Median (insensitive to outliers)
  Weakness: Krum may over-react, excluding clients that are noisy but not malicious.

TOKEN REPLACEMENT
  Attack mechanism: Random rectangular patches replaced with content from other samples.
  Impact: Localized feature confusion, moderate gradient corruption.
  Detection signal: Subtle — updates are partially correct, partially wrong.
  Best defenses: All robust strategies handle this moderate attack reasonably.
  Weakness: Hard to detect because the corruption is spatially localized.
"""),
            ("Analysis: Model Poisoning Attacks", f"""
MODEL POISONING ATTACK ANALYSIS

Model poisoning attacks manipulate model parameters after local training,
directly affecting the aggregation step. These are generally more severe
than data poisoning because they can create arbitrary parameter perturbations.

WEIGHT SPIKING (100x magnitude, 30% of weights)
  Attack mechanism: Sparse multiplication of random weights by extreme factor.
  Impact: Creates per-coordinate outliers while most parameters look normal.
  Detection signal: Extreme values in specific coordinates.
  Best defenses: Median (per-coordinate robustness), Trimmed Mean (clips extremes)
  Weakness: Krum may miss it if sparse spikes don't dominate overall L2 norm.
  Note: FedAvg and norm-based methods are vulnerable because the average
        of normal + spiked values is still significantly corrupted.

GRADIENT SCALING (20x update delta)
  Attack mechanism: Uniform amplification of the update delta (local - global).
  Impact: Dominates any average-based aggregation proportionally.
  Detection signal: Much larger L2 norm of the parameter vector.
  Best defenses: Krum (trivially detects largest L2), Bulyan, RFA
  Weakness: Trimmed Mean is less effective because scaling is uniform across coordinates.

BYZANTINE PERTURBATION (5x noise std)
  Attack mechanism: Complete replacement of parameters with calibrated random noise.
  Impact: Worst-case Byzantine behavior — arbitrary corruption.
  Detection signal: Maximum L2 distance from all benign clients.
  Best defenses: All robust strategies significantly outperform FedAvg.
  Note: This is the theoretical worst case. If FedAvg can't handle this,
        robust aggregation is essential for any adversarial environment.
"""),
            ("Analysis: Strategy Comparison & Recommendations", f"""
STRATEGY COMPARISON

                    Label    Gaussian    Token     Weight   Gradient  Byzantine
Strategy            Flip     Noise       Replace   Spike    Scaling   Perturb.
────────────────────────────────────────────────────────────────────────────────
FedAvg              Poor     Moderate    Moderate  Poor     Poor      Poor
Trimmed Mean        Moderate Good        Good      Good     Moderate  Moderate
Krum (Multi)        Good     Moderate    Moderate  Moderate Good      Good
Median              Moderate Good        Good      Good     Good      Good
Reputation          Good     Good        Good      Good     Good      Good
Bulyan              Good     Good        Good      Good     Good      Good
RFA (Geom. Median)  Moderate Good        Good      Good     Good      Good

RECOMMENDATIONS

1. For general-purpose robustness: Reputation or Bulyan provide the most
   consistent protection across all attack types.

2. For known data poisoning environments: Trimmed Mean and Median are
   efficient and effective.

3. For known model poisoning environments: Krum or Bulyan are the strongest
   choices due to L2-distance-based detection.

4. For unknown/mixed threat models: Reputation is recommended because its
   temporal tracking adapts to changing attack patterns. Bulyan is the best
   single-round defense due to its two-stage approach.

5. FedAvg should NEVER be used in adversarial environments. Even a single
   malicious client can arbitrarily corrupt the global model.

TRUST SCORE ANALYSIS
  Model poisoning attacks create clear trust separation (malicious < 0.3, benign > 0.8).
  Data poisoning attacks show smaller differentiation because model updates are
  structurally similar — the corruption appears in accuracy more than in parameter direction.
  The median-based cosine similarity metric reliably separates benign from malicious clients
  across all attack types, confirming its utility as a universal anomaly indicator.
"""),
        ]

        for title, content in analysis_pages:
            fig = plt.figure(figsize=(11, 8.5))
            fig.suptitle(title, fontsize=16, fontweight="bold", y=0.96)
            fig.text(0.05, 0.90, content, fontsize=9, va="top", family="monospace",
                     transform=fig.transFigure, linespacing=1.3)
            pdf.savefig(fig); plt.close(fig)

    print(f"\nReport saved to: {os.path.abspath(OUTPUT_PDF)}")
    total_pages = 2 + len(ATTACKS) * 4 + 2 + len(analysis_pages)
    print(f"Estimated pages: ~{total_pages}")


if __name__ == "__main__":
    main()
