"""Reproduce core results from Blanchard et al. 2017 (Krum) using FEDSIM.

Paper: "Machine learning with adversaries: Byzantine tolerant gradient descent"
       Blanchard, Guerraoui, Stainer et al., NeurIPS 2017.

Setup (matching paper Section 5):
  - MNIST, 10 classes
  - MLP with 2 hidden layers of 200 neurons (paper calls this "3-NN")
  - N=25 clients, f=6 Byzantine (24%)
  - IID data partition (equal-size splits)
  - E=1 local epoch per round
  - Attacks: None, Label Flipping, Byzantine Perturbation (Gaussian model replacement)

Extended with FEDSIM's full strategy suite and anomaly detection layer.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "fedsim"))


import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec
from simulation.runner import SimulationConfig, AttackConfig, run_simulation, RoundEvent

# ── Paper-matching configuration ──────────────────────────────────
MODEL = "mlp"
DATASET = "mnist"
NUM_CLIENTS = 25
NUM_MALICIOUS = 6  # f=6, ~24%
MALICIOUS_FRACTION = NUM_MALICIOUS / NUM_CLIENTS
NUM_ROUNDS = 50
LOCAL_EPOCHS = 1
BATCH_SIZE = 32
LR = 0.01
SEED = 42
PARTITION = "iid"

# MLP hidden size: paper uses 200 (not FEDSIM's default of 256)
MLP_HIDDEN_SIZE = 200

# Reputation hyperparameters
REP_TRUTH_THRESHOLD = 0.7
REP_SELECTION_FRACTION = 0.6
REP_INITIAL_REPUTATION = 0.5

# ── Attack configurations ─────────────────────────────────────────
ATTACKS = {
    "No Attack": AttackConfig(attack_type="none", malicious_fraction=0.0),
    "Label Flipping": AttackConfig(
        attack_type="label_flipping",
        malicious_fraction=MALICIOUS_FRACTION,
    ),
    "Byzantine\n(Gaussian)": AttackConfig(
        attack_type="byzantine_perturbation",
        malicious_fraction=MALICIOUS_FRACTION,
        attack_params={"noise_std": 1.0},
    ),
    "Gradient\nScaling": AttackConfig(
        attack_type="gradient_scaling",
        malicious_fraction=MALICIOUS_FRACTION,
        attack_params={"scale_factor": 10.0},
    ),
    "Weight\nSpiking": AttackConfig(
        attack_type="weight_spiking",
        malicious_fraction=MALICIOUS_FRACTION,
        attack_params={"magnitude": 100.0, "spike_fraction": 0.1},
    ),
}

# ── Strategy list ─────────────────────────────────────────────────
STRATEGIES = [
    "fedavg", "krum", "median", "trimmed_mean", "bulyan", "rfa", "reputation",
]

STRATEGY_LABELS = {
    "fedavg": "FedAvg",
    "krum": "Krum",
    "median": "Median",
    "trimmed_mean": "Trim. Mean",
    "bulyan": "Bulyan",
    "rfa": "RFA",
    "reputation": "Reputation",
}

STRATEGY_COLORS = {
    "fedavg": "#e74c3c",
    "krum": "#3498db",
    "median": "#9b59b6",
    "trimmed_mean": "#2ecc71",
    "bulyan": "#1abc9c",
    "rfa": "#e67e22",
    "reputation": "#f39c12",
}

OUTPUT_PDF = "krum_reproduction.pdf"


def run_experiment(attack_name, attack_config):
    """Run one attack scenario across all strategies."""
    config = SimulationConfig(
        model_name=MODEL,
        dataset_name=DATASET,
        num_clients=NUM_CLIENTS,
        num_rounds=NUM_ROUNDS,
        local_epochs=LOCAL_EPOCHS,
        learning_rate=LR,
        batch_size=BATCH_SIZE,
        partition_type=PARTITION,
        strategies=STRATEGIES,
        seed=SEED,
        attack=attack_config,
        reputation_truth_threshold=REP_TRUTH_THRESHOLD,
        reputation_selection_fraction=REP_SELECTION_FRACTION,
        reputation_initial_reputation=REP_INITIAL_REPUTATION,
        plugin_params={"models": {"hidden_size": MLP_HIDDEN_SIZE}},
    )

    print(f"  Running: {attack_name.replace(chr(10), ' ')} ...", end=" ", flush=True)
    t0 = time.time()
    results = run_simulation(config)
    elapsed = time.time() - t0
    print(f"done ({elapsed:.0f}s)")
    return results


def generate_report(all_results, output_path):
    """Generate a comprehensive PDF report."""

    attack_names = list(all_results.keys())
    n_attacks = len(attack_names)

    # ── Collect final metrics ─────────────────────────────────────
    # accuracy_matrix[attack][strategy] = final accuracy
    accuracy_matrix = {}
    for atk_name, results in all_results.items():
        accuracy_matrix[atk_name] = {}
        for r in results:
            accuracy_matrix[atk_name][r.strategy_name] = r.round_accuracies[-1] if r.round_accuracies else 0.0

    # Get no-attack baseline for delta computation
    baseline = accuracy_matrix.get("No Attack", {})

    with PdfPages(output_path) as pdf:

        # ══════════════════════════════════════════════════════════
        # PAGE 1: Title
        # ══════════════════════════════════════════════════════════
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.5, 0.78, "Byzantine-Robust FL Strategy Benchmark", fontsize=24,
                 ha="center", fontweight="bold")
        fig.text(0.5, 0.70, "Reproducing Blanchard et al. 2017 (Krum)", fontsize=16,
                 ha="center", style="italic", color="#555")
        fig.text(0.5, 0.62, "Extended with FEDSIM's Full Strategy Suite & Anomaly Detection",
                 fontsize=13, ha="center", color="#777")

        config_text = (
            f"Model: MLP (784 → {MLP_HIDDEN_SIZE} → {MLP_HIDDEN_SIZE} → 10)\n"
            f"Dataset: MNIST (10 classes, IID partition)\n"
            f"Clients: N={NUM_CLIENTS}, Byzantine: f={NUM_MALICIOUS} ({MALICIOUS_FRACTION:.0%})\n"
            f"Rounds: {NUM_ROUNDS} | Local Epochs: {LOCAL_EPOCHS} | Batch Size: {BATCH_SIZE}\n"
            f"Learning Rate: {LR} | Seed: {SEED}\n\n"
            f"Strategies: {', '.join(STRATEGY_LABELS[s] for s in STRATEGIES)}\n"
            f"Attacks: {', '.join(a.replace(chr(10), ' ') for a in attack_names)}\n\n"
            f"Paper: Blanchard, Guerraoui, Stainer et al.\n"
            f"'Machine learning with adversaries: Byzantine tolerant gradient descent'\n"
            f"NeurIPS 2017"
        )
        fig.text(0.5, 0.30, config_text, fontsize=11, ha="center", family="monospace",
                 va="center", linespacing=1.5)

        fig.text(0.5, 0.05, f"Generated by FEDSIM — {time.strftime('%Y-%m-%d %H:%M')}",
                 fontsize=9, ha="center", color="#999")
        pdf.savefig(fig); plt.close()

        # ══════════════════════════════════════════════════════════
        # PAGE 2: Convergence Curves (main result — like paper's Fig. 3)
        # ══════════════════════════════════════════════════════════
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle("Test Accuracy Over Rounds — All Strategies × All Attacks",
                     fontsize=16, fontweight="bold", y=0.98)
        axes_flat = axes.flatten()

        for idx, atk_name in enumerate(attack_names):
            if idx >= len(axes_flat):
                break
            ax = axes_flat[idx]
            results = all_results[atk_name]
            for r in results:
                label = STRATEGY_LABELS.get(r.strategy_name, r.strategy_name)
                color = STRATEGY_COLORS.get(r.strategy_name, "#888")
                ax.plot(range(len(r.round_accuracies)), r.round_accuracies,
                        linewidth=1.8, color=color, label=label, alpha=0.85)
            ax.set_title(atk_name.replace('\n', ' '), fontsize=12, fontweight="bold")
            ax.set_xlabel("Round")
            ax.set_ylabel("Test Accuracy")
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.3)

        # Hide unused subplot(s)
        for idx in range(n_attacks, len(axes_flat)):
            axes_flat[idx].set_visible(False)

        # Shared legend at the bottom
        handles = [plt.Line2D([0], [0], color=STRATEGY_COLORS[s], linewidth=2,
                              label=STRATEGY_LABELS[s]) for s in STRATEGIES]
        fig.legend(handles=handles, loc="lower center", ncol=len(STRATEGIES),
                   fontsize=10, frameon=True, fancybox=True)
        fig.tight_layout(rect=[0, 0.06, 1, 0.96])
        pdf.savefig(fig); plt.close()

        # ══════════════════════════════════════════════════════════
        # PAGE 3: Loss Curves
        # ══════════════════════════════════════════════════════════
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle("Training Loss Over Rounds — All Strategies × All Attacks",
                     fontsize=16, fontweight="bold", y=0.98)
        axes_flat = axes.flatten()

        for idx, atk_name in enumerate(attack_names):
            if idx >= len(axes_flat):
                break
            ax = axes_flat[idx]
            results = all_results[atk_name]
            for r in results:
                label = STRATEGY_LABELS.get(r.strategy_name, r.strategy_name)
                color = STRATEGY_COLORS.get(r.strategy_name, "#888")
                losses = r.round_losses
                # Clip extreme values for readability
                clipped = [min(l, 10.0) for l in losses]
                ax.plot(range(len(clipped)), clipped,
                        linewidth=1.8, color=color, label=label, alpha=0.85)
            ax.set_title(atk_name.replace('\n', ' '), fontsize=12, fontweight="bold")
            ax.set_xlabel("Round")
            ax.set_ylabel("Loss")
            ax.grid(True, alpha=0.3)

        for idx in range(n_attacks, len(axes_flat)):
            axes_flat[idx].set_visible(False)

        fig.legend(handles=handles, loc="lower center", ncol=len(STRATEGIES),
                   fontsize=10, frameon=True, fancybox=True)
        fig.tight_layout(rect=[0, 0.06, 1, 0.96])
        pdf.savefig(fig); plt.close()

        # ══════════════════════════════════════════════════════════
        # PAGE 4: Final Accuracy Table (heatmap)
        # ══════════════════════════════════════════════════════════
        fig, ax = plt.subplots(figsize=(14, 6))
        fig.suptitle("Final Test Accuracy — Strategy × Attack",
                     fontsize=16, fontweight="bold")

        table_data = []
        row_labels = [STRATEGY_LABELS[s] for s in STRATEGIES]
        col_labels = [a.replace('\n', ' ') for a in attack_names]

        for s in STRATEGIES:
            row = []
            for atk_name in attack_names:
                acc = accuracy_matrix[atk_name].get(s, 0.0)
                row.append(acc)
            table_data.append(row)

        table_arr = np.array(table_data)
        im = ax.imshow(table_arr, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, fontsize=11, fontweight="bold")
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=11, fontweight="bold")

        # Annotate cells
        for i in range(len(row_labels)):
            for j in range(len(col_labels)):
                val = table_arr[i, j]
                color = "white" if val < 0.4 else "black"
                ax.text(j, i, f"{val:.1%}", ha="center", va="center",
                        fontsize=12, fontweight="bold", color=color)

        fig.colorbar(im, ax=ax, label="Accuracy", shrink=0.8)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        pdf.savefig(fig); plt.close()

        # ══════════════════════════════════════════════════════════
        # PAGE 5: Robustness Delta (accuracy drop from No Attack baseline)
        # ══════════════════════════════════════════════════════════
        if baseline:
            fig, ax = plt.subplots(figsize=(14, 6))
            fig.suptitle("Robustness: Accuracy Delta from No-Attack Baseline",
                         fontsize=16, fontweight="bold")

            attack_labels_no_base = [a.replace('\n', ' ') for a in attack_names if a != "No Attack"]
            x = np.arange(len(attack_labels_no_base))
            width = 0.8 / len(STRATEGIES)

            for i, s in enumerate(STRATEGIES):
                base_acc = baseline.get(s, 0.0)
                deltas = []
                for atk_name in attack_names:
                    if atk_name == "No Attack":
                        continue
                    atk_acc = accuracy_matrix[atk_name].get(s, 0.0)
                    deltas.append(atk_acc - base_acc)
                offset = (i - len(STRATEGIES) / 2 + 0.5) * width
                bars = ax.bar(x + offset, deltas, width, label=STRATEGY_LABELS[s],
                              color=STRATEGY_COLORS[s], edgecolor="white", linewidth=0.5)

            ax.axhline(y=0, color="white", linewidth=0.8, linestyle="--")
            ax.set_xticks(x)
            ax.set_xticklabels(attack_labels_no_base, fontsize=11)
            ax.set_ylabel("Accuracy Delta (vs No Attack)", fontsize=12)
            ax.legend(fontsize=9, ncol=4)
            ax.grid(True, alpha=0.2, axis="y")
            fig.tight_layout(rect=[0, 0, 1, 0.93])
            pdf.savefig(fig); plt.close()

        # ══════════════════════════════════════════════════════════
        # PAGE 6: Anomaly Detection Metrics
        # ══════════════════════════════════════════════════════════
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle("Anomaly Detection: Removal F1, Precision, Recall (Under Attack Scenarios)",
                     fontsize=14, fontweight="bold")

        metric_keys = ["cumulative_f1", "cumulative_precision", "cumulative_recall"]
        metric_titles = ["Removal F1", "Precision", "Recall"]

        for midx, (mkey, mtitle) in enumerate(zip(metric_keys, metric_titles)):
            ax = axes[midx]
            attack_labels_no_base = [a.replace('\n', ' ') for a in attack_names if a != "No Attack"]
            x = np.arange(len(attack_labels_no_base))
            width = 0.8 / len(STRATEGIES)

            for i, s in enumerate(STRATEGIES):
                vals = []
                for atk_name in attack_names:
                    if atk_name == "No Attack":
                        continue
                    result = None
                    for r in all_results[atk_name]:
                        if r.strategy_name == s:
                            result = r
                            break
                    if result and result.anomaly_summary:
                        v = result.anomaly_summary.get(mkey, 0)
                        vals.append(v if v == v else 0.0)  # NaN guard
                    else:
                        vals.append(0.0)
                offset = (i - len(STRATEGIES) / 2 + 0.5) * width
                ax.bar(x + offset, vals, width, label=STRATEGY_LABELS[s],
                       color=STRATEGY_COLORS[s], edgecolor="white", linewidth=0.5)

            ax.set_title(mtitle, fontsize=12, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(attack_labels_no_base, fontsize=8, rotation=15)
            ax.set_ylim(0, 1.1)
            ax.grid(True, alpha=0.2, axis="y")

        axes[2].legend(fontsize=7, ncol=2)
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        pdf.savefig(fig); plt.close()

        # ══════════════════════════════════════════════════════════
        # PAGE 7: Reputation Trajectories (per-attack)
        # ══════════════════════════════════════════════════════════
        for atk_name in attack_names:
            if atk_name == "No Attack":
                continue
            rep_result = None
            for r in all_results[atk_name]:
                if r.strategy_name == "reputation":
                    rep_result = r
                    break
            if not rep_result or not rep_result.reputation_history:
                continue

            # Determine malicious clients
            malicious = set()
            for rnd_st in rep_result.client_statuses_history:
                for cid, status in rnd_st.items():
                    if status in ("attacked", "malicious_idle"):
                        malicious.add(cid)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            clean_atk_name = atk_name.replace('\n', ' ')
            fig.suptitle(f"Reputation Strategy — {clean_atk_name}",
                         fontsize=14, fontweight="bold")

            # Left: trust history
            if rep_result.trust_history:
                for cid in sorted(rep_result.trust_history.keys()):
                    vals = rep_result.trust_history[cid]
                    is_mal = cid in malicious
                    ax1.plot(range(1, len(vals) + 1), vals,
                             linewidth=1.5 if not is_mal else 2.5,
                             color="#e74c3c" if is_mal else "#2ecc71",
                             alpha=0.4 if not is_mal else 0.9,
                             linestyle="--" if is_mal else "-",
                             label=f"C{cid}*" if is_mal else None)
                ax1.set_title("Trust Scores Over Rounds")
                ax1.set_xlabel("Round"); ax1.set_ylabel("Trust")
                ax1.set_ylim(-0.05, 1.1)
                ax1.grid(True, alpha=0.3)
                # Only show legend for malicious clients
                mal_handles = [plt.Line2D([0], [0], color="#e74c3c", linewidth=2.5,
                               linestyle="--", label=f"Malicious (f={len(malicious)})"),
                               plt.Line2D([0], [0], color="#2ecc71", linewidth=1.5,
                               label=f"Benign (n={NUM_CLIENTS - len(malicious)})")]
                ax1.legend(handles=mal_handles, fontsize=9)

            # Right: reputation history
            for cid in sorted(rep_result.reputation_history.keys()):
                vals = rep_result.reputation_history[cid]
                is_mal = cid in malicious
                ax2.plot(range(1, len(vals) + 1), vals,
                         linewidth=1.5 if not is_mal else 2.5,
                         color="#e74c3c" if is_mal else "#2ecc71",
                         alpha=0.4 if not is_mal else 0.9,
                         linestyle="--" if is_mal else "-")
            ax2.set_title("Reputation Scores Over Rounds")
            ax2.set_xlabel("Round"); ax2.set_ylabel("Reputation")
            ax2.set_ylim(-0.05, 1.1)
            ax2.grid(True, alpha=0.3)
            ax2.legend(handles=mal_handles, fontsize=9)

            fig.tight_layout(rect=[0, 0, 1, 0.93])
            pdf.savefig(fig); plt.close()

        # ══════════════════════════════════════════════════════════
        # PAGE 8: Per-strategy convergence comparison (one page per strategy)
        # ══════════════════════════════════════════════════════════
        for s in STRATEGIES:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            sname = STRATEGY_LABELS[s]
            fig.suptitle(f"{sname} — Performance Across All Attack Scenarios",
                         fontsize=14, fontweight="bold")

            atk_colors = plt.cm.Set2(np.linspace(0, 1, n_attacks))
            for idx, atk_name in enumerate(attack_names):
                for r in all_results[atk_name]:
                    if r.strategy_name == s:
                        clean_label = atk_name.replace('\n', ' ')
                        ax1.plot(range(len(r.round_accuracies)), r.round_accuracies,
                                 linewidth=2, color=atk_colors[idx], label=clean_label)
                        clipped = [min(l, 10.0) for l in r.round_losses]
                        ax2.plot(range(len(clipped)), clipped,
                                 linewidth=2, color=atk_colors[idx], label=clean_label)
                        break

            ax1.set_title("Test Accuracy"); ax1.set_xlabel("Round"); ax1.set_ylabel("Accuracy")
            ax1.set_ylim(-0.05, 1.05); ax1.grid(True, alpha=0.3); ax1.legend(fontsize=9)
            ax2.set_title("Loss"); ax2.set_xlabel("Round"); ax2.set_ylabel("Loss")
            ax2.grid(True, alpha=0.3); ax2.legend(fontsize=9)

            fig.tight_layout(rect=[0, 0, 1, 0.93])
            pdf.savefig(fig); plt.close()

        # ══════════════════════════════════════════════════════════
        # PAGE LAST: Key Findings
        # ══════════════════════════════════════════════════════════
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle("Key Findings & Paper Comparison", fontsize=16, fontweight="bold", y=0.95)

        # Build findings text
        findings = "FINAL ACCURACY TABLE\n\n"
        header = f"{'Strategy':<15s}"
        for atk_name in attack_names:
            header += f"  {atk_name.replace(chr(10), ' '):<16s}"
        findings += header + "\n" + "=" * len(header) + "\n"

        for s in STRATEGIES:
            row = f"{STRATEGY_LABELS[s]:<15s}"
            for atk_name in attack_names:
                acc = accuracy_matrix[atk_name].get(s, 0.0)
                row += f"  {acc:<16.1%}"
            findings += row + "\n"

        findings += "\n\nPAPER CLAIMS (Blanchard et al. 2017):\n\n"
        findings += "  1. Krum achieves near-optimal accuracy under Byzantine attacks\n"
        findings += "     when n >= 2f + 3 (we have n=25, f=6: 25 >= 15 ✓)\n\n"
        findings += "  2. Krum outperforms coordinate-wise median under\n"
        findings += "     non-omniscient attacks\n\n"
        findings += "  3. FedAvg (simple averaging) degrades severely under\n"
        findings += "     Byzantine model poisoning\n\n"

        # Check claims against our data
        findings += "OUR RESULTS:\n\n"
        for atk_name in attack_names:
            if atk_name == "No Attack":
                continue
            clean_name = atk_name.replace('\n', ' ')
            krum_acc = accuracy_matrix[atk_name].get("krum", 0)
            fedavg_acc = accuracy_matrix[atk_name].get("fedavg", 0)
            median_acc = accuracy_matrix[atk_name].get("median", 0)
            best_s = max(STRATEGIES, key=lambda s: accuracy_matrix[atk_name].get(s, 0))
            best_acc = accuracy_matrix[atk_name].get(best_s, 0)
            findings += (f"  {clean_name}:\n"
                        f"    Krum={krum_acc:.1%}, FedAvg={fedavg_acc:.1%}, "
                        f"Median={median_acc:.1%}\n"
                        f"    Best: {STRATEGY_LABELS[best_s]} ({best_acc:.1%})\n\n")

        fig.text(0.05, 0.88, findings, fontsize=9, va="top", family="monospace",
                 transform=fig.transFigure, linespacing=1.4)
        pdf.savefig(fig); plt.close()


def main():
    print("=" * 70)
    print("  Krum Paper Reproduction — Byzantine-Robust FL Benchmark")
    print("  Blanchard et al. 2017 (NeurIPS)")
    print("=" * 70)
    print(f"  Model: MLP ({MLP_HIDDEN_SIZE} hidden) | Dataset: {DATASET.upper()}")
    print(f"  Clients: {NUM_CLIENTS} | Byzantine: {NUM_MALICIOUS} ({MALICIOUS_FRACTION:.0%})")
    print(f"  Rounds: {NUM_ROUNDS} | Epochs: {LOCAL_EPOCHS} | LR: {LR}")
    print(f"  Strategies: {', '.join(STRATEGY_LABELS[s] for s in STRATEGIES)}")
    print(f"  Attacks: {', '.join(a.replace(chr(10), ' ') for a in ATTACKS.keys())}")
    print()

    all_results = {}
    total_start = time.time()

    for atk_name, atk_config in ATTACKS.items():
        all_results[atk_name] = run_experiment(atk_name, atk_config)

    total_elapsed = time.time() - total_start
    print(f"\nAll experiments complete in {total_elapsed:.0f}s ({total_elapsed/60:.1f}m)")
    print(f"\nGenerating report...")

    generate_report(all_results, OUTPUT_PDF)
    print(f"Report saved to: {os.path.abspath(OUTPUT_PDF)}")

    # Print summary table to console
    print(f"\n{'Strategy':<15s}", end="")
    for atk_name in ATTACKS.keys():
        print(f"  {atk_name.replace(chr(10), ' '):<16s}", end="")
    print()
    print("=" * (15 + 18 * len(ATTACKS)))
    for s in STRATEGIES:
        print(f"{STRATEGY_LABELS[s]:<15s}", end="")
        for atk_name in ATTACKS.keys():
            for r in all_results[atk_name]:
                if r.strategy_name == s:
                    acc = r.round_accuracies[-1] if r.round_accuracies else 0
                    print(f"  {acc:<16.1%}", end="")
                    break
        print()


if __name__ == "__main__":
    main()
