"""Reproduce Experiment 2 from the capstone report:
'Exploring the Robustness of Federated Learning for Image Classification' (Abhik Roy)

Experiment 2: FedAvg vs Reputation under label flipping with 1, 2, and 3 poisoned clients.
- 6 clients, 15 rounds
- Dataset: CIFAR-10 (similar to the traffic sign dataset used in the paper)
- Model: CNN (small, as in the paper)
- Reputation selects top 4 of 6 clients (67% selection)
- Label flipping attack via bijective derangement
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
from simulation.runner import SimulationConfig, AttackConfig, run_simulation, RoundEvent

# ── Configuration matching capstone paper ──────────────────────────
MODEL = "cnn"
DATASET = "cifar10"
NUM_CLIENTS = 6
NUM_ROUNDS = 15
LOCAL_EPOCHS = 3
LR = 0.01
SEED = 42

# Reputation params matching paper: select top 4 of 6 = 67%
REP_SELECTION_FRACTION = 0.67
REP_TRUTH_THRESHOLD = 0.7
REP_INITIAL_REPUTATION = 0.5

# Three scenarios: 1, 2, 3 poisoned clients out of 6
SCENARIOS = [
    {"name": "1 Poisoned Client", "malicious_fraction": 1/6},
    {"name": "2 Poisoned Clients", "malicious_fraction": 2/6},
    {"name": "3 Poisoned Clients", "malicious_fraction": 3/6},
]

STRATEGIES = ["fedavg", "reputation"]
STRAT_LABELS = {"fedavg": "FL without Reputation", "reputation": "FL with Reputation"}
STRAT_COLORS = {"fedavg": "#3274A1", "reputation": "#E1812C"}

OUTPUT_PDF = "capstone_reproduction.pdf"


def run_scenario(malicious_fraction, scenario_name):
    """Run one scenario with both FedAvg and Reputation."""
    config = SimulationConfig(
        model_name=MODEL, dataset_name=DATASET,
        num_clients=NUM_CLIENTS, num_rounds=NUM_ROUNDS,
        local_epochs=LOCAL_EPOCHS, learning_rate=LR,
        partition_type="iid", alpha=0.5,
        strategies=STRATEGIES, seed=SEED,
        attack=AttackConfig(
            attack_type="label_flipping",
            malicious_fraction=malicious_fraction,
        ),
        reputation_truth_threshold=REP_TRUTH_THRESHOLD,
        reputation_selection_fraction=REP_SELECTION_FRACTION,
        reputation_initial_reputation=REP_INITIAL_REPUTATION,
    )

    all_client_acc = {}
    all_reputation = {}
    malicious_set = set()

    def round_cb(e: RoundEvent):
        s = e.strategy_name
        if s not in all_client_acc:
            all_client_acc[s] = {}
            all_reputation[s] = {}
        for cid, st in e.client_statuses.items():
            if st in ("attacked", "malicious_idle"):
                malicious_set.add(cid)
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

    return results, all_client_acc, all_reputation, malicious_set


def main():
    print("=" * 60)
    print("Capstone Experiment 2 Reproduction")
    print("FedAvg vs Reputation under Label Flipping")
    print("=" * 60)
    print(f"Model: {MODEL} | Dataset: {DATASET} | Clients: {NUM_CLIENTS}")
    print(f"Rounds: {NUM_ROUNDS} | Epochs: {LOCAL_EPOCHS} | LR: {LR}")
    print(f"Reputation: selection={REP_SELECTION_FRACTION:.0%}, "
          f"threshold={REP_TRUTH_THRESHOLD}, init={REP_INITIAL_REPUTATION}")
    print()

    all_scenarios = []
    for scenario in SCENARIOS:
        results, client_acc, reputation, malicious = run_scenario(
            scenario["malicious_fraction"], scenario["name"]
        )
        all_scenarios.append({
            "name": scenario["name"],
            "results": results,
            "client_acc": client_acc,
            "reputation": reputation,
            "malicious": malicious,
        })

    print(f"\nGenerating PDF report...")

    with PdfPages(OUTPUT_PDF) as pdf:
        # Title page
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.5, 0.75, "Capstone Experiment 2 Reproduction", fontsize=24, ha="center", fontweight="bold")
        fig.text(0.5, 0.65, "FedAvg vs Reputation under Label Flipping", fontsize=18, ha="center")
        fig.text(0.5, 0.50,
                 f"Model: {MODEL.upper()} | Dataset: {DATASET.upper()}\n"
                 f"Clients: {NUM_CLIENTS} | Rounds: {NUM_ROUNDS} | Epochs: {LOCAL_EPOCHS}\n"
                 f"Partition: IID | Attack: Label Flipping\n"
                 f"Reputation: top {int(NUM_CLIENTS * REP_SELECTION_FRACTION)} of {NUM_CLIENTS} "
                 f"selected per round\n\n"
                 f"Scenarios: 1, 2, 3 poisoned clients out of {NUM_CLIENTS}",
                 fontsize=12, ha="center", family="monospace")
        fig.text(0.5, 0.30,
                 "Reproducing results from:\n"
                 "'Exploring the Robustness of Federated Learning\n"
                 "for Image Classification' (Abhik Roy, RIT)",
                 fontsize=11, ha="center", style="italic")
        pdf.savefig(fig); plt.close()

        # Per-scenario pages
        for scenario in all_scenarios:
            name = scenario["name"]
            results = scenario["results"]
            client_acc = scenario["client_acc"]
            reputation = scenario["reputation"]
            malicious = scenario["malicious"]

            # ── Page: Per-client accuracy comparison (bar chart like Fig 3/4/5) ──
            fig, axes = plt.subplots(2, 3, figsize=(14, 8))
            fig.suptitle(f"Model Accuracies — {name}", fontsize=16, fontweight="bold")

            for cid in range(NUM_CLIENTS):
                ax = axes[cid // 3][cid % 3]
                is_mal = cid in malicious

                # Get final accuracy for each strategy
                accs = {}
                for s in STRATEGIES:
                    if s in client_acc and cid in client_acc[s] and client_acc[s][cid]:
                        accs[s] = client_acc[s][cid][-1]  # last round accuracy
                    else:
                        accs[s] = 0.0

                x = np.arange(len(STRATEGIES))
                bars = ax.bar(x, [accs[s] for s in STRATEGIES],
                              color=[STRAT_COLORS[s] for s in STRATEGIES],
                              edgecolor="white", linewidth=0.5)

                ax.set_title(f"Client {cid}" + (" (poisoned)" if is_mal else ""),
                             fontsize=10, fontweight="bold" if is_mal else "normal",
                             color="#e74c3c" if is_mal else "black")
                ax.set_xticks(x)
                ax.set_xticklabels([STRAT_LABELS[s] for s in STRATEGIES], fontsize=7, rotation=15)
                ax.set_ylim(0, 1.05)
                ax.set_ylabel("Accuracy (%)" if cid % 3 == 0 else "")

                # Add value labels on bars
                for bar, val in zip(bars, [accs[s] for s in STRATEGIES]):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                            f"{val:.0%}", ha="center", fontsize=8)

            # Add legend to last subplot
            ax = axes[1][2]
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor=STRAT_COLORS["fedavg"], label=STRAT_LABELS["fedavg"]),
                Patch(facecolor=STRAT_COLORS["reputation"], label=STRAT_LABELS["reputation"]),
            ]
            ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

            fig.tight_layout(rect=[0, 0, 1, 0.95])
            pdf.savefig(fig); plt.close()

            # ── Page: Reputation scores over rounds (like Fig 6/7/8) ──
            if "reputation" in reputation and reputation["reputation"]:
                fig, ax = plt.subplots(1, 1, figsize=(10, 6))
                fig.suptitle(f"Reputation Scores Over Rounds — {name}",
                             fontsize=16, fontweight="bold")

                for cid in range(NUM_CLIENTS):
                    if cid in reputation["reputation"]:
                        vals = reputation["reputation"][cid]
                        is_mal = cid in malicious
                        ax.plot(range(1, len(vals) + 1), vals,
                                "o-", linewidth=2, markersize=4,
                                label=f"Client {cid}" + (" (poisoned)" if is_mal else ""),
                                color="#e74c3c" if is_mal else None,
                                linestyle="--" if is_mal else "-")

                ax.set_xlabel("Rounds", fontsize=12)
                ax.set_ylabel("Reputation", fontsize=12)
                ax.set_ylim(-0.05, 1.1)
                ax.set_xticks(range(1, NUM_ROUNDS + 1, 2))
                ax.legend(fontsize=9, ncol=2)
                ax.grid(True, alpha=0.3)

                fig.tight_layout(rect=[0, 0, 1, 0.95])
                pdf.savefig(fig); plt.close()

            # ── Page: Global accuracy comparison ──
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle(f"Global Model Performance — {name}", fontsize=16, fontweight="bold")

            for r in results:
                label = STRAT_LABELS.get(r.strategy_name, r.strategy_name)
                color = STRAT_COLORS.get(r.strategy_name, "gray")
                rounds = list(range(len(r.round_losses)))
                ax1.plot(rounds, r.round_losses, "o-", label=label, color=color, markersize=4, linewidth=2)
                ax2.plot(rounds, r.round_accuracies, "o-", label=label, color=color, markersize=4, linewidth=2)

            ax1.set_xlabel("Round"); ax1.set_ylabel("Loss")
            ax1.set_title("Aggregated Loss"); ax1.legend(); ax1.grid(True, alpha=0.3)
            ax2.set_xlabel("Round"); ax2.set_ylabel("Accuracy")
            ax2.set_title("Global Accuracy"); ax2.set_ylim(0, 1.05)
            ax2.legend(); ax2.grid(True, alpha=0.3)

            fig.tight_layout(rect=[0, 0, 1, 0.95])
            pdf.savefig(fig); plt.close()

        # ── Summary page ──
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle("Summary: Final Accuracy by Scenario", fontsize=16, fontweight="bold", y=0.95)

        # Bar chart comparing FedAvg vs Reputation across scenarios
        ax = fig.add_subplot(111)
        x = np.arange(len(SCENARIOS))
        w = 0.35

        fedavg_accs = []
        rep_accs = []
        for scenario in all_scenarios:
            for r in scenario["results"]:
                final_acc = r.round_accuracies[-1]
                if r.strategy_name == "fedavg":
                    fedavg_accs.append(final_acc)
                elif r.strategy_name == "reputation":
                    rep_accs.append(final_acc)

        bars1 = ax.bar(x - w/2, fedavg_accs, w, label=STRAT_LABELS["fedavg"],
                       color=STRAT_COLORS["fedavg"])
        bars2 = ax.bar(x + w/2, rep_accs, w, label=STRAT_LABELS["reputation"],
                       color=STRAT_COLORS["reputation"])

        for b, v in zip(bars1, fedavg_accs):
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                    f"{v:.1%}", ha="center", fontsize=10)
        for b, v in zip(bars2, rep_accs):
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                    f"{v:.1%}", ha="center", fontsize=10)

        ax.set_xticks(x)
        ax.set_xticklabels([s["name"] for s in SCENARIOS], fontsize=11)
        ax.set_ylabel("Final Global Accuracy", fontsize=12)
        ax.set_ylim(0, 1.15)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3, axis="y")

        fig.tight_layout(rect=[0, 0, 1, 0.93])
        pdf.savefig(fig); plt.close()

        # ── Key findings page ──
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle("Key Findings", fontsize=16, fontweight="bold", y=0.95)

        findings = (
            f"EXPERIMENT 2 REPRODUCTION: FedAvg vs Reputation under Label Flipping\n\n"
            f"Setup: {NUM_CLIENTS} clients, {NUM_ROUNDS} rounds, {LOCAL_EPOCHS} epochs, "
            f"IID partition, CIFAR-10\n"
            f"Reputation selects top {int(NUM_CLIENTS * REP_SELECTION_FRACTION)} of "
            f"{NUM_CLIENTS} clients per round\n\n"
            f"RESULTS:\n\n"
            f"  Scenario           FedAvg Acc    Reputation Acc    Improvement\n"
            f"  {'='*60}\n"
        )
        for i, scenario in enumerate(all_scenarios):
            fa = fedavg_accs[i]
            ra = rep_accs[i]
            diff = ra - fa
            findings += f"  {scenario['name']:20s} {fa:>10.1%}    {ra:>14.1%}    {diff:>+10.1%}\n"

        findings += (
            f"\n\nCAPSTONE PAPER CLAIMS vs OUR RESULTS:\n\n"
            f"  1. 'FL model with Reputation outperforms the standard FL model\n"
            f"     on each client dataset except the poisoned dataset'\n"
            f"     -> Check per-client accuracy bar charts above\n\n"
            f"  2. 'Direct correlation between the decrease in accuracy of the\n"
            f"     standard model, and the increase in poisoned clients'\n"
            f"     -> Compare FedAvg accuracy across 1, 2, 3 poisoned scenarios\n\n"
            f"  3. 'Accuracy of the FL model with reputation is almost unchanged\n"
            f"     with additional poisoned clients'\n"
            f"     -> Compare Reputation accuracy across scenarios\n\n"
            f"  4. 'Reputations of all poisoned clients either start at 0 or\n"
            f"     eventually go to 0'\n"
            f"     -> Check reputation score plots above\n"
        )

        fig.text(0.05, 0.88, findings, fontsize=10, va="top", family="monospace",
                 transform=fig.transFigure, linespacing=1.3)
        pdf.savefig(fig); plt.close()

    print(f"\nReport saved to: {os.path.abspath(OUTPUT_PDF)}")


if __name__ == "__main__":
    main()
