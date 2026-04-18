#!/usr/bin/env python3
"""Strategy Comparison Experiments — demonstrates differences between FL aggregation strategies.

Three experiments:
  1. Label Flipping Attack (data poisoning) — CNN/CIFAR-10, 12 clients, 20% malicious
  2. Weight Spiking Attack (model poisoning) — CNN/CIFAR-10, 12 clients, 20% malicious
  3. No Attack Baseline — same setup, no adversary

Each experiment runs all 6 strategies: FedAvg, Krum, Trimmed Mean, Median, Bulyan, RFA.
Results are saved to experiments/results/ as JSON checkpoints and a PDF report.
"""

import sys
import os

# Add fedsim to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.runner import SimulationConfig, AttackConfig
from api.experiment import Experiment
from api.report import Report

# ── Common config ────────────────────────────────────────────────────────────
STRATEGIES = ["fedavg", "krum", "trimmed_mean", "median", "bulyan", "rfa"]
NUM_CLIENTS = 12
NUM_ROUNDS = 10
LOCAL_EPOCHS = 3
SEED = 42
MODEL = "cnn"
DATASET = "cifar10"
MALICIOUS_FRACTION = 0.2   # f=2 out of 12 → Bulyan needs n>=11, have 12 ✓

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def make_config(attack_type="none", attack_params=None, strategies=None):
    """Build a SimulationConfig with the given attack and strategy list."""
    return SimulationConfig(
        model_name=MODEL,
        dataset_name=DATASET,
        num_clients=NUM_CLIENTS,
        num_rounds=NUM_ROUNDS,
        local_epochs=LOCAL_EPOCHS,
        learning_rate=0.01,
        partition_type="non_iid",
        alpha=0.5,
        val_split=0.1,
        strategies=strategies or STRATEGIES,
        batch_size=32,
        seed=SEED,
        attack=AttackConfig(
            attack_type=attack_type,
            malicious_fraction=MALICIOUS_FRACTION if attack_type != "none" else 0.0,
            attack_params=attack_params or {},
        ),
    )


def run_experiments():
    # ── Experiment 1: Label Flipping (data poisoning) ────────────────────────
    print("\n" + "=" * 70)
    print("  EXPERIMENT 1: Label Flipping Attack")
    print("=" * 70)

    exp1 = Experiment("Label Flipping — Strategy Comparison")
    for strat in STRATEGIES:
        exp1.add_run(
            f"label_flip/{strat}",
            make_config(attack_type="label_flipping", strategies=[strat]),
        )
    ckpt1 = os.path.join(RESULTS_DIR, "exp1_label_flipping.json")
    results1 = exp1.run(checkpoint_path=ckpt1)

    # ── Experiment 2: Weight Spiking (model poisoning) ───────────────────────
    print("\n" + "=" * 70)
    print("  EXPERIMENT 2: Weight Spiking Attack")
    print("=" * 70)

    exp2 = Experiment("Weight Spiking — Strategy Comparison")
    for strat in STRATEGIES:
        exp2.add_run(
            f"weight_spike/{strat}",
            make_config(
                attack_type="weight_spiking",
                attack_params={"magnitude": 100.0, "spike_fraction": 0.1},
                strategies=[strat],
            ),
        )
    ckpt2 = os.path.join(RESULTS_DIR, "exp2_weight_spiking.json")
    results2 = exp2.run(checkpoint_path=ckpt2)

    # ── Experiment 3: No Attack Baseline ─────────────────────────────────────
    print("\n" + "=" * 70)
    print("  EXPERIMENT 3: No Attack Baseline")
    print("=" * 70)

    exp3 = Experiment("No Attack Baseline")
    for strat in STRATEGIES:
        exp3.add_run(
            f"baseline/{strat}",
            make_config(attack_type="none", strategies=[strat]),
        )
    ckpt3 = os.path.join(RESULTS_DIR, "exp3_baseline.json")
    results3 = exp3.run(checkpoint_path=ckpt3)

    # ── Build PDF Report ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  Generating PDF Report")
    print("=" * 70)

    report = Report("FEDSIM Strategy Comparison Report")

    report.add_text(
        "Experiment Overview",
        "This report compares 6 FL aggregation strategies under different attack scenarios.\n\n"
        f"Setup: {MODEL.upper()}/{DATASET.upper()}, {NUM_CLIENTS} clients, "
        f"{NUM_ROUNDS} rounds, {LOCAL_EPOCHS} local epochs, "
        f"non-IID (alpha=0.5), {int(MALICIOUS_FRACTION*100)}% malicious.\n\n"
        "Strategies: FedAvg (no defense), Krum (nearest-neighbor selection), "
        "Trimmed Mean (coordinate-wise trimming), Median (coordinate-wise median), "
        "Bulyan (Krum + trimmed mean), RFA (geometric median).\n\n"
        "Attacks:\n"
        "  1. Label Flipping — permutes class labels for malicious clients\n"
        "  2. Weight Spiking — multiplies 10% of weights by 100x after training\n"
        "  3. No Attack — clean baseline for convergence comparison"
    )

    # Convergence plots per experiment
    report.add_convergence_plot(
        results1, [f"label_flip/{s}" for s in STRATEGIES],
        title="Exp 1: Label Flipping — Loss & Accuracy",
    )
    report.add_convergence_plot(
        results2, [f"weight_spike/{s}" for s in STRATEGIES],
        title="Exp 2: Weight Spiking — Loss & Accuracy",
    )
    report.add_convergence_plot(
        results3, [f"baseline/{s}" for s in STRATEGIES],
        title="Exp 3: No Attack Baseline — Loss & Accuracy",
    )

    # Summary tables
    report.add_accuracy_table(results1, [f"label_flip/{s}" for s in STRATEGIES])
    report.add_accuracy_table(results2, [f"weight_spike/{s}" for s in STRATEGIES])
    report.add_accuracy_table(results3, [f"baseline/{s}" for s in STRATEGIES])

    # Accuracy heatmap: strategies × attacks
    heatmap_data = []
    attack_names = ["Label Flipping", "Weight Spiking", "No Attack"]
    for strat in STRATEGIES:
        row = [
            results1.final_accuracy(f"label_flip/{strat}"),
            results2.final_accuracy(f"weight_spike/{strat}"),
            results3.final_accuracy(f"baseline/{strat}"),
        ]
        heatmap_data.append(row)

    report.add_heatmap(
        heatmap_data,
        row_labels=[s.replace("_", " ").title() for s in STRATEGIES],
        col_labels=attack_names,
        title="Final Accuracy: Strategies × Attacks",
    )

    pdf_path = os.path.join(RESULTS_DIR, "strategy_comparison_report.pdf")
    report.save_pdf(pdf_path)
    print(f"\nReport saved to: {pdf_path}")

    # ── Print Summary Table ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n{'Strategy':<16} {'Label Flip':>12} {'Wt Spike':>12} {'Baseline':>12}")
    print("-" * 54)
    for strat in STRATEGIES:
        a1 = results1.final_accuracy(f"label_flip/{strat}")
        a2 = results2.final_accuracy(f"weight_spike/{strat}")
        a3 = results3.final_accuracy(f"baseline/{strat}")
        print(f"{strat:<16} {a1:>11.4f} {a2:>11.4f} {a3:>11.4f}")
    print()


if __name__ == "__main__":
    run_experiments()
