#!/usr/bin/env python3
"""Capstone Replication: FedAvg vs Reputation under Label Flipping.

Replicates Experiment 2 from Roy's capstone paper using TextCNN on AG News
instead of CNN on CIFAR-10. Tests whether reputation-based aggregation
outperforms FedAvg when nearly half the clients are poisoned.

Setup: 12 clients, 5 poisoned (42%), label flipping attack, 15 rounds.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fedsim"))

from api import Experiment, Report
from simulation.runner import SimulationConfig, AttackConfig

BASE = dict(
    model_name="custom:TextCNN",
    dataset_name="custom:AG News (4-class)",
    num_clients=12,
    num_rounds=15,
    local_epochs=3,
    learning_rate=0.01,
    batch_size=32,
    seed=42,
    partition_type="iid",
    use_amp=True,
    pin_memory=True,
    plugin_params={"models": {"embed_dim": 128, "num_filters": 100}},
)

ATTACK = AttackConfig(
    attack_type="label_flipping",
    malicious_fraction=5/12,  # 5 of 12 clients
)

exp = Experiment("Capstone Replication — TextCNN / AG News")

# FedAvg under attack
exp.add_run("FedAvg (attacked)", SimulationConfig(
    **BASE,
    strategies=["fedavg"],
    attack=ATTACK,
))

# Reputation under attack
exp.add_run("Reputation (attacked)", SimulationConfig(
    **BASE,
    strategies=["reputation"],
    attack=ATTACK,
))

# Clean baseline (no attack) for reference
exp.add_run("FedAvg (clean)", SimulationConfig(
    **BASE,
    strategies=["fedavg"],
    attack=AttackConfig(),
))

# Export configs for dashboard execution, OR run directly
import sys
if "--export" in sys.argv:
    exp.export_configs("results/capstone_textcnn_configs.json")
    print(f"Exported {len(exp._runs)} run configs to results/capstone_textcnn_configs.json")
    print("Upload this file to the dashboard via Load Experiment.")
    sys.exit(0)

results = exp.run(checkpoint_path="results/capstone_textcnn.json")

# Report
report = Report("Capstone Replication: FedAvg vs Reputation — TextCNN on AG News")

report.add_text("Setup",
    "Replicates Experiment 2 from Roy's capstone paper.\n\n"
    "Model: TextCNN (Kim 2014) — embedding + 3 parallel Conv1d branches\n"
    "Dataset: AG News (4-class text classification, 120K samples)\n"
    "Clients: 12 total, 5 poisoned (42% malicious)\n"
    "Attack: Label flipping (bijective derangement)\n"
    "Rounds: 15, Local epochs: 3, LR: 0.01\n"
    "Partition: IID\n\n"
    "Hypothesis: Reputation-based aggregation should maintain higher accuracy\n"
    "than FedAvg under label flipping by excluding poisoned clients.")

report.add_convergence_plot(results,
    names=["FedAvg (clean)", "FedAvg (attacked)", "Reputation (attacked)"],
    title="Accuracy Convergence: Clean vs Attacked",
    colors={
        "FedAvg (clean)": "#2ca02c",
        "FedAvg (attacked)": "#d62728",
        "Reputation (attacked)": "#9467bd",
    })

fa_clean = results.final_accuracy("FedAvg (clean)")
fa_attack = results.final_accuracy("FedAvg (attacked)")
rep_attack = results.final_accuracy("Reputation (attacked)")

report.add_text("Results",
    f"Final Accuracies:\n"
    f"  FedAvg (clean):    {fa_clean:.4f}\n"
    f"  FedAvg (attacked): {fa_attack:.4f}  (Δ = {fa_attack - fa_clean:+.4f})\n"
    f"  Reputation (attacked): {rep_attack:.4f}  (Δ = {rep_attack - fa_clean:+.4f})\n\n"
    f"Reputation advantage over FedAvg under attack: {rep_attack - fa_attack:+.4f}\n\n"
    f"Capstone paper finding: Reputation improved accuracy by +11.2% with 50% poisoned\n"
    f"clients on CNN/CIFAR-10. This experiment tests the same hypothesis on a text\n"
    f"classification task with 42% poisoned clients.")

report.add_heatmap(
    [[fa_clean, fa_attack, rep_attack]],
    row_labels=["Final Accuracy"],
    col_labels=["FedAvg (clean)", "FedAvg (attacked)", "Reputation (attacked)"],
    title="Strategy Comparison")

report.save_pdf("capstone_textcnn_report.pdf")

print(f"\nFedAvg (clean):       {fa_clean:.4f}")
print(f"FedAvg (attacked):    {fa_attack:.4f}")
print(f"Reputation (attacked): {rep_attack:.4f}")
print(f"Reputation advantage:  {rep_attack - fa_attack:+.4f}")
print(f"\nReport: capstone_textcnn_report.pdf")
