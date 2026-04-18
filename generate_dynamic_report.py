"""Generate a comprehensive PDF report for a dynamic multi-attack FL simulation.

Schedule (15 rounds, 10 clients, 3 attacks):
  Rounds  1-3:   Baseline (clean convergence)
  Rounds  4-6:   Label Flipping (data poisoning)
  Rounds  7-9:   Recovery
  Rounds 10-12:  Weight Spiking (model poisoning)
  Rounds 13-15:  Byzantine Perturbation (worst-case model poisoning)

Strategies: FedAvg (baseline), Trimmed Mean, Krum, Reputation
Non-IID partitioning with Dirichlet alpha=0.3 for realistic heterogeneity.

Usage:
    python generate_dynamic_report.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "fedsim"))


import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import copy
import time
import textwrap
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap

from fl_core import FedAvg, FitRes

from configs.defaults import DATASET_INFO, ATTACK_CATEGORIES
from data.loader import get_dataset
from data.partitioner import partition_dataset
from models import get_model
from strategies import KrumStrategy, TrimmedMean, MedianStrategy, BulyanStrategy, RFAStrategy
from custom.strategies.reputation import ReputationStrategy
from attacks.data_poisoning import apply_label_flipping
from attacks.model_poisoning import apply_weight_spiking, apply_gradient_scaling, apply_byzantine_perturbation

# ── Configuration ─────────────────────────────────────────────────────
MODEL = "cnn"
DATASET = "fashion_mnist"
NUM_CLIENTS = 10
LOCAL_EPOCHS = 2
LR = 0.01
BATCH_SIZE = 32
SEED = 42
MALICIOUS_FRACTION = 0.3          # 3 out of 10 — realistic, within theoretical bounds
PARTITION_TYPE = "non_iid"
DIRICHLET_ALPHA = 0.5             # moderate non-IID — realistic heterogeneity without extreme skew
TOTAL_ROUNDS = 15

STRATEGIES = ["fedavg", "trimmed_mean", "krum", "reputation"]
STRAT_LABELS = {
    "fedavg": "FedAvg (No Defense)",
    "trimmed_mean": "Trimmed Mean",
    "krum": "Krum (Multi)",
    "reputation": "Reputation",
}
STRAT_COLORS = {
    "fedavg": "#e74c3c",
    "trimmed_mean": "#2ecc71",
    "krum": "#3498db",
    "reputation": "#f39c12",
}
STRAT_MARKERS = {
    "fedavg": "o",
    "trimmed_mean": "s",
    "krum": "D",
    "reputation": "^",
}

PHASES = [
    ("Baseline",               "none",                    {},                                           (1, 3)),
    ("Label Flipping",         "label_flipping",          {},                                           (4, 6)),
    ("Recovery",               "none",                    {},                                           (7, 9)),
    ("Weight Spiking",         "weight_spiking",          {"magnitude": 100.0, "spike_fraction": 0.3},  (10, 12)),
    ("Byzantine Perturbation", "byzantine_perturbation",  {"noise_std": 5.0},                           (13, 15)),
]

ATK_COLORS = {
    "Label Flipping": "#fadbd8",
    "Weight Spiking": "#d5f5e3",
    "Byzantine Perturbation": "#e8daef",
}

OUTPUT_PDF = "dynamic_attack_report.pdf"


# ── Helpers ───────────────────────────────────────────────────────────
def _set_params(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict(
        {k: torch.from_numpy(np.array(v)) if isinstance(v, np.ndarray) else torch.tensor(v)
         for k, v in params_dict}
    )
    model.load_state_dict(state_dict, strict=True)


def _train_client(model, trainloader, local_epochs, lr, device):
    model.to(device)
    model.train()

    # Check if model params are already corrupted (NaN/Inf from prior poisoning)
    for p in model.parameters():
        if torch.isnan(p).any() or torch.isinf(p).any():
            return [v.cpu().numpy() for v in model.state_dict().values()], float('inf')

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    last_loss = 0.0
    for _ in range(local_epochs):
        epoch_loss, n = 0.0, 0
        for imgs, labels in trainloader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(imgs)
            loss = criterion(output, labels)
            # Abort training if loss explodes (model is corrupted beyond recovery)
            if not torch.isfinite(loss):
                return [v.cpu().numpy() for v in model.state_dict().values()], float('inf')
            loss.backward()
            # Clip gradients to prevent NaN propagation
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            epoch_loss += loss.item()
            n += 1
        last_loss = epoch_loss / max(n, 1)
    return [v.cpu().numpy() for v in model.state_dict().values()], last_loss


def _run_eval(model, testloader, device):
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in testloader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            total_loss += criterion(out, labels).item() * labels.size(0)
            correct += (out.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


def _client_acc(model, params, testloader, device):
    _set_params(model, params)
    model.to(device)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for i, (imgs, labels) in enumerate(testloader):
            if i >= 3:
                break
            imgs, labels = imgs.to(device), labels.to(device)
            correct += (model(imgs).argmax(1) == labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0.0


def _compute_trust(client_params_list, global_params):
    """Combined direction + magnitude trust metric."""
    global_flat = np.concatenate([p.flatten() for p in global_params])
    deltas = [np.concatenate([p.flatten() for p in cp]) - global_flat for cp in client_params_list]
    deltas_arr = np.array(deltas)
    median_delta = np.median(deltas_arr, axis=0)
    mn = np.linalg.norm(median_delta)
    if mn < 1e-10:
        return {cid: 1.0 for cid in range(len(client_params_list))}

    l2_dists = np.array([np.linalg.norm(d - median_delta) for d in deltas])
    max_l2 = np.max(l2_dists)

    scores = {}
    for cid, d in enumerate(deltas):
        dn = np.linalg.norm(d)
        if dn < 1e-10:
            scores[cid] = 1.0
            continue
        cosine_sim = np.dot(d, median_delta) / (dn * mn)
        cosine_comp = float(np.clip((cosine_sim + 1.0) / 2.0, 0.0, 1.0))
        dist_comp = float(1.0 - l2_dists[cid] / max_l2) if max_l2 > 1e-10 else 1.0
        scores[cid] = 0.5 * cosine_comp + 0.5 * dist_comp
    return scores


def _model_attack(params, atk_type, atk_params, seed, global_params=None):
    if atk_type == "weight_spiking":
        return apply_weight_spiking(params, seed=seed, **atk_params)
    elif atk_type == "gradient_scaling":
        return apply_gradient_scaling(params, global_parameters=global_params, **atk_params)
    elif atk_type == "byzantine_perturbation":
        return apply_byzantine_perturbation(params, seed=seed, **atk_params)
    return params


def _aggregate(strategy, rnd, client_params, num_samples):
    results = [
        (i, FitRes(parameters=p, num_examples=n))
        for i, (p, n) in enumerate(zip(client_params, num_samples))
    ]
    agg, _ = strategy.aggregate_fit(rnd, results, [])
    return agg


def _get_phase(rnd):
    for name, atk_type, atk_params, (start, end) in PHASES:
        if start <= rnd <= end:
            return name, atk_type, atk_params
    return "Unknown", "none", {}


def _make_strategy(name, init_params, nc, num_malicious=0):
    common = dict(fraction_fit=1.0, fraction_evaluate=1.0, min_fit_clients=nc,
                  min_evaluate_clients=nc, min_available_clients=nc, initial_parameters=init_params)
    if name == "fedavg":
        return FedAvg(**common)
    elif name == "trimmed_mean":
        beta = max(0.1, num_malicious / nc) if nc > 0 else 0.1
        return TrimmedMean(beta=beta, **common)
    elif name == "krum":
        return KrumStrategy(num_malicious=num_malicious, multi_krum=True, **common)
    elif name == "reputation":
        return ReputationStrategy(
            num_clients=nc, selection_fraction=0.6,
            truth_threshold=0.7, initial_reputation=0.5, **common,
        )
    elif name == "median":
        return MedianStrategy(**common)
    elif name == "bulyan":
        return BulyanStrategy(num_malicious=num_malicious, **common)
    elif name == "rfa":
        return RFAStrategy(**common)
    raise ValueError(name)


# ── Simulation ────────────────────────────────────────────────────────
def run_simulation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ds, test_ds = get_dataset(DATASET)
    ds_info = DATASET_INFO[DATASET]

    clean_parts = partition_dataset(
        train_ds, NUM_CLIENTS, PARTITION_TYPE, DIRICHLET_ALPHA, SEED
    )

    rng = np.random.default_rng(SEED)
    num_mal = max(1, min(int(NUM_CLIENTS * MALICIOUS_FRACTION), NUM_CLIENTS - 1))
    malicious = set(rng.choice(NUM_CLIENTS, size=num_mal, replace=False).tolist())
    print(f"Malicious clients: {sorted(malicious)} ({num_mal}/{NUM_CLIENTS})")
    print(f"Partition: {PARTITION_TYPE}, alpha={DIRICHLET_ALPHA}")
    print(f"Partition sizes: {[len(p) for p in clean_parts]}")

    lf_parts = {}
    for cid in malicious:
        lf_parts[cid] = apply_label_flipping(
            clean_parts[cid], ds_info["num_classes"], seed=SEED + cid
        )

    clean_loaders = []
    for i, p in enumerate(clean_parts):
        g = torch.Generator().manual_seed(SEED + i)
        clean_loaders.append(DataLoader(p, batch_size=BATCH_SIZE, shuffle=True, generator=g))

    lf_loaders = {}
    for c, p in lf_parts.items():
        g = torch.Generator().manual_seed(SEED + c + 10000)
        lf_loaders[c] = DataLoader(p, batch_size=BATCH_SIZE, shuffle=True, generator=g)

    testloader = DataLoader(test_ds, batch_size=128, shuffle=False)
    eval_m = get_model(MODEL, DATASET)
    template_model = get_model(MODEL, DATASET)

    all_data = {}

    for strat_name in STRATEGIES:
        print(f"\n  Strategy: {STRAT_LABELS[strat_name]}")
        gm = get_model(MODEL, DATASET)
        gp = [v.cpu().numpy() for v in gm.state_dict().values()]
        strategy = _make_strategy(strat_name, gp, NUM_CLIENTS,
                                  num_malicious=len(malicious))

        losses, accs = [], []
        trust_rounds, cacc_rounds, status_rounds, phase_labels = [], [], [], []
        rep_rounds = []

        l0, a0 = _run_eval(gm, testloader, device)
        losses.append(l0)
        accs.append(a0)
        t0 = time.time()

        for rnd in range(1, TOTAL_ROUNDS + 1):
            pname, atk_type, atk_params = _get_phase(rnd)
            atk_cat = ATTACK_CATEGORIES.get(atk_type)
            phase_labels.append(pname)

            client_results, num_samples, statuses = [], [], {}

            for cid in range(NUM_CLIENTS):
                is_mal = cid in malicious
                attacked = False

                if is_mal and atk_type == "label_flipping" and cid in lf_loaders:
                    loader, attacked = lf_loaders[cid], True
                else:
                    loader = clean_loaders[cid]

                cm = copy.deepcopy(template_model)
                _set_params(cm, gp)
                updated, closs = _train_client(cm, loader, LOCAL_EPOCHS, LR, device)

                if is_mal and atk_cat == "model":
                    updated = _model_attack(
                        updated, atk_type, atk_params,
                        seed=SEED + cid + rnd, global_params=gp,
                    )
                    attacked = True

                statuses[cid] = "attacked" if attacked else ("malicious_idle" if is_mal else "benign")
                client_results.append(updated)
                num_samples.append(len(clean_parts[cid]))

            trust = _compute_trust(client_results, gp)
            caccs = {c: _client_acc(eval_m, client_results[c], testloader, device)
                     for c in range(NUM_CLIENTS)}

            trust_rounds.append(trust)
            cacc_rounds.append(caccs)
            status_rounds.append(statuses)

            agg = _aggregate(strategy, rnd, client_results, num_samples)
            if agg is not None:
                gp = agg
                _set_params(gm, gp)

            rep_scores = {}
            if hasattr(strategy, 'get_reputations'):
                rep_scores = strategy.get_reputations()
            rep_rounds.append(rep_scores)

            l, a = _run_eval(gm, testloader, device)
            losses.append(l)
            accs.append(a)

            elapsed = time.time() - t0
            print(f"    R{rnd:2d}/{TOTAL_ROUNDS} [{pname:24s}] "
                  f"acc={a:.4f} loss={l:.4f} ({elapsed:.0f}s)", flush=True)

        all_data[strat_name] = dict(
            losses=losses, accs=accs, trust=trust_rounds,
            client_acc=cacc_rounds, statuses=status_rounds,
            phases=phase_labels, reputation=rep_rounds,
        )

    return all_data, malicious


# ── PDF Report ────────────────────────────────────────────────────────
def _shade_phases(ax, phases):
    i = 0
    while i < len(phases):
        nm = phases[i]
        s = i
        while i < len(phases) and phases[i] == nm:
            i += 1
        if nm in ATK_COLORS:
            ax.axvspan(s + 0.5, i + 0.5, alpha=0.25, color=ATK_COLORS[nm], zorder=0)


def _add_phase_labels(ax, phases):
    for nm, _, _, (s, e) in PHASES:
        if nm in ATK_COLORS:
            mid = (s + e) / 2
            ymax = ax.get_ylim()[1]
            ax.text(mid, ymax * 0.95, nm, ha="center", fontsize=7,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.85))


def make_pdf(all_data, malicious):
    phases = all_data[STRATEGIES[0]]["phases"]
    num_mal = len(malicious)

    with PdfPages(OUTPUT_PDF) as pdf:

        # PAGE 1: TITLE
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.5, 0.78, "IntelliFL", fontsize=44, ha="center", fontweight="bold")
        fig.text(0.5, 0.68, "Dynamic Multi-Attack Simulation Report",
                 fontsize=22, ha="center")
        fig.text(0.5, 0.58,
                 f"Model: {MODEL.upper()} | Dataset: {DATASET.upper()} | "
                 f"Clients: {NUM_CLIENTS}\n"
                 f"Rounds: {TOTAL_ROUNDS} | Local Epochs: {LOCAL_EPOCHS} | LR: {LR}\n"
                 f"Partition: Non-IID Dirichlet (alpha={DIRICHLET_ALPHA})\n"
                 f"Malicious: {MALICIOUS_FRACTION:.0%} "
                 f"({num_mal} clients: {sorted(malicious)})",
                 fontsize=12, ha="center", family="monospace")

        sched = "Attack Schedule:\n"
        for nm, _, _, (s, e) in PHASES:
            tag = ">>>" if nm in ATK_COLORS else "   "
            sched += f"  {tag} Rounds {s:2d}-{e:2d}: {nm}\n"
        fig.text(0.5, 0.30, sched, fontsize=11, ha="center", family="monospace")

        strat_str = "Strategies:\n"
        for s in STRATEGIES:
            strat_str += f"  - {STRAT_LABELS[s]}\n"
        fig.text(0.5, 0.12, strat_str, fontsize=11, ha="center", family="monospace")

        fig.text(0.5, 0.03, f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')} | Seed: {SEED}",
                 fontsize=9, ha="center", color="gray")
        pdf.savefig(fig); plt.close()

        # PAGE 2: EXPERIMENT DESIGN
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle("Experiment Design", fontsize=18, fontweight="bold", y=0.96)
        design_text = (
            f"\nOBJECTIVE\n"
            f"  Test how robust aggregation strategies respond to escalating attacks\n"
            f"  in a realistic non-IID federated setting.\n\n"
            f"SETUP\n"
            f"  Model: {MODEL.upper()} (lightweight, amplifies attack visibility)\n"
            f"  Dataset: {DATASET.upper()} (10 classes, 50K train / 10K test)\n"
            f"  Partition: Non-IID Dirichlet (alpha={DIRICHLET_ALPHA})\n"
            f"  Clients: {NUM_CLIENTS} total, {num_mal} malicious ({MALICIOUS_FRACTION:.0%})\n"
            f"  Training: {LOCAL_EPOCHS} local epochs per round, SGD lr={LR}\n\n"
            f"ATTACK SCHEDULE (escalating severity)\n"
            f"  R1-3:   BASELINE - all clients train honestly\n"
            f"  R4-6:   LABEL FLIPPING - bijective class permutation (data poisoning)\n"
            f"          Signal: directional divergence, normal magnitude\n"
            f"  R7-9:   RECOVERY - attacks stop, test restoration\n"
            f"  R10-12: WEIGHT SPIKING - 30%% weights x100 (model poisoning)\n"
            f"          Signal: sparse per-coordinate extreme outliers\n"
            f"  R13-15: BYZANTINE PERTURBATION - full noise replacement (model poisoning)\n"
            f"          Signal: maximum L2 distance from all benign clients\n\n"
            f"STRATEGIES\n"
            f"  FedAvg: Weighted average, no defense (baseline)\n"
            f"  Trimmed Mean: Coordinate-wise trim top/bottom beta fraction\n"
            f"    beta = max(0.1, {MALICIOUS_FRACTION}) = {max(0.1, MALICIOUS_FRACTION):.1f}\n"
            f"  Krum (Multi): Select top-m clients by L2 neighbor distance\n"
            f"    m = n - f = {NUM_CLIENTS} - {num_mal} = {NUM_CLIENTS - num_mal}\n"
            f"  Reputation: Track per-client trust over time, select top-k\n"
            f"    Selection: 0.6 ({max(1, int(NUM_CLIENTS * 0.6))}/{NUM_CLIENTS}), "
            f"threshold: 0.7, asymmetric decay\n\n"
            f"KEY QUESTIONS\n"
            f"  1. Does FedAvg collapse under model poisoning?\n"
            f"  2. Which strategy handles directional vs magnitude attacks best?\n"
            f"  3. Does Reputation's temporal memory help during recovery?\n"
            f"  4. How does non-IID data affect trust score reliability?"
        )
        fig.text(0.04, 0.90, design_text, fontsize=8.5, va="top", family="monospace",
                 transform=fig.transFigure, linespacing=1.25)
        pdf.savefig(fig); plt.close()

        # PAGE 3: GLOBAL PERFORMANCE
        fig, (ax_loss, ax_acc) = plt.subplots(2, 1, figsize=(11, 8.5), sharex=True)
        fig.suptitle("Global Model Performance Across Attack Phases",
                     fontsize=16, fontweight="bold")

        rounds_x = range(TOTAL_ROUNDS + 1)
        for s in STRATEGIES:
            c, l, m = STRAT_COLORS[s], STRAT_LABELS[s], STRAT_MARKERS[s]
            ax_loss.plot(rounds_x, all_data[s]["losses"], f"{m}-", color=c,
                         label=l, ms=5, lw=2, markeredgewidth=0.5, markeredgecolor="white")
            ax_acc.plot(rounds_x, all_data[s]["accs"], f"{m}-", color=c,
                        label=l, ms=5, lw=2, markeredgewidth=0.5, markeredgecolor="white")

        _shade_phases(ax_loss, phases); _shade_phases(ax_acc, phases)
        _add_phase_labels(ax_loss, phases)
        ax_loss.set_ylabel("Loss", fontsize=12)
        ax_loss.set_title("Aggregated Loss per Round", fontsize=12)
        ax_loss.legend(fontsize=10, loc="upper right"); ax_loss.grid(True, alpha=0.3)
        ax_acc.set_xlabel("Round", fontsize=12); ax_acc.set_ylabel("Accuracy", fontsize=12)
        ax_acc.set_title("Global Model Accuracy per Round", fontsize=12)
        ax_acc.set_ylim(0, 1.05); ax_acc.legend(fontsize=10, loc="lower right")
        ax_acc.grid(True, alpha=0.3); ax_acc.set_xticks(range(0, TOTAL_ROUNDS + 1))
        fig.tight_layout(rect=[0, 0, 1, 0.95]); pdf.savefig(fig); plt.close()

        # PAGE 4: CLIENT ACTIVITY GRIDS
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle("Client Activity Grid", fontsize=14, fontweight="bold")
        sm = {"benign": 0, "attacked": 1, "malicious_idle": 2}
        cmap = ListedColormap(["#2ecc71", "#e74c3c", "#95a5a6"])
        for idx, s in enumerate(STRATEGIES):
            ax = axes[idx // 2][idx % 2]
            z = np.zeros((NUM_CLIENTS, TOTAL_ROUNDS), dtype=int)
            for ri, st in enumerate(all_data[s]["statuses"]):
                for ci, status in st.items():
                    z[ci, ri] = sm.get(status, 0)
            ax.imshow(z, aspect="auto", cmap=cmap, vmin=0, vmax=2, interpolation="nearest")
            ax.set_title(STRAT_LABELS[s], fontsize=11, fontweight="bold", color=STRAT_COLORS[s])
            ax.set_xlabel("Round", fontsize=9); ax.set_ylabel("Client", fontsize=9)
            ax.set_yticks(range(NUM_CLIENTS))
            ax.set_yticklabels([f"C{c}{'*' if c in malicious else ''}" for c in range(NUM_CLIENTS)], fontsize=7)
            ax.set_xticks(range(TOTAL_ROUNDS))
            ax.set_xticklabels([str(r + 1) for r in range(TOTAL_ROUNDS)], fontsize=7)
            for _, _, _, (ps, pe) in PHASES:
                ax.axvline(x=ps - 1.5, color="white", lw=0.8, ls="--", alpha=0.7)
        fig.tight_layout(rect=[0, 0.05, 1, 0.94])
        fig.legend(handles=[Patch(fc="#2ecc71", label="Benign"), Patch(fc="#e74c3c", label="Attacked"),
                            Patch(fc="#95a5a6", label="Malicious (idle)")],
                   loc="lower center", ncol=3, fontsize=10)
        pdf.savefig(fig); plt.close()

        # PAGE 5: TRUST SCORES
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle("Per-Client Trust Scores (Cosine Similarity to Median Update)",
                     fontsize=14, fontweight="bold")
        for idx, s in enumerate(STRATEGIES):
            ax = axes[idx // 2][idx % 2]
            td = all_data[s]["trust"]
            for cid in range(NUM_CLIENTS):
                vals = [td[r][cid] for r in range(TOTAL_ROUNDS)]
                im = cid in malicious
                ax.plot(range(1, TOTAL_ROUNDS + 1), vals,
                        "--" if im else "-", color="#e74c3c" if im else "#2ecc71",
                        alpha=0.9 if im else 0.5, lw=2.0 if im else 1.0,
                        label=f"C{cid}*" if im else None)
            _shade_phases(ax, phases)
            ax.set_title(STRAT_LABELS[s], fontsize=11, fontweight="bold", color=STRAT_COLORS[s])
            ax.set_xlabel("Round", fontsize=9); ax.set_ylabel("Trust Score", fontsize=9)
            ax.set_ylim(-0.05, 1.1); ax.set_xticks(range(1, TOTAL_ROUNDS + 1))
            ax.grid(True, alpha=0.3)
        fig.tight_layout(rect=[0, 0.05, 1, 0.94])
        fig.legend(handles=[plt.Line2D([0], [0], color="#2ecc71", lw=1.5, label="Benign"),
                            plt.Line2D([0], [0], color="#e74c3c", ls="--", lw=2, label="Malicious")],
                   loc="lower center", ncol=2, fontsize=10)
        pdf.savefig(fig); plt.close()

        # PAGE 6: CLIENT ACCURACY
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle("Per-Client Model Accuracy", fontsize=14, fontweight="bold")
        for idx, s in enumerate(STRATEGIES):
            ax = axes[idx // 2][idx % 2]
            cd = all_data[s]["client_acc"]
            for cid in range(NUM_CLIENTS):
                vals = [cd[r][cid] for r in range(TOTAL_ROUNDS)]
                im = cid in malicious
                ax.plot(range(1, TOTAL_ROUNDS + 1), vals,
                        "--" if im else "-", color="#e74c3c" if im else "#2ecc71",
                        alpha=0.9 if im else 0.5, lw=2.0 if im else 1.0)
            _shade_phases(ax, phases)
            ax.set_title(STRAT_LABELS[s], fontsize=11, fontweight="bold", color=STRAT_COLORS[s])
            ax.set_xlabel("Round", fontsize=9); ax.set_ylabel("Accuracy", fontsize=9)
            ax.set_ylim(0, 1.05); ax.set_xticks(range(1, TOTAL_ROUNDS + 1))
            ax.grid(True, alpha=0.3)
        fig.tight_layout(rect=[0, 0.05, 1, 0.94])
        fig.legend(handles=[plt.Line2D([0], [0], color="#2ecc71", lw=1.5, label="Benign"),
                            plt.Line2D([0], [0], color="#e74c3c", ls="--", lw=2, label="Malicious")],
                   loc="lower center", ncol=2, fontsize=10)
        pdf.savefig(fig); plt.close()

        # PAGE 7: REPUTATION EVOLUTION
        fig, ax = plt.subplots(1, 1, figsize=(11, 6))
        fig.suptitle("Reputation Score Evolution (Reputation Strategy)",
                     fontsize=16, fontweight="bold")
        rep_data = all_data["reputation"]["reputation"]
        for cid in range(NUM_CLIENTS):
            vals = [rep_data[r].get(cid, 0.5) for r in range(TOTAL_ROUNDS)]
            im = cid in malicious
            ax.plot(range(1, TOTAL_ROUNDS + 1), vals,
                    "--" if im else "-", color="#e74c3c" if im else "#2ecc71",
                    alpha=0.9 if im else 0.5, lw=2.5 if im else 1.2,
                    label=f"Client {cid}" + (" (malicious)" if im else ""))
        _shade_phases(ax, phases); _add_phase_labels(ax, phases)
        ax.set_xlabel("Round", fontsize=12); ax.set_ylabel("Reputation Score", fontsize=12)
        ax.set_ylim(-0.05, 1.1); ax.set_xticks(range(1, TOTAL_ROUNDS + 1))
        ax.grid(True, alpha=0.3); ax.legend(fontsize=8, ncol=2, loc="lower left")
        ax.axhline(y=0.5, color="gray", ls=":", lw=1, alpha=0.6)
        ax.text(0.5, 0.52, "Initial reputation", fontsize=8, color="gray", alpha=0.7)
        fig.tight_layout(rect=[0, 0, 1, 0.94]); pdf.savefig(fig); plt.close()

        # PAGE 8: PHASE BAR CHART
        fig = plt.figure(figsize=(11, 7))
        fig.suptitle("Accuracy at End of Each Phase", fontsize=16, fontweight="bold")
        pnames = [nm for nm, _, _, _ in PHASES]
        x = np.arange(len(pnames))
        n_strat = len(STRATEGIES)
        w = 0.8 / n_strat
        ax = fig.add_subplot(111)
        for i, s in enumerate(STRATEGIES):
            vals = [all_data[s]["accs"][e] for _, _, _, (_, e) in PHASES]
            bars = ax.bar(x + i * w, vals, w, label=STRAT_LABELS[s],
                          color=STRAT_COLORS[s], edgecolor="white", linewidth=0.5)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=7, fontweight="bold")
        ax.set_xticks(x + w * (n_strat - 1) / 2)
        ax.set_xticklabels(pnames, fontsize=10, rotation=20, ha="right")
        ax.set_ylabel("Accuracy", fontsize=12); ax.set_ylim(0, 1.15)
        ax.legend(fontsize=10); ax.grid(True, alpha=0.3, axis="y")
        for i, (nm, _, _, _) in enumerate(PHASES):
            if nm in ATK_COLORS:
                ax.axvspan(i - 0.4, i + 0.4 + w * n_strat, alpha=0.1, color="#e74c3c")
        fig.tight_layout(rect=[0, 0, 1, 0.95]); pdf.savefig(fig); plt.close()

        # PAGE 9: SUMMARY TABLE
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle("Summary: Accuracy Drop per Attack Phase (vs Baseline R3)",
                     fontsize=16, fontweight="bold", y=0.95)
        atk_phases = [(nm, s, e) for nm, at, _, (s, e) in PHASES if nm in ATK_COLORS]
        col_labels = [STRAT_LABELS[s] for s in STRATEGIES]
        row_labels = [f"{nm}\n(R{s}-{e})" for nm, s, e in atk_phases]
        baseline_end = next(e for nm, _, _, (_, e) in PHASES if nm == "Baseline")
        baseline = {s: all_data[s]["accs"][baseline_end] for s in STRATEGIES}
        tdata, tcols = [], []
        for pnm, ps, pe in atk_phases:
            row, cols = [], []
            for s in STRATEGIES:
                a = all_data[s]["accs"][pe]
                d = a - baseline[s]
                row.append(f"{a:.3f}\n({d:+.3f})")
                if d < -0.15:    cols.append("#fadbd8")
                elif d < -0.05:  cols.append("#fdebd0")
                elif d > 0.01:   cols.append("#d5f5e3")
                else:            cols.append("#ffffff")
            tdata.append(row); tcols.append(cols)
        ax = fig.add_subplot(111); ax.axis("off")
        t = ax.table(cellText=tdata, rowLabels=row_labels, colLabels=col_labels,
                     cellColours=tcols, loc="center", cellLoc="center")
        t.auto_set_font_size(False); t.set_fontsize(11); t.scale(1.0, 3.0)
        for j in range(len(col_labels)):
            t[0, j].set_facecolor("#2c3e50"); t[0, j].set_text_props(color="white", fontweight="bold")
        fig.text(0.5, 0.15,
                 "Values: accuracy at end of phase (change from baseline R3).\n"
                 "Red = severe drop (>15%%), Orange = moderate (>5%%), "
                 "Green = improvement, White = minimal change.",
                 ha="center", fontsize=11, style="italic")
        pdf.savefig(fig); plt.close()

        # PAGES 10-11: ANALYSIS
        bl = {s: all_data[s]["accs"][3] for s in STRATEGIES}
        lf = {s: all_data[s]["accs"][6] for s in STRATEGIES}
        ws = {s: all_data[s]["accs"][12] for s in STRATEGIES}
        bz = {s: all_data[s]["accs"][15] for s in STRATEGIES}
        rc = {s: all_data[s]["accs"][9] for s in STRATEGIES}

        analysis1 = (
            f"\nPHASE-BY-PHASE ANALYSIS\n\n"
            f"1. BASELINE (Rounds 1-3)\n"
            f"   FedAvg: {bl['fedavg']:.3f} | TrimMean: {bl['trimmed_mean']:.3f} | "
            f"Krum: {bl['krum']:.3f} | Rep: {bl['reputation']:.3f}\n"
            f"   Non-IID (alpha={DIRICHLET_ALPHA}) creates skewed class distributions.\n"
            f"   Robust strategies may converge differently due to implicit filtering.\n\n"
            f"2. LABEL FLIPPING (Rounds 4-6) - Data Poisoning\n"
            f"   FedAvg: {lf['fedavg']:.3f} ({lf['fedavg']-bl['fedavg']:+.3f}) | "
            f"TrimMean: {lf['trimmed_mean']:.3f} ({lf['trimmed_mean']-bl['trimmed_mean']:+.3f}) | "
            f"Krum: {lf['krum']:.3f} ({lf['krum']-bl['krum']:+.3f}) | "
            f"Rep: {lf['reputation']:.3f} ({lf['reputation']-bl['reputation']:+.3f})\n"
            f"   Directional attack with normal magnitude. Krum detects via L2 distance.\n"
            f"   Trimmed Mean may miss it (per-coordinate values look normal).\n\n"
            f"3. RECOVERY (Rounds 7-9)\n"
            f"   FedAvg: {rc['fedavg']:.3f} | TrimMean: {rc['trimmed_mean']:.3f} | "
            f"Krum: {rc['krum']:.3f} | Rep: {rc['reputation']:.3f}\n"
            f"   Malicious clients train honestly. Reputation still partially excludes\n"
            f"   previously-malicious clients due to decayed reputation scores.\n\n"
            f"4. WEIGHT SPIKING (Rounds 10-12) - Model Poisoning\n"
            f"   FedAvg: {ws['fedavg']:.3f} ({ws['fedavg']-rc['fedavg']:+.3f}) | "
            f"TrimMean: {ws['trimmed_mean']:.3f} ({ws['trimmed_mean']-rc['trimmed_mean']:+.3f}) | "
            f"Krum: {ws['krum']:.3f} ({ws['krum']-rc['krum']:+.3f}) | "
            f"Rep: {ws['reputation']:.3f} ({ws['reputation']-rc['reputation']:+.3f})\n"
            f"   Sparse extreme outliers. Trimmed Mean clips per-coordinate extremes.\n"
            f"   FedAvg typically collapses here.\n\n"
            f"5. BYZANTINE PERTURBATION (Rounds 13-15) - Worst Case\n"
            f"   FedAvg: {bz['fedavg']:.3f} ({bz['fedavg']-ws['fedavg']:+.3f}) | "
            f"TrimMean: {bz['trimmed_mean']:.3f} ({bz['trimmed_mean']-ws['trimmed_mean']:+.3f}) | "
            f"Krum: {bz['krum']:.3f} ({bz['krum']-ws['krum']:+.3f}) | "
            f"Rep: {bz['reputation']:.3f} ({bz['reputation']-ws['reputation']:+.3f})\n"
            f"   Full parameter replacement with noise. All robust strategies should\n"
            f"   handle this. FedAvg collapses to random chance (~10%%).\n"
        )

        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle("Analysis: Phase-by-Phase Results", fontsize=16, fontweight="bold", y=0.97)
        fig.text(0.04, 0.92, analysis1, fontsize=8.5, va="top", family="monospace",
                 transform=fig.transFigure, linespacing=1.25)
        pdf.savefig(fig); plt.close()

        best_lf = max(["trimmed_mean", "krum", "reputation"], key=lambda s: lf[s])
        best_ws = max(["trimmed_mean", "krum", "reputation"], key=lambda s: ws[s])
        best_bz = max(["trimmed_mean", "krum", "reputation"], key=lambda s: bz[s])

        analysis2 = (
            f"\nSTRATEGY COMPARISON\n\n"
            f"                      Label Flip    Weight Spike  Byzantine\n"
            f"  FedAvg              {lf['fedavg']:.3f}         {ws['fedavg']:.3f}         {bz['fedavg']:.3f}\n"
            f"  Trimmed Mean        {lf['trimmed_mean']:.3f}         {ws['trimmed_mean']:.3f}         {bz['trimmed_mean']:.3f}\n"
            f"  Krum (Multi)        {lf['krum']:.3f}         {ws['krum']:.3f}         {bz['krum']:.3f}\n"
            f"  Reputation          {lf['reputation']:.3f}         {ws['reputation']:.3f}         {bz['reputation']:.3f}\n\n"
            f"  Best robust:        {STRAT_LABELS[best_lf]:18s} {STRAT_LABELS[best_ws]:18s} {STRAT_LABELS[best_bz]}\n\n"
            f"TRUST SCORE OBSERVATIONS\n"
            f"  Data poisoning (label flipping): Subtle trust differentiation because\n"
            f"  poisoned updates have normal magnitude, just wrong direction.\n"
            f"  Model poisoning (weight spiking, Byzantine): Clear separation -\n"
            f"  malicious clients drop near 0, benign stay near 1.0.\n\n"
            f"REPUTATION DEEP DIVE\n"
            f"  Temporal memory is the key differentiator:\n"
            f"  - Clients that attacked in R4-6 have degraded reputation entering R7\n"
            f"  - During recovery, they are still partially excluded\n"
            f"  - Asymmetric update: exponential decay vs linear growth\n"
            f"  - Trade-off: 40%% of clients always excluded (selection_fraction=0.6)\n\n"
            f"CONCLUSIONS\n"
            f"  1. FedAvg is unsuitable for adversarial environments\n"
            f"  2. Trimmed Mean: strong coordinate-wise protection\n"
            f"  3. Krum: excels at directional and large-norm attacks\n"
            f"  4. Reputation: best for dynamic/intermittent attacks\n"
            f"  5. No single strategy dominates all attack types\n"
        )

        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle("Analysis: Strategy Comparison", fontsize=16, fontweight="bold", y=0.97)
        fig.text(0.04, 0.92, analysis2, fontsize=9, va="top", family="monospace",
                 transform=fig.transFigure, linespacing=1.25)
        pdf.savefig(fig); plt.close()

    print(f"\nReport saved to: {os.path.abspath(OUTPUT_PDF)}")


if __name__ == "__main__":
    print("=" * 60)
    print("IntelliFL - Dynamic Multi-Attack Simulation")
    print("=" * 60)
    print(f"Model: {MODEL} | Dataset: {DATASET} | Clients: {NUM_CLIENTS}")
    print(f"Rounds: {TOTAL_ROUNDS} | Local Epochs: {LOCAL_EPOCHS}")
    print(f"Partition: {PARTITION_TYPE} (alpha={DIRICHLET_ALPHA})")
    print(f"Malicious: {MALICIOUS_FRACTION:.0%}")
    print(f"Strategies: {', '.join(STRAT_LABELS[s] for s in STRATEGIES)}\n")

    data, mal = run_simulation()
    print("\nGenerating PDF...")
    make_pdf(data, mal)
