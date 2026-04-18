# FEDSIM ↔ IntelliFL Algorithmic Divergence

Record of behavior differences between FEDSIM (`fedsim/`) and the upstream IntelliFL reference implementation (`fl-execution-framework-dev`, Korobeinikov & Reznik, `github.com/dmitrykoro/fl-execution-framework-dev`).

Captured **2026-04-18** against:
- FEDSIM branch `feat/anomaly-detection-layer`, commit `9c9041c`.
- IntelliFL `main`, commit `82c4e8ad8`.

**Purpose.** Where both projects claim to implement the same algorithm or attack, the implementations diverge in ways that affect numerical results, client-selection trajectories, and/or reproducibility. This doc enumerates those divergences so that (a) cross-framework benchmarks are interpreted correctly, (b) any future effort to align FEDSIM to IntelliFL's behavior has a concrete bill-of-materials to work from, and (c) any FEDSIM audit fix that would be reversed by such alignment is explicitly flagged.

No code changes are implied by this document.

---

## Top-Level Verdict

Given identical datasets and identical user-visible configs, only two overlapping components produce numerically equivalent results end-to-end:

- `trimmed_mean` aggregation math (both compute coordinate-wise trimmed mean with the same trim count).
- `gradient_scaling` attack (both apply `global + scale·(local − global)`).

Every other overlapping component diverges in formula, edge-case handling, stateful-vs-stateless removal semantics, or RNG determinism.

| Component | Aggregation formula | Per-round client set | End-to-end trajectory |
|---|:-:|:-:|:-:|
| Krum (single) | differs | differs | differs |
| Multi-Krum | near-match | differs | differs |
| Bulyan | differs | differs | differs |
| RFA | same fixed point, different path | differs | differs |
| Trimmed Mean | **same** | differs (IntelliFL removes permanently) | differs |
| Trust / Reputation | differs | differs | differs |
| label_flipping attack | n/a | n/a | **different threat model** |
| gaussian_noise attack | n/a | n/a | qualitatively similar |
| gradient_scaling attack | n/a | n/a | **same in FP32** |

---

## Overlapping Strategies

### 1. Krum (single)

FEDSIM: `fedsim/strategies/krum.py` (`KrumStrategy`, `multi_krum=False`)
IntelliFL: `src/simulation_strategies/krum_based_removal_strategy.py` (`KrumBasedRemovalStrategy`)

| Aspect | FEDSIM | IntelliFL |
|---|---|---|
| Score formula | `sum of n − f − 2 nearest` (Blanchard canonical) | `sum(sorted_distances[:num_malicious − 2])` |
| Self-distance | excluded via `np.fill_diagonal(distances, np.inf)` | included in the sort (always the smallest term, which is 0) |
| Invariant check | warns if `n < 2f + 3` | none |
| Distance metric | squared L2 via broadcasting | L2 via nested-loop `np.linalg.norm` |
| NaN/Inf input filter | `filter_nan_clients` drops offenders | none |
| Removal model | stateless per round | stateful permanent (highest-score client removed each round, never re-admitted) |
| Aggregation | unweighted mean of selected | Flower FedAvg weighted by `num_examples` |
| dtype | preserved (`.astype(weights[0][idx].dtype)`) | implicit upcast to float64 |

**Net effect.** For typical small f (1–3), IntelliFL's truncated sum produces near-degenerate scores — `num_malicious − 2` can be 0, 1, or 2 terms, often just the self-distance (0). Client ranking is effectively arbitrary in that regime. FEDSIM's canonical formula does not have this issue.

### 2. Multi-Krum

FEDSIM: `fedsim/strategies/krum.py` (`KrumStrategy`, `multi_krum=True`)
IntelliFL: `src/simulation_strategies/multi_krum_strategy.py` (`MultiKrumStrategy`)

| Aspect | FEDSIM | IntelliFL |
|---|---|---|
| Score formula | `sum of n − f − 2 nearest` | `sum(sorted_distances[:num_krum_selections − 2])` |
| Self-distance | excluded | included |
| Number selected | `max(1, n − 2f)` (from paper) | `num_krum_selections` (user hyperparameter) |
| Chunked distance calc | no | yes, activated at >50M parameters (for transformer-scale models) |
| Removal model | stateless | stateless per-round reset (set is recomputed each round) |
| Aggregation | unweighted mean of top-k | Flower FedAvg weighted by `num_examples` on top-k |

**Net effect.** For clear-cut attack scenarios the top-k selected sets usually agree. Under ambiguous attack or Non-IID data, IntelliFL's score off-by-constant can select different clients. `num_examples` weighting diverges the aggregated weights every round under Non-IID Dirichlet partitioning.

### 3. Bulyan

FEDSIM: `fedsim/strategies/bulyan.py` (`BulyanStrategy`)
IntelliFL: `src/simulation_strategies/bulyan_strategy.py` (`BulyanStrategy`)

| Aspect | FEDSIM | IntelliFL |
|---|---|---|
| Candidate selection | **iterative Krum** — pick best, remove from pool, repeat θ times (canonical El Mhamdi et al. 2018) | **single-batch Multi-Krum** — one `argpartition(scores, C)` pass picks all C at once |
| # candidates | `θ = max(1, n − 2f)` | `C = num_bulyan_benign_clients = n − f` |
| Self-distance in Krum step | excluded | included |
| Precondition check | **raises `ValueError`** if `n < 4f + 3` | warns and falls back to plain FedAvg mean |
| Trimming per dim | drops `f` from each end (with safety clamp) | drops `f` from each end |
| Client scores metric | returns normalized `1 − dist/max_dist` per client | returns absolute deviation |
| Removal model | stateless | stateless per-round reset |

**Net effect.** The iterative vs batch candidate selection can pick different subsets when multiple malicious clients have similar properties — the canonical algorithm is iterative because the removal of the first malicious pick should change subsequent scores. Different `C` vs `θ` values give fundamentally different candidate-set sizes.

### 4. RFA (Robust Federated Aggregation, geometric median)

FEDSIM: `fedsim/strategies/rfa.py` (`RFAStrategy`)
IntelliFL: `src/simulation_strategies/rfa_based_removal_strategy.py` (`RFABasedRemovalStrategy`)

| Aspect | FEDSIM | IntelliFL |
|---|---|---|
| Weiszfeld initialization | **coordinate-wise median** (`np.median`) — robust seed | `np.mean` — Byzantine-biased seed |
| Denominator floor | `max(distances, 1e-4)` | `max(distances, 1e-5)` |
| Max iterations | 100 | 1000 |
| Convergence tolerance | 1e-6 | 1e-5 |
| Extra multiplier | none | `weighted_median_factor` (default 1.0, unused in current configs — would shift the aggregated model if set ≠ 1.0) |
| NaN/Inf input filter | `filter_nan_clients` | none |
| dtype | preserved | implicit upcast |
| Removal model | none (pure aggregation) | stateful permanent (highest-deviation client permanently excluded each round) |

**Net effect.** Both converge toward the true geometric median for well-separated inputs, but the *initial point* matters under contamination. Under attack, the mean init is already poisoned and needs more iterations to unpoison itself. The different regularization floors (1e-5 vs 1e-4) weight near-coincident points by ~10⁵ vs ~10⁴ respectively, biasing the weighted average.

### 5. Trimmed Mean

FEDSIM: `fedsim/strategies/trimmed_mean.py` (`TrimmedMean`)
IntelliFL: `src/simulation_strategies/trimmed_mean_based_removal_strategy.py` (`TrimmedMeanBasedRemovalStrategy`)

| Aspect | FEDSIM | IntelliFL |
|---|---|---|
| Aggregation formula | coordinate-wise, `trim = int(beta · n)` per side | coordinate-wise, `trim = int(trim_ratio · n)` per side |
| Safety clamp | `trim = min(trim, (n − 1) // 2)` — guarantees non-empty middle slice | none — `trim_ratio ≥ 0.5` crashes on empty mean |
| Warning when no trimming occurs | yes (`beta > 0` but `trim = 0`) | no — silently no-ops |
| Vectorization | `np.sort(axis=0)` fully vectorized | per-scalar Python loop over flattened layer (slow for large models) |
| NaN/Inf input filter | yes | none |
| dtype | preserved | implicit upcast |
| Removal model | none | stateful permanent (client with highest trim frequency removed each round) |

**Net effect.** When the trim count resolves to the same value on both sides, the aggregated weights are **numerically identical** (same formula). IntelliFL's permanent client removal adds a layer on top that changes the client pool over rounds, so end-to-end trajectories still diverge.

### 6. Trust / Reputation (Chuprov 2019 + Patel et al. ASIA'24)

FEDSIM: `fedsim/custom/strategies/reputation.py` (`ReputationStrategy`, plugin `custom:Reputation`)
IntelliFL: `src/simulation_strategies/trust_based_removal_strategy.py` (`TrustBasedRemovalStrategy`)

Both projects cite the same underlying papers. The implementations disagree on the formulas.

| Aspect | FEDSIM | IntelliFL |
|---|---|---|
| Reputation growth branch (d ≥ α) | `R_new = prev_R + d + prev_R/round` | `R_new = prev_R + d − prev_R/round` |
| Reputation decay branch (d < α) | `R_new = prev_R + d − exp(−(1 − d)) · prev_R/round` | `R_new = prev_R + d − exp(−(1 − d · prev_R/round))` |
| Reputation smoothing | none on R | `R = β·R + (1 − β)·prev_R` (β shared with trust) |
| Truth value (d) clustering | **K-means K=2** on parameter **deltas from global**, uses larger cluster centroid | **K-means K=1** on **raw parameter vectors**, uses single centroid with MinMax scaler |
| KMeans seeding | `random_state=0` | unseeded (run-to-run non-determinism in cluster labeling and, with k-means++, init) |
| Trust raw formula | `√(R² + d²) − √((1 − R)² + (1 − d)²)` (range `[-√2, √2]`) | same |
| Trust range mapping | smooth linear `(raw + √2) / (2√2) → [0, 1]` (monotone, preserves ordering) | hard clip raw to `[0, 1]` (negatives collapse to 0, multiple malicious clients become indistinguishable) |
| Smoothing β scope | `smoothing_beta` applies only to trust | `beta_value` applies to both reputation and trust |
| Removal model | stateless per-round exclusion below `trust_exclusion_threshold` | stateful permanent — first round removes lowest-trust client, subsequent rounds remove all below `trust_threshold` |

**Net effect.** The most consequential divergence in the overlap. The sign discrepancy on the growth branch alone means the two frameworks produce meaningfully different reputation trajectories: IntelliFL subtracts `prev_R/round` in the growth case, suppressing reputation even for trusted clients, while FEDSIM adds it (consistent with the Chuprov paper's description of asymmetric-growth). Combined with K=1 vs K=2 truth clustering and hard-clip vs smooth trust mapping, the two implementations will flag **different clients at different rounds with different trust scores**. Published results from one are not numerically reproducible in the other.

---

## Overlapping Attacks

### 1. Label Flipping

FEDSIM: `fedsim/attacks/data_poisoning.py::LabelFlippedDataset`
IntelliFL: `src/attack_utils/poisoning.py::apply_label_flipping`

| Aspect | FEDSIM | IntelliFL |
|---|---|---|
| RNG | seeded `np.random.default_rng(seed=42)` | global `torch.randperm` (unseeded) |
| Permutation lifetime | **fixed at dataset construction** — persists for the malicious client's lifetime | **redrawn on every `fit()` call** — the flip map changes every round/batch |
| Derangement guarantee | shuffle + retry; fallback cyclic shift by 1 | swap `i` with `(i + 1) % num_classes` if any self-map remains |
| Application site | dataset wrapper (`__getitem__`) | inline in `FlowerClient.fit` per batch |

**Threat-model implication.** IntelliFL's version is effectively *random relabeling* — labels change identity every round. FEDSIM's is classical *label flipping* (class A consistently → class B). These are meaningfully different adversarial models. A defense that detects consistent malicious gradient direction (Reputation, PID) performs differently under the two.

### 2. Gaussian Noise

FEDSIM: `fedsim/attacks/data_poisoning.py::GaussianNoiseDataset`
IntelliFL: `src/attack_utils/poisoning.py::apply_gaussian_noise`

| Aspect | FEDSIM | IntelliFL |
|---|---|---|
| Application site | dataset `__getitem__` | inline in `FlowerClient.fit` |
| RNG | per-sample seeded `torch.Generator().manual_seed(base + idx)` | global `torch.randn_like` (thread-shared in Ray) |
| Signal-power calculation | `data.pow(2).mean()` — shape-agnostic | `data[indices]**2.mean(dim=(1, 2, 3))` — assumes 4D image tensor |
| Noise floor | `max(noise_std, 1e-4)` | none |
| Output clamping | none | `torch.clamp(..., 0, 1)` |
| Selection | `attack_fraction` pre-selected indices at dataset creation | `attack_ratio` sampled per call |

**Net effect.** The 4D shape assumption in IntelliFL silently breaks on non-image data (tabular, 1D text embeddings). The `[0, 1]` clamp in IntelliFL truncates noise for normalized data; FEDSIM allows values outside that range to propagate. Results are qualitatively similar for standard image benchmarks under mid-range SNR.

### 3. Gradient Scaling

FEDSIM: `fedsim/attacks/model_poisoning.py::apply_gradient_scaling`
IntelliFL: `src/attack_utils/poisoning.py::apply_gradient_scaling`

| Aspect | FEDSIM | IntelliFL |
|---|---|---|
| Formula | `global + scale · (local − global)` | `global + scale · (local − global)` |
| Null-global handling | raises `ValueError` (explicit contract) | no guard — silent no-op or NaN |
| Length-mismatch check | raises `ValueError` | none |
| dtype preservation | `.astype(local.dtype)` | implicit upcast |
| NaN/Inf guard | `np.nan_to_num` clamp on result (matters for FP16 with `scale_factor ≥ 50`) | none |

**Net effect.** Mathematically identical in FP32 with moderate `scale_factor`. FEDSIM adds safety rails that prevent bad configs from producing corrupted state, but those rails don't change the aggregation path under valid configs.

---

## Cross-Cutting Reproducibility Divergences

Even when algorithms are component-wise aligned, these pipeline-level differences can produce different results:

| Source | FEDSIM | IntelliFL |
|---|---|---|
| `KMeans(random_state=…)` | seeded (`0`) in Reputation clustering | **unseeded** in Krum, MultiKrum, RFA, Bulyan, Trust (k-means++ init is non-deterministic across runs) |
| Per-strategy model init | explicit `torch.manual_seed` before each model build | Torch default (depends on process-global state; non-deterministic across runs on GPU) |
| Client sampling per round | seeded RNG in custom runner | Flower internal sampler (Ray-threaded, order-non-deterministic) |
| Parallelism | `ThreadPoolExecutor` + CUDA streams (shared memory) | Ray (isolated workers with `num_cpus`/`num_gpus` quotas) |
| Aggregation weighting | generally unweighted mean of selected | Flower `FedAvg` weighted by client `num_examples` |
| Checkpoint / resume | supported via `api/experiment.py` | not supported — any crash re-runs from scratch |

The `num_examples` weighting alone is load-bearing: under Non-IID Dirichlet partitioning, client sample counts differ and the two frameworks produce different aggregated models every round even when all upstream steps agree.

---

## FEDSIM Audit Fixes That Would Be Reverted by Literal Alignment

Enumerated against the audit memory in `~/.claude/projects/-home-abhikroy-Desktop-FEDSIM/memory/` (snapshots 2026-03-25, 2026-04-09, 2026-04-17). Listed here for traceability if a future effort seeks literal behavior-parity with IntelliFL.

**Krum / Multi-Krum.** Canonical score formula (audit 2026-04-09 round 1 CRITICAL "Krum m=n-2f"), self-distance exclusion, `n >= 2f + 3` invariant check, NaN/Inf client filter (audit 2026-03-25 LOW-23), dtype preservation (audit 2026-04-09 round 2).

**Bulyan.** Hard `n >= 4f + 3` raise (audit 2026-04-09 round 1 CRITICAL "Bulyan hard error"), iterative Krum candidate selection (audit 2026-04-09 round 1 "Bulyan invariant"), client_scores export (audit 2026-04-09 round 1), fraction_fit-aware pre-flight (audit 2026-04-17 HIGH-4).

**Trimmed Mean.** `(n − 1) // 2` safety clamp (audit 2026-04-09 round 1 "beta floor"), warning on no-trim case, dtype preservation (round 2).

**RFA.** NaN/Inf client filter (audit 2026-03-25 LOW-23, 2026-04-09 round 2), dtype preservation (round 2).

**Reputation.** Smooth `[-√2, √2] → [0, 1]` trust mapping (audit 2026-03-25 HIGH-3 "smooth trust clipping"), `trust_scores` NaN guard (audit 2026-04-09 round 3), `rep_hist` NaN padding (audit 2026-04-09 round 3), real-client-ID mapping (audit 2026-03-25 MEDIUM-12), seeded `KMeans`.

**Attacks.** `label_flipping` `numpy.floating` guard (audit 2026-04-09 round 3), `gradient_scaling` null-global `ValueError` + FP16 NaN clamp + dtype preservation (audit 2026-04-17 MEDIUM).

---

## Appendix: What Matches As-Is

For completeness, the overlapping components whose core math *is* equivalent between the two projects:

- **Trimmed Mean aggregation**: given the same `trim_ratio` (IntelliFL) / `beta` (FEDSIM) and same n, both produce identical aggregated weights. Divergence comes from IntelliFL's added permanent client removal, not from the aggregation itself.
- **Gradient Scaling attack**: formula is identical in FP32; numerics diverge only at FP16 with large `scale_factor`, where FEDSIM clamps and IntelliFL produces NaN.

All other overlapping components diverge in at least one load-bearing way.
