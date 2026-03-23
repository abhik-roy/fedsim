"""
=============================================================================
FEDSIM — Custom Aggregation Strategy Plugin Template
=============================================================================

HOW IT WORKS
------------
Drop a .py file in this folder (custom/strategies/) and it will be auto-
discovered. The strategy will appear in the UI dropdown as "[Plugin] <NAME>".

Strategies control how the server aggregates model updates from clients each
round. They are fl_core Strategy objects.

REQUIRED EXPORTS
~~~~~~~~~~~~~~~~
  NAME     : str   — Human-readable name shown in the dropdown.
  build()  : func  — Factory that returns a fl_core.Strategy.

OPTIONAL EXPORTS
~~~~~~~~~~~~~~~~
  DESCRIPTION : str  — One-line tooltip shown in the UI.
  PARAMS      : dict — Extra hyperparameters exposed as UI controls.

build() SIGNATURE
~~~~~~~~~~~~~~~~~
  def build(initial_parameters, num_clients, num_malicious=0, **kwargs):
      ...
      return strategy   # fl_core.Strategy

  - initial_parameters : list[np.ndarray] — initial model weights as NDArrays;
                         pass this to the base strategy.
  - num_clients        : int — total number of federated clients.
  - num_malicious      : int — number of simulated malicious clients (for
                         Byzantine-robust strategies).
  - **kwargs           : dict — common parameters forwarded by the framework:
      fraction_fit, fraction_evaluate, min_fit_clients,
      min_evaluate_clients, min_available_clients, initial_parameters,
      plus any values from your PARAMS dict.

PARAMS FORMAT
~~~~~~~~~~~~~
  PARAMS = {
      "param_name": {
          "type": "int" | "float" | "select" | "bool",
          "default": <value>,
          "min": <number>,          # int/float only
          "max": <number>,          # int/float only
          "step": <number>,         # int/float only (optional)
          "options": ["a", "b"],    # select only
          "label": "UI Label",      # human-friendly label
      },
  }

TIPS
~~~~
  - The simplest approach is to subclass fl_core.FedAvg and override
    aggregate_fit().  FedAvg handles client sampling, parameter serialization,
    and basic weighted averaging for you.
  - Parameters are already NDArrays (list[np.ndarray]) — no conversion needed.
  - For Byzantine-robust strategies, you can use num_malicious to inform
    trimming ratios, trust thresholds, etc.
  - See strategies/reputation.py for a full real-world example with
    per-client scoring, asymmetric reputation updates, and top-k selection.

COMPLETE EXAMPLE — Trimmed-Mean aggregation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
(Uncomment and save as e.g. trimmed_mean.py to use.)

  import numpy as np
  from fl_core import FedAvg, FitRes

  NAME = "Trimmed Mean"
  DESCRIPTION = "Coordinate-wise trimmed mean — discards top/bottom β updates."

  PARAMS = {
      "trim_fraction": {
          "type": "float", "default": 0.1, "min": 0.0, "max": 0.45,
          "step": 0.05, "label": "Trim Fraction (β)",
      },
  }

  class TrimmedMeanStrategy(FedAvg):
      def __init__(self, trim_fraction=0.1, **kwargs):
          super().__init__(**kwargs)
          self.trim_fraction = trim_fraction

      def aggregate_fit(self, server_round, results, failures):
          if not results:
              return None, {}

          # Collect all client weight arrays (already NDArrays)
          weights = [r.parameters for _, r in results]
          n = len(weights)
          trim_count = max(1, int(n * self.trim_fraction))

          # Coordinate-wise trimmed mean per layer
          aggregated = []
          for layer_idx in range(len(weights[0])):
              stacked = np.stack([w[layer_idx] for w in weights], axis=0)
              sorted_layer = np.sort(stacked, axis=0)
              trimmed = sorted_layer[trim_count : n - trim_count]
              aggregated.append(trimmed.mean(axis=0))

          return aggregated, {}

  def build(initial_parameters, num_clients, num_malicious=0, **kwargs):
      trim_frac = kwargs.pop("trim_fraction",
                             PARAMS["trim_fraction"]["default"])
      return TrimmedMeanStrategy(
          trim_fraction=trim_frac,
          initial_parameters=initial_parameters,
          **kwargs,
      )

=============================================================================
"""

from fl_core import Strategy

# ---------------------------------------------------------------------------
# Required exports
# ---------------------------------------------------------------------------
NAME = "My Custom Strategy"

# ---------------------------------------------------------------------------
# Optional exports (uncomment and edit as needed)
# ---------------------------------------------------------------------------
# DESCRIPTION = "Short description shown as tooltip in the UI."
# PARAMS = {
#     "selection_fraction": {
#         "type": "float", "default": 0.6, "min": 0.1, "max": 1.0,
#         "step": 0.1, "label": "Client Selection Fraction",
#     },
# }
#
# ---------------------------------------------------------------------------
# Optional: Client-side training overrides
# ---------------------------------------------------------------------------
# If your strategy requires modified client training (e.g., FedProx proximal
# term, SCAFFOLD control variates), provide train_step or fit.
# These are used when no model plugin provides its own training override.
# Model plugins always take priority over strategy plugins.
#
# GRADIENT CLIPPING CONTRACT:
# train_step owns the full step: zero_grad, backward, clip (if desired), and
# optimizer.step(). The runner's clip_grad_norm_ after train_step is a legacy
# safety net but fires after optimizer.step() and is effectively a no-op.
# If you need clipping, do it INSIDE train_step before optimizer.step().
#
# KWARGS:
# Values from your PARAMS dict are passed to BOTH build() and train_step()/fit().
# If a parameter is only needed by train_step, filter it out in build() — do
# NOT use kwargs.pop() as the dict may be shared.
#
# Tier 2 (per-step, runner manages epoch/batch loops):
#   def train_step(model, batch, optimizer, device, **kwargs) -> dict[str, float]:
#       """Return dict with at least 'loss'."""
#
# Tier 1 (full control, plugin manages epochs):
#   def fit(model, dataloader, optimizer, device, local_epochs, **kwargs) -> dict[str, float]:
#       """Return dict with at least 'loss'."""


def build(initial_parameters, num_clients: int, num_malicious: int = 0, **kwargs) -> Strategy:
    """Return a fl_core Strategy instance.

    Args:
        initial_parameters: Initial model weights as list[np.ndarray].
        num_clients:        Total number of federated clients in the simulation.
        num_malicious:      Number of simulated malicious/Byzantine clients.
        **kwargs:           Common params (fraction_fit, etc.) plus any values
                            from PARAMS.

    Returns:
        A fl_core.Strategy (e.g. FedAvg subclass).
    """
    raise NotImplementedError("Copy this template and implement build()")
