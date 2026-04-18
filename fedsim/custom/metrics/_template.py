"""
=============================================================================
FEDSIM — Custom Evaluation Metric Plugin Template
=============================================================================

HOW IT WORKS
------------
Drop a .py file in this folder (custom/metrics/) and it will be auto-
discovered. The metric values are computed after each aggregation round and
displayed in the dashboard. The metric name appears as "[Plugin] <NAME>".

REQUIRED EXPORTS
~~~~~~~~~~~~~~~~
  NAME       : str   — Human-readable name shown in the dashboard.
  CHART_TYPE : str   — How to visualize: "line" (per-round time series),
                       "bar" (final summary), or "scalar" (single number).
  compute()  : func  — Evaluates the model and returns metric values.

OPTIONAL EXPORTS
~~~~~~~~~~~~~~~~
  DESCRIPTION : str  — One-line tooltip shown in the UI.
  PARAMS      : dict — Configurable parameters exposed in the UI. Values
                       are passed as **kwargs to compute().

compute() SIGNATURE
~~~~~~~~~~~~~~~~~~~
  def compute(model, dataloader, device, **kwargs):
      ...
      return {"metric_name": float_value, ...}

  - model      : nn.Module — the global model after aggregation.
  - dataloader : DataLoader — test set DataLoader for evaluation.
  - device     : torch.device — device to run inference on.
  - **kwargs   : dict — values from PARAMS plus any plugin_params["metrics"]
                 overrides from the config.

  Must return a dict mapping metric names (str) to float values.  Each key
  becomes a separate series in the dashboard chart.

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
  - compute() is called once per round with the freshly aggregated global
    model. Keep it fast — avoid training-mode operations.
  - Always call model.eval() and use torch.no_grad() for inference.
  - You can return multiple metrics in a single plugin (e.g., both
    macro and weighted F1). Each key in the returned dict becomes its
    own chart series.
  - For sklearn metrics, collect all predictions first, then compute.

COMPLETE EXAMPLE — F1 Score (macro and weighted)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
(Uncomment and save as e.g. f1_score.py to use.)

  import torch
  from sklearn.metrics import f1_score

  NAME = "F1 Score"
  CHART_TYPE = "line"
  DESCRIPTION = "Per-round macro and weighted F1 score on the test set."

  PARAMS = {
      "average": {
          "type": "select", "default": "both",
          "options": ["macro", "weighted", "both"],
          "label": "Averaging Method",
      },
  }

  def compute(model, dataloader, device, **kwargs):
      average = kwargs.get("average", PARAMS["average"]["default"])

      model.to(device)
      model.eval()
      all_preds, all_labels = [], []

      with torch.no_grad():
          for inputs, labels in dataloader:
              inputs = inputs.to(device)
              outputs = model(inputs)
              preds = outputs.argmax(dim=1).cpu()
              all_preds.extend(preds.tolist())
              all_labels.extend(labels.tolist())

      results = {}
      if average in ("macro", "both"):
          results["f1_macro"] = f1_score(
              all_labels, all_preds, average="macro", zero_division=0,
          )
      if average in ("weighted", "both"):
          results["f1_weighted"] = f1_score(
              all_labels, all_preds, average="weighted", zero_division=0,
          )
      return results

=============================================================================
"""

import torch

# ---------------------------------------------------------------------------
# Required exports
# ---------------------------------------------------------------------------
NAME = "My Custom Metric"
CHART_TYPE = "line"   # "line" = per-round, "bar" = final summary, "scalar" = single value

# ---------------------------------------------------------------------------
# Optional exports (uncomment and edit as needed)
# ---------------------------------------------------------------------------
# DESCRIPTION = "Short description shown as tooltip in the UI."
# PARAMS = {
#     "top_k": {
#         "type": "int", "default": 5, "min": 1, "max": 20,
#         "label": "Top-K for Accuracy",
#     },
# }


def compute(model, dataloader, device, **kwargs) -> dict:
    """Evaluate the model and return metric values.

    Args:
        model:      The global model after aggregation (nn.Module).
        dataloader: Test DataLoader for evaluation.
        device:     torch.device to run inference on.
        **kwargs:   Values from PARAMS (if defined) and plugin_params config.

    Returns:
        Dict mapping metric names (str) to float values.
    """
    raise NotImplementedError("Copy this template and implement compute()")
