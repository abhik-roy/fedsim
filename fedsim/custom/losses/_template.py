"""
=============================================================================
FEDSIM — Custom Loss Function Plugin Template
=============================================================================

HOW IT WORKS
------------
Drop a .py file in this folder (custom/losses/) and it will be auto-
discovered. The loss will appear in the UI dropdown as "[Plugin] <NAME>".

REQUIRED EXPORTS
~~~~~~~~~~~~~~~~
  NAME     : str   — Human-readable name shown in the dropdown.
  build()  : func  — Factory that returns a torch.nn.Module loss function.

OPTIONAL EXPORTS
~~~~~~~~~~~~~~~~
  DESCRIPTION : str  — One-line tooltip shown in the UI.
  PARAMS      : dict — Hyperparameters exposed as UI controls. Values are
                       passed as **kwargs to build().

build() SIGNATURE
~~~~~~~~~~~~~~~~~
  def build(**kwargs):
      ...
      return loss_module   # torch.nn.Module with forward(inputs, targets)

  The returned module is used as:
      loss_fn = build(**kwargs)
      loss = loss_fn(model_output, target_labels)

  So its forward() must accept (predictions, targets) and return a scalar.

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
  - The standard losses (CrossEntropyLoss, etc.) are already built in.
    Use plugins for specialized losses like Focal Loss, Label Smoothing,
    contrastive losses, etc.
  - Your loss class must subclass nn.Module so it can be serialized and
    moved to device along with the model.
  - forward() receives raw logits (not softmax) and integer class labels
    in the standard classification workflow.

COMPLETE EXAMPLE — Focal Loss for class-imbalanced datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
(Uncomment and save as e.g. focal_loss.py to use.)

  import torch
  import torch.nn as nn
  import torch.nn.functional as F

  NAME = "Focal Loss"
  DESCRIPTION = "Down-weights easy examples; useful for class imbalance."

  PARAMS = {
      "gamma": {
          "type": "float", "default": 2.0, "min": 0.0, "max": 5.0,
          "step": 0.5, "label": "Focusing Parameter (γ)",
      },
      "alpha": {
          "type": "float", "default": 0.25, "min": 0.0, "max": 1.0,
          "step": 0.05, "label": "Balance Factor (α)",
      },
  }

  class FocalLoss(nn.Module):
      def __init__(self, gamma=2.0, alpha=0.25):
          super().__init__()
          self.gamma = gamma
          self.alpha = alpha

      def forward(self, inputs, targets):
          # inputs: (batch, num_classes) raw logits
          # targets: (batch,) integer class labels
          ce_loss = F.cross_entropy(inputs, targets, reduction="none")
          pt = torch.exp(-ce_loss)   # probability of correct class
          focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
          return focal_loss.mean()

  def build(**kwargs):
      gamma = kwargs.get("gamma", PARAMS["gamma"]["default"])
      alpha = kwargs.get("alpha", PARAMS["alpha"]["default"])
      return FocalLoss(gamma=gamma, alpha=alpha)

=============================================================================
"""

import torch.nn as nn

# ---------------------------------------------------------------------------
# Required exports
# ---------------------------------------------------------------------------
NAME = "My Custom Loss"

# ---------------------------------------------------------------------------
# Optional exports (uncomment and edit as needed)
# ---------------------------------------------------------------------------
# DESCRIPTION = "Short description shown as tooltip in the UI."
# PARAMS = {
#     "label_smoothing": {
#         "type": "float", "default": 0.1, "min": 0.0, "max": 0.5,
#         "step": 0.05, "label": "Label Smoothing Factor",
#     },
# }


def build(**kwargs) -> nn.Module:
    """Return a torch.nn.Module loss function.

    The returned module's forward() must accept (predictions, targets) and
    return a scalar loss tensor.

    Args:
        **kwargs: Values from PARAMS (if defined), populated by the UI.

    Returns:
        An nn.Module usable as a loss function.
    """
    raise NotImplementedError("Copy this template and implement build()")
