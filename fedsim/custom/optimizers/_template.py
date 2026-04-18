"""
=============================================================================
FEDSIM — Custom Optimizer Plugin Template
=============================================================================

HOW IT WORKS
------------
Drop a .py file in this folder (custom/optimizers/) and it will be auto-
discovered. The optimizer will appear in the UI dropdown as
"[Plugin] <NAME>".

REQUIRED EXPORTS
~~~~~~~~~~~~~~~~
  NAME     : str   — Human-readable name shown in the dropdown.
  build()  : func  — Factory that returns a torch.optim.Optimizer instance.

OPTIONAL EXPORTS
~~~~~~~~~~~~~~~~
  DESCRIPTION : str  — One-line tooltip shown in the UI.
  PARAMS      : dict — Hyperparameters exposed as UI controls. Values are
                       passed as **kwargs to build().

build() SIGNATURE
~~~~~~~~~~~~~~~~~
  def build(model_params, lr=0.001, **kwargs):
      ...
      return optimizer   # torch.optim.Optimizer

  - model_params : iterable — model.parameters(); pass directly to the
                   optimizer constructor.
  - lr           : float — learning rate selected in the main UI panel.
  - **kwargs     : dict — any additional values from your PARAMS dict.

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
  - The standard optimizers (SGD, Adam, AdamW) are already built in.
    Use plugins for custom schedulers, exotic optimizers (LAMB, LARS,
    Lookahead, SAM), or wrappers that combine an optimizer with warmup.
  - The learning rate slider in the main UI is passed as the `lr` argument.
    Your PARAMS should NOT duplicate the `lr` parameter.
  - Each federated client creates its own optimizer every round, so any
    stateful warmup logic needs to be self-contained.

COMPLETE EXAMPLE — SGD with momentum and optional warmup wrapper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
(Uncomment and save as e.g. sgd_warmup.py to use.)

  import torch.optim as optim

  NAME = "SGD + Warmup"
  DESCRIPTION = "SGD with momentum and linear LR warmup for N steps."

  PARAMS = {
      "momentum": {
          "type": "float", "default": 0.9, "min": 0.0, "max": 0.99,
          "step": 0.05, "label": "Momentum",
      },
      "weight_decay": {
          "type": "float", "default": 1e-4, "min": 0.0, "max": 0.1,
          "step": 0.0001, "label": "Weight Decay",
      },
      "nesterov": {
          "type": "bool", "default": True,
          "label": "Nesterov Momentum",
      },
      "warmup_steps": {
          "type": "int", "default": 100, "min": 0, "max": 5000,
          "step": 50, "label": "Warmup Steps",
      },
  }

  def build(model_params, lr=0.001, **kwargs):
      momentum     = kwargs.get("momentum",     PARAMS["momentum"]["default"])
      weight_decay = kwargs.get("weight_decay",  PARAMS["weight_decay"]["default"])
      nesterov     = kwargs.get("nesterov",      PARAMS["nesterov"]["default"])

      optimizer = optim.SGD(
          model_params, lr=lr,
          momentum=momentum,
          weight_decay=weight_decay,
          nesterov=nesterov,
      )

      # Optionally attach a warmup scheduler (client code can call
      # scheduler.step() each batch if it checks for optimizer.scheduler)
      warmup_steps = kwargs.get("warmup_steps", PARAMS["warmup_steps"]["default"])
      if warmup_steps > 0:
          scheduler = optim.lr_scheduler.LinearLR(
              optimizer, start_factor=0.1, total_iters=warmup_steps,
          )
          # Stash on the optimizer so the training loop can find it
          optimizer._lr_scheduler = scheduler

      return optimizer

=============================================================================
"""

import torch.optim as optim

# ---------------------------------------------------------------------------
# Required exports
# ---------------------------------------------------------------------------
NAME = "My Custom Optimizer"

# ---------------------------------------------------------------------------
# Optional exports (uncomment and edit as needed)
# ---------------------------------------------------------------------------
# DESCRIPTION = "Short description shown as tooltip in the UI."
# PARAMS = {
#     "momentum": {
#         "type": "float", "default": 0.9, "min": 0.0, "max": 0.99,
#         "step": 0.05, "label": "Momentum",
#     },
#     "weight_decay": {
#         "type": "float", "default": 0.0, "min": 0.0, "max": 0.1,
#         "step": 0.0001, "label": "Weight Decay",
#     },
# }


def build(model_params, lr: float = 0.001, **kwargs) -> optim.Optimizer:
    """Return a torch.optim.Optimizer instance.

    Args:
        model_params: model.parameters() iterable.
        lr:           Learning rate from the main UI slider.
        **kwargs:     Values from PARAMS (if defined), populated by the UI.

    Returns:
        A torch.optim.Optimizer ready for training.
    """
    raise NotImplementedError("Copy this template and implement build()")
