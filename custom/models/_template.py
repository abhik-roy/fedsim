"""
=============================================================================
FEDSIM — Custom Model Plugin Template
=============================================================================

HOW IT WORKS
------------
Drop a .py file in this folder (custom/models/) and it will be auto-
discovered. The model will appear in the UI dropdown as "[Plugin] <NAME>".

REQUIRED EXPORTS
~~~~~~~~~~~~~~~~
  NAME     : str   — Human-readable name shown in the dropdown.
  build()  : func  — Factory that returns a torch.nn.Module instance.

OPTIONAL EXPORTS
~~~~~~~~~~~~~~~~
  DESCRIPTION : str  — One-line tooltip shown in the UI.
  PARAMS      : dict — Hyperparameters exposed as UI controls. Values are
                       passed as **kwargs to build().

build() SIGNATURE
~~~~~~~~~~~~~~~~~
  def build(dataset_info: dict, **kwargs) -> nn.Module:
      ...
      return model   # torch.nn.Module

  dataset_info is a dict with the following keys:
      task_type      : str  — e.g. "image_classification", "text_classification"
      num_classes    : int  — number of output classes
      input_channels : int  — e.g. 3 for RGB, 1 for grayscale (images)
      image_size     : int  — spatial dimension for images (assumes square)
      input_size     : int  — flattened input dim (input_channels * image_size^2)
      vocab_size     : int | None — vocabulary size (text datasets only)
      seq_length     : int | None — sequence length (text datasets only)

  **kwargs contains any values from PARAMS that the user configured in the UI.

OVERRIDE TIERS (optional)
~~~~~~~~~~~~~~~~~~~~~~~~~
  Model plugins MAY export any of the following functions to customize the
  training and evaluation loop. If not provided, the default classification
  loop is used (cross-entropy loss + argmax accuracy).

  The tiers are checked in order; the first match wins.

  TRAINING:
    Tier 1 — Full epoch control (you manage the epoch & batch loop):
      def fit(model, dataloader, optimizer, device, local_epochs, **kwargs) -> dict[str, float]:
          # Must return a dict with at least {"loss": <float>}
          ...

    Tier 2 — Per-step control (runner manages epochs & batches):
      def train_step(model, batch, optimizer, device, **kwargs) -> dict[str, float]:
          # Must return a dict with at least {"loss": <float>}
          ...

  EVALUATION:
    Tier 1 — Full eval control (you iterate over the dataloader):
      def evaluate(model, dataloader, device, **kwargs) -> dict[str, float]:
          # Must return a dict with at least {"loss": <float>}
          ...

    Tier 2 — Per-step eval (runner iterates batches & aggregates):
      def eval_step(model, batch, device, **kwargs) -> dict[str, float]:
          # Must return a dict with at least {"loss": <float>}
          # Values are averaged weighted by batch size.
          ...

    Tier 3 (default) — No override needed:
      The runner uses cross-entropy loss and argmax accuracy. This is
      appropriate for standard image/text classification tasks.

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

  Inside build(), retrieve them with:
      value = kwargs.get("param_name", PARAMS["param_name"]["default"])

TIPS
~~~~
  - The model receives raw tensors from the dataset. For image datasets the
    input is (batch, channels, H, W). For text datasets it may be
    (batch, 1, seq_len) — squeeze the channel dim if needed (see TextCNN).
  - If your model is only compatible with certain datasets, document that
    in DESCRIPTION.
  - You can import torchvision.models or timm for pre-built architectures.

COMPLETE EXAMPLE — simple 2-layer CNN
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
(Uncomment and save as e.g. my_simple_cnn.py to use.)

  import torch.nn as nn

  NAME = "Simple 2-Layer CNN"
  DESCRIPTION = "Minimal CNN for small image datasets (MNIST, CIFAR-10)."

  PARAMS = {
      "hidden_channels": {
          "type": "int", "default": 32, "min": 8, "max": 256, "step": 8,
          "label": "Hidden Channels",
      },
      "dropout": {
          "type": "float", "default": 0.25, "min": 0.0, "max": 0.8,
          "step": 0.05, "label": "Dropout Rate",
      },
  }

  def build(dataset_info, **kwargs):
      input_channels = dataset_info["input_channels"]
      num_classes = dataset_info["num_classes"]
      image_size = dataset_info["image_size"]
      hid = kwargs.get("hidden_channels", PARAMS["hidden_channels"]["default"])
      drop = kwargs.get("dropout", PARAMS["dropout"]["default"])

      # After two 3x3 convs with stride 2, spatial dim = image_size // 4
      spatial = image_size // 4

      return nn.Sequential(
          nn.Conv2d(input_channels, hid, 3, stride=2, padding=1),
          nn.ReLU(),
          nn.Conv2d(hid, hid * 2, 3, stride=2, padding=1),
          nn.ReLU(),
          nn.Flatten(),
          nn.Dropout(drop),
          nn.Linear(hid * 2 * spatial * spatial, num_classes),
      )

  # --- Optional training/eval overrides (uncomment to customize) ---
  #
  # import torch
  # import torch.nn.functional as F
  #
  # def train_step(model, batch, optimizer, device, **kwargs):
  #     images, labels = batch
  #     images, labels = images.to(device), labels.to(device)
  #     optimizer.zero_grad()
  #     output = model(images)
  #     loss = F.cross_entropy(output, labels)
  #     loss.backward()
  #     optimizer.step()
  #     acc = (output.argmax(1) == labels).float().mean().item()
  #     return {"loss": loss.item(), "accuracy": acc}
  #
  # def fit(model, dataloader, optimizer, device, local_epochs, **kwargs):
  #     total_loss, total_correct, total_samples = 0.0, 0, 0
  #     for epoch in range(local_epochs):
  #         for images, labels in dataloader:
  #             images, labels = images.to(device), labels.to(device)
  #             optimizer.zero_grad()
  #             output = model(images)
  #             loss = F.cross_entropy(output, labels)
  #             loss.backward()
  #             optimizer.step()
  #             total_loss += loss.item() * labels.size(0)
  #             total_correct += (output.argmax(1) == labels).sum().item()
  #             total_samples += labels.size(0)
  #     return {
  #         "loss": total_loss / max(total_samples, 1),
  #         "accuracy": total_correct / max(total_samples, 1),
  #     }
  #
  # def eval_step(model, batch, device, **kwargs):
  #     images, labels = batch
  #     images, labels = images.to(device), labels.to(device)
  #     output = model(images)
  #     loss = F.cross_entropy(output, labels).item()
  #     acc = (output.argmax(1) == labels).float().mean().item()
  #     return {"loss": loss, "accuracy": acc}
  #
  # def evaluate(model, dataloader, device, **kwargs):
  #     total_loss, total_correct, total_samples = 0.0, 0, 0
  #     for images, labels in dataloader:
  #         images, labels = images.to(device), labels.to(device)
  #         output = model(images)
  #         total_loss += F.cross_entropy(output, labels).item() * labels.size(0)
  #         total_correct += (output.argmax(1) == labels).sum().item()
  #         total_samples += labels.size(0)
  #     return {
  #         "loss": total_loss / max(total_samples, 1),
  #         "accuracy": total_correct / max(total_samples, 1),
  #     }

=============================================================================
"""

import torch.nn as nn

# ---------------------------------------------------------------------------
# Required exports
# ---------------------------------------------------------------------------
NAME = "My Custom Model"

# ---------------------------------------------------------------------------
# Optional exports (uncomment and edit as needed)
# ---------------------------------------------------------------------------
# DESCRIPTION = "Short description shown as tooltip in the UI."
# PARAMS = {
#     "hidden_size": {
#         "type": "int", "default": 128, "min": 16, "max": 1024, "step": 16,
#         "label": "Hidden Layer Size",
#     },
#     "activation": {
#         "type": "select", "default": "relu",
#         "options": ["relu", "gelu", "silu"],
#         "label": "Activation Function",
#     },
# }


def build(dataset_info: dict, **kwargs) -> nn.Module:
    """Return a torch.nn.Module instance.

    Args:
        dataset_info: Dict with keys: task_type, num_classes, input_channels,
                      image_size, input_size, vocab_size, seq_length.
        **kwargs:     Values from PARAMS (if defined), populated by the UI.

    Returns:
        An nn.Module ready for training.
    """
    raise NotImplementedError("Copy this template and implement build()")
