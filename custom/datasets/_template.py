"""
=============================================================================
FEDSIM — Custom Dataset Plugin Template
=============================================================================

HOW IT WORKS
------------
Drop a .py file in this folder (custom/datasets/) and it will be auto-
discovered. The dataset will appear in the UI dropdown as "[Plugin] <NAME>".

REQUIRED EXPORTS
~~~~~~~~~~~~~~~~
  NAME           : str   — Human-readable name shown in the dropdown.
  NUM_CLASSES    : int   — Number of target classes (e.g. 10 for CIFAR-10).
  INPUT_CHANNELS : int   — 3 for RGB images, 1 for grayscale or text.
  IMAGE_SIZE     : int   — Spatial dimension for images (assumes square),
                           OR sequence length for text datasets.
  load()         : func  — Must return (train_dataset, test_dataset), both
                           standard torch.utils.data.Dataset objects.

OPTIONAL EXPORTS
~~~~~~~~~~~~~~~~
  TASK_TYPE      : str   — Task type identifier for the dataset. Determines
                           which training loop and evaluation logic to use.
                           Common values: "image_classification",
                           "text_classification", "language_modeling",
                           "regression". Defaults to "image_classification"
                           if not set.
  SEQ_LENGTH     : int   — Sequence length for text datasets. Used to build
                           the dataset_info dict passed to model plugins.
                           Defaults to IMAGE_SIZE if not set.
  VOCAB_SIZE     : int   — Required for text datasets so the model plugin
                           can size its embedding layer.  Set this inside
                           load() after building the vocabulary.
  DESCRIPTION    : str   — One-line description shown as a tooltip.
  PARAMS         : dict  — Configurable parameters exposed in the UI.
                           See the PARAMS format section below.

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

  The values are passed as **kwargs to load() when PARAMS is defined.

TIPS
~~~~
  - load() is called once per simulation run.  You can download / cache data
    inside load() — see the AG News plugin for a real-world example.
  - The returned Dataset objects must yield (input_tensor, label) tuples.
    For images: input_tensor shape = (C, H, W).
    For text:   input_tensor shape = (seq_len,) with dtype=torch.long.
  - For non-IID partitioning, the framework reads dataset.targets (a list of
    int labels).  If your Dataset doesn't have a .targets attribute, add one
    (e.g. self.targets = [label for _, label in samples]).

COMPLETE EXAMPLE — simplified CIFAR-like dataset using random data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
(Uncomment the block below and save as e.g. my_random_dataset.py to test.)

  import torch
  from torch.utils.data import TensorDataset

  NAME = "Random CIFAR-like"
  NUM_CLASSES = 10
  INPUT_CHANNELS = 3
  IMAGE_SIZE = 32
  DESCRIPTION = "Random noise dataset for testing (1000 train / 200 test)"

  PARAMS = {
      "num_train": {
          "type": "int", "default": 1000, "min": 100, "max": 50000,
          "step": 100, "label": "Training Samples",
      },
      "num_test": {
          "type": "int", "default": 200, "min": 50, "max": 10000,
          "step": 50, "label": "Test Samples",
      },
  }

  def load(**kwargs):
      n_train = kwargs.get("num_train", PARAMS["num_train"]["default"])
      n_test  = kwargs.get("num_test",  PARAMS["num_test"]["default"])

      train_x = torch.randn(n_train, INPUT_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
      train_y = torch.randint(0, NUM_CLASSES, (n_train,))
      test_x  = torch.randn(n_test, INPUT_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
      test_y  = torch.randint(0, NUM_CLASSES, (n_test,))

      train_ds = TensorDataset(train_x, train_y)
      test_ds  = TensorDataset(test_x, test_y)

      # Attach .targets so non-IID partitioning works
      train_ds.targets = train_y.tolist()
      test_ds.targets  = test_y.tolist()

      return train_ds, test_ds

=============================================================================
"""

from typing import Tuple
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Required exports
# ---------------------------------------------------------------------------
NAME = "My Custom Dataset"
NUM_CLASSES = 10
INPUT_CHANNELS = 3   # 3 = RGB, 1 = grayscale or text
IMAGE_SIZE = 32       # square image side length, or sequence length for text

# ---------------------------------------------------------------------------
# Optional exports (uncomment and edit as needed)
# ---------------------------------------------------------------------------
# VOCAB_SIZE = 25002           # Only needed for text datasets
# DESCRIPTION = "Short description shown as tooltip in the UI."
# PARAMS = {
#     "augment": {
#         "type": "bool", "default": True,
#         "label": "Enable Data Augmentation",
#     },
# }


def load(**kwargs) -> Tuple[Dataset, Dataset]:
    """Return (train_dataset, test_dataset).

    Both must be torch.utils.data.Dataset objects that yield
    (input_tensor, label_int) pairs.

    Args:
        **kwargs: Values from PARAMS (if defined), populated by the UI.

    Returns:
        Tuple of (train_dataset, test_dataset).
    """
    raise NotImplementedError("Copy this template and implement load()")
