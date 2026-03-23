"""Template for custom LR scheduler plugins.

Copy this file, rename it, and implement the build() function.
The scheduler will be auto-discovered and appear in the dashboard.

Example:
    # warmup_cosine.py
    NAME = "Warmup Cosine"
    PARAMS = {
        "warmup_steps": {"type": "int", "default": 5, "min": 1, "max": 50,
                         "label": "Warmup Steps"},
    }

    def build(optimizer, **kwargs):
        warmup_steps = kwargs.get("warmup_steps", 5)
        # Return a PyTorch LR scheduler
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=warmup_steps
        )
"""
NAME = "My Custom Scheduler"

PARAMS = {}

def build(optimizer, **kwargs):
    """Build and return a torch.optim.lr_scheduler instance."""
    raise NotImplementedError("Copy this template and implement build()")
