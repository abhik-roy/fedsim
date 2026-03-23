"""Regression MLP model plugin with train_step/eval_step overrides.

Demonstrates the Phase 2 three-tier override system for non-classification tasks.
Uses MSE loss and reports MAE as a custom metric.
"""

import torch
import torch.nn as nn

NAME = "RegressionMLP"
TASK_TYPE = "regression"
COMPATIBLE_TASKS = ["regression"]
DESCRIPTION = "Multi-layer perceptron for regression tasks with MSE loss"

PARAMS = {
    "hidden_dim": {"type": "int", "default": 64, "min": 16, "max": 256, "step": 16,
                   "label": "Hidden Dimension"},
    "num_layers": {"type": "int", "default": 2, "min": 1, "max": 5, "step": 1,
                   "label": "Number of Hidden Layers"},
    "dropout": {"type": "float", "default": 0.1, "min": 0.0, "max": 0.5, "step": 0.05,
                "label": "Dropout Rate"},
}


class RegressionMLPModel(nn.Module):
    def __init__(self, input_size, hidden_dim=64, num_layers=2, dropout=0.1):
        super().__init__()
        layers = []
        in_dim = input_size
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)


def build(dataset_info, **kwargs):
    """Build a regression MLP from dataset_info."""
    input_size = dataset_info.get("input_size", 8)
    hidden_dim = kwargs.get("hidden_dim", PARAMS["hidden_dim"]["default"])
    num_layers = kwargs.get("num_layers", PARAMS["num_layers"]["default"])
    dropout = kwargs.get("dropout", PARAMS["dropout"]["default"])
    return RegressionMLPModel(input_size, hidden_dim, num_layers, dropout)


def train_step(model, batch, optimizer, device, **kwargs):
    """Custom training step using MSE loss."""
    features, targets = batch
    features = features.to(device)
    targets = targets.to(device)

    optimizer.zero_grad()
    predictions = model(features)
    loss = nn.functional.mse_loss(predictions, targets)
    loss.backward()
    optimizer.step()

    return {"loss": loss.item()}


def eval_step(model, batch, device, **kwargs):
    """Custom evaluation step returning MSE and MAE."""
    features, targets = batch
    features = features.to(device)
    targets = targets.to(device)

    predictions = model(features)
    mse = nn.functional.mse_loss(predictions, targets).item()
    mae = torch.mean(torch.abs(predictions - targets)).item()

    return {"loss": mse, "mae": mae}
