"""FedProx strategy plugin — FedAvg + proximal term (Li et al., MLSys 2020).

FedProx addresses client drift under non-IID data by adding a proximal
regularization term to each client's local objective:

    h_k(w; w^t) = F_k(w) + (mu/2) * ||w - w^t||^2

where w^t is the global model. Aggregation is identical to FedAvg.

Reference: https://arxiv.org/abs/1812.06127
"""
import torch
import torch.nn.functional as F
from fl_core import FedAvg

NAME = "FedProx"
DESCRIPTION = "FedAvg + proximal term for heterogeneous networks (Li et al., MLSys 2020)"
COMPATIBLE_TASKS = ["image_classification", "text_classification"]

PARAMS = {
    "mu": {
        "type": "float",
        "default": 0.01,
        "min": 0.0,
        "max": 10.0,
        "step": 0.01,
        "label": "Proximal \u03bc",
    },
}


def build(initial_parameters, num_clients, num_malicious=0, **kwargs):
    """Return a FedAvg strategy (aggregation is identical to FedAvg).

    mu is consumed by train_step, not aggregation. We filter it out
    to avoid passing an unexpected kwarg to FedAvg.
    """
    filtered = {k: v for k, v in kwargs.items() if k != "mu"}
    return FedAvg(initial_parameters=initial_parameters, **filtered)


def train_step(model, batch, optimizer, device, mu=0.01, **kwargs):
    """Custom training step with proximal regularization.

    On the first call per client, snapshots the current model parameters
    as the global reference. Each client receives a fresh copy.deepcopy'd
    model from the runner, so the snapshot is guaranteed to be the global
    weights with no prior optimizer steps.
    """
    # Snapshot global params on first call per client
    if not hasattr(model, '_fedprox_global_params'):
        model._fedprox_global_params = [p.clone().detach() for p in model.parameters()]

    images, labels = batch
    images, labels = images.to(device), labels.to(device)

    optimizer.zero_grad()
    output = model(images)
    loss = F.cross_entropy(output, labels)

    # Proximal term: (mu/2) * ||w - w_global||^2
    if mu > 0:
        proximal = sum(
            ((p - gp) ** 2).sum()
            for p, gp in zip(model.parameters(), model._fedprox_global_params)
        )
        loss = loss + (mu / 2.0) * proximal

    loss.backward()
    optimizer.step()

    pred = output.argmax(dim=1)
    acc = (pred == labels).float().mean().item()
    return {"loss": loss.item(), "accuracy": acc}
