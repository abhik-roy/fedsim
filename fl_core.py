"""Lightweight FL core types and aggregation — replaces flwr dependency."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

NDArrays = list[np.ndarray]


class Code:
    OK = 0


@dataclass
class Status:
    code: int = Code.OK
    message: str = ""


@dataclass
class FitRes:
    """Result of a client fit round."""
    parameters: NDArrays
    num_examples: int
    metrics: dict[str, Any] = field(default_factory=dict)
    status: Status = field(default_factory=Status)


class Strategy:
    """Abstract base for aggregation strategies."""

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[int, FitRes]],
        failures: list,
    ) -> tuple[NDArrays | None, dict[str, Any]]:
        raise NotImplementedError


class FedAvg(Strategy):
    """Federated Averaging — weighted mean of client parameters."""

    def __init__(
        self,
        *,
        initial_parameters: NDArrays | None = None,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        **kwargs,
    ):
        self.initial_parameters = initial_parameters
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[int, FitRes]],
        failures: list,
    ) -> tuple[NDArrays | None, dict[str, Any]]:
        if not results:
            return None, {}

        total_examples = sum(r.num_examples for _, r in results)
        if total_examples == 0:
            return None, {}

        aggregated = [
            np.zeros(param.shape, dtype=np.float64)
            for param in results[0][1].parameters
        ]
        for _, fit_res in results:
            weight = fit_res.num_examples / total_examples
            for i, param in enumerate(fit_res.parameters):
                aggregated[i] += param.astype(np.float64) * weight

        return aggregated, {}
