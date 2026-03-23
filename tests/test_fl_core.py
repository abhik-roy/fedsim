"""Tests for fl_core — lightweight Flower replacement."""
import numpy as np
from fl_core import FedAvg, FitRes, Strategy, Status, Code, NDArrays


class TestFitRes:
    def test_fitres_holds_ndarrays_directly(self):
        params = [np.array([1.0, 2.0]), np.array([3.0])]
        res = FitRes(parameters=params, num_examples=100)
        assert res.parameters is params
        assert res.num_examples == 100

    def test_fitres_default_status(self):
        res = FitRes(parameters=[], num_examples=0)
        assert res.status.code == Code.OK


class TestFedAvg:
    def test_weighted_average_two_clients(self):
        """FedAvg computes weighted mean by num_examples."""
        results = [
            (0, FitRes(parameters=[np.array([1.0, 0.0])], num_examples=3)),
            (1, FitRes(parameters=[np.array([0.0, 1.0])], num_examples=1)),
        ]
        strategy = FedAvg(initial_parameters=[np.zeros(2)])
        aggregated, metrics = strategy.aggregate_fit(1, results, [])
        expected = np.array([0.75, 0.25])
        np.testing.assert_allclose(aggregated[0], expected)

    def test_weighted_average_equal_weights(self):
        results = [
            (0, FitRes(parameters=[np.array([2.0])], num_examples=10)),
            (1, FitRes(parameters=[np.array([4.0])], num_examples=10)),
        ]
        strategy = FedAvg(initial_parameters=[np.zeros(1)])
        aggregated, _ = strategy.aggregate_fit(1, results, [])
        np.testing.assert_allclose(aggregated[0], np.array([3.0]))

    def test_empty_results_returns_none(self):
        strategy = FedAvg(initial_parameters=[np.zeros(2)])
        result, metrics = strategy.aggregate_fit(1, [], [])
        assert result is None

    def test_constructor_accepts_flower_kwargs(self):
        """FedAvg silently accepts kwargs it doesn't use (Flower compat)."""
        strategy = FedAvg(
            initial_parameters=None,
            fraction_fit=0.5, fraction_evaluate=0.8,
            min_fit_clients=3, min_evaluate_clients=2,
            min_available_clients=5, mu=0.01,
        )
        assert strategy is not None

    def test_multi_layer_aggregation(self):
        """FedAvg handles multiple layers."""
        results = [
            (0, FitRes(parameters=[np.array([1.0]), np.array([10.0])], num_examples=1)),
            (1, FitRes(parameters=[np.array([3.0]), np.array([20.0])], num_examples=1)),
        ]
        strategy = FedAvg(initial_parameters=[np.zeros(1), np.zeros(1)])
        aggregated, _ = strategy.aggregate_fit(1, results, [])
        np.testing.assert_allclose(aggregated[0], np.array([2.0]))
        np.testing.assert_allclose(aggregated[1], np.array([15.0]))
