"""Tests for the FEDSIM scripting API."""
import os
import json
import tempfile
import pytest
from simulation.runner import SimulationConfig, AttackConfig
from api import Experiment, ExperimentResults, Report


class TestExperiment:
    def test_add_run(self):
        exp = Experiment("test")
        exp.add_run("run1", SimulationConfig(
            model_name="cnn", dataset_name="mnist",
            num_clients=3, num_rounds=1, local_epochs=1,
            learning_rate=0.01, strategies=["fedavg"],
            attack=AttackConfig(),
        ))
        assert len(exp._runs) == 1

    def test_duplicate_name_raises(self):
        exp = Experiment("test")
        cfg = SimulationConfig(model_name="cnn", dataset_name="mnist",
            num_clients=3, num_rounds=1, local_epochs=1,
            learning_rate=0.01, strategies=["fedavg"], attack=AttackConfig())
        exp.add_run("run1", cfg)
        with pytest.raises(ValueError, match="Duplicate"):
            exp.add_run("run1", cfg)

    def test_run_produces_results(self):
        exp = Experiment("test")
        exp.add_run("baseline", SimulationConfig(
            model_name="cnn", dataset_name="mnist",
            num_clients=3, num_rounds=1, local_epochs=1,
            learning_rate=0.01, strategies=["fedavg"],
            attack=AttackConfig(),
        ))
        results = exp.run()
        assert "baseline" in results
        assert len(results) == 1
        assert results.final_accuracy("baseline") > 0

    def test_checkpoint_and_resume(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "checkpoint.json")

            exp1 = Experiment("test")
            exp1.add_run("run1", SimulationConfig(
                model_name="cnn", dataset_name="mnist",
                num_clients=3, num_rounds=1, local_epochs=1,
                learning_rate=0.01, strategies=["fedavg"],
                attack=AttackConfig(),
            ))
            exp1.run(checkpoint_path=path)
            assert os.path.exists(path)

            # Resume — run1 should be skipped
            exp2 = Experiment("test")
            exp2.add_run("run1", SimulationConfig(
                model_name="cnn", dataset_name="mnist",
                num_clients=3, num_rounds=1, local_epochs=1,
                learning_rate=0.01, strategies=["fedavg"],
                attack=AttackConfig(),
            ))
            exp2.add_run("run2", SimulationConfig(
                model_name="cnn", dataset_name="mnist",
                num_clients=3, num_rounds=1, local_epochs=1,
                learning_rate=0.01, strategies=["fedavg"],
                attack=AttackConfig(),
            ))
            results = exp2.run(checkpoint_path=path)
            assert "run1" in results
            assert "run2" in results
            assert len(results) == 2


class TestExperimentResults:
    def test_final_accuracy_empty(self):
        r = ExperimentResults()
        assert str(r.final_accuracy("nonexistent")) == "nan"

    def test_items(self):
        r = ExperimentResults({"a": [1], "b": [2]})
        assert len(list(r.items())) == 2


class TestReport:
    def test_save_pdf(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.pdf")

            report = Report("Test Report")
            report.add_text("Introduction", "This is a test.")

            # Create minimal results for plotting
            exp = Experiment("test")
            exp.add_run("baseline", SimulationConfig(
                model_name="cnn", dataset_name="mnist",
                num_clients=3, num_rounds=2, local_epochs=1,
                learning_rate=0.01, strategies=["fedavg"],
                attack=AttackConfig(),
            ))
            results = exp.run()

            report.add_convergence_plot(results, ["baseline"], title="Test Plot")
            report.add_accuracy_table(results)
            report.add_heatmap([[0.95, 0.90], [0.85, 0.80]],
                                ["row1", "row2"], ["col1", "col2"],
                                title="Test Heatmap")
            report.save_pdf(path)

            assert os.path.exists(path)
            assert os.path.getsize(path) > 1000  # non-trivial PDF
