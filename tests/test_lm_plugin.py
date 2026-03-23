import pytest
import torch


class TestWikiText2Plugin:
    def test_dataset_loads(self):
        from custom.datasets.wikitext2 import load
        train, test = load()
        assert len(train) > 0
        assert len(test) > 0
        x, y = train[0]
        assert x.dtype == torch.long
        assert y.dtype == torch.long
        assert len(x) == 128  # SEQ_LENGTH
        assert hasattr(train, "targets")

    def test_model_builds(self):
        from custom.models.language_model import build
        model = build({"vocab_size": 100, "seq_length": 32, "task_type": "language_modeling"})
        assert model is not None
        x = torch.randint(0, 100, (2, 32))
        logits = model(x)
        assert logits.shape == (2, 32, 100)

    def test_train_step(self):
        from custom.models.language_model import build, train_step
        model = build({"vocab_size": 100, "seq_length": 16, "task_type": "language_modeling"})
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        batch = (torch.randint(1, 100, (4, 16)), torch.randint(1, 100, (4, 16)))
        metrics = train_step(model, batch, optimizer, torch.device("cpu"))
        assert "loss" in metrics
        assert "perplexity" in metrics
        assert metrics["loss"] > 0
        assert metrics["perplexity"] > 0

    def test_eval_step(self):
        from custom.models.language_model import build, eval_step
        model = build({"vocab_size": 100, "seq_length": 16, "task_type": "language_modeling"})
        batch = (torch.randint(1, 100, (4, 16)), torch.randint(1, 100, (4, 16)))
        metrics = eval_step(model, batch, torch.device("cpu"))
        assert "loss" in metrics
        assert "perplexity" in metrics
