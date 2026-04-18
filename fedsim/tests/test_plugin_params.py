import pytest
from models import get_model
from data.loader import get_dataset


class TestGetModelKwargs:
    def test_builtin_model_works(self):
        model = get_model("cnn", "mnist")
        assert model is not None

    def test_custom_model_receives_kwargs(self):
        model = get_model("custom:TextCNN", "custom:AG News (4-class)", embed_dim=64)
        assert model is not None
        # Verify the kwarg was actually used
        assert model.embedding.embedding_dim == 64


class TestGetDatasetKwargs:
    def test_builtin_dataset_works(self):
        train, test = get_dataset("mnist")
        assert len(train) > 0
