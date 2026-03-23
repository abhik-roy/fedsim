import pytest

class TestBuildDatasetInfo:
    def test_builtin_dataset_has_task_type(self):
        from simulation.runner import _build_dataset_info
        info = _build_dataset_info("cifar10")
        assert info["task_type"] == "image_classification"
        assert info["num_classes"] == 10
        assert info["input_channels"] == 3
        assert info["image_size"] == 32
        assert info.get("vocab_size") is None
        assert info.get("seq_length") is None

    def test_builtin_mnist(self):
        from simulation.runner import _build_dataset_info
        info = _build_dataset_info("mnist")
        assert info["task_type"] == "image_classification"
        assert info["vocab_size"] is None

    def test_custom_ag_news(self):
        from simulation.runner import _build_dataset_info
        info = _build_dataset_info("custom:AG News (4-class)")
        assert info["task_type"] == "text_classification"
        assert info["num_classes"] == 4
        assert info.get("vocab_size") is not None
        assert info.get("seq_length") == 256

    def test_unknown_raises(self):
        from simulation.runner import _build_dataset_info
        with pytest.raises((ValueError, KeyError)):
            _build_dataset_info("nonexistent")

    def test_custom_not_found_raises(self):
        from simulation.runner import _build_dataset_info
        with pytest.raises(ValueError, match="not found"):
            _build_dataset_info("custom:NoSuchPlugin")
