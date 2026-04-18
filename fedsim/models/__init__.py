import types

from models.cnn import SimpleCNN
from models.mlp import MLP
from models.resnet import ResNet18
from models.densenet import DenseNet121


def get_model(model_name: str, dataset_name: str, **kwargs):
    """Instantiate and return a model for the given model and dataset names.

    Args:
        model_name: Identifier for the model architecture. One of "cnn", "mlp",
            "resnet18", "densenet121", or a custom plugin name prefixed with
            "custom:" (e.g., "custom:MyResNet").
        dataset_name: Identifier for the dataset, used to look up input channels,
            image size, and number of classes. One of the built-in dataset keys or
            a custom plugin prefixed with "custom:".

    Returns:
        A torch.nn.Module instance configured for the specified dataset.

    Raises:
        ValueError: If model_name or dataset_name is unknown, or the corresponding
            plugin cannot be found.
    """
    from simulation.runner import _build_dataset_info
    dataset_info = _build_dataset_info(dataset_name)

    if model_name == "cnn":
        return SimpleCNN(
            input_channels=dataset_info["input_channels"],
            num_classes=dataset_info["num_classes"],
            image_size=dataset_info["image_size"],
        )
    elif model_name == "mlp":
        return MLP(
            input_size=dataset_info["input_size"],
            num_classes=dataset_info["num_classes"],
            hidden_size=kwargs.get("hidden_size", 256),
        )
    elif model_name == "resnet18":
        return ResNet18(
            input_channels=dataset_info["input_channels"],
            num_classes=dataset_info["num_classes"],
            image_size=dataset_info["image_size"],
        )
    elif model_name == "densenet121":
        return DenseNet121(
            input_channels=dataset_info["input_channels"],
            num_classes=dataset_info["num_classes"],
            image_size=dataset_info["image_size"],
        )
    elif model_name.startswith("custom:"):
        from plugins import discover_plugins
        plugin_name = model_name.replace("custom:", "")
        plugins = discover_plugins("models")
        for name, mod in plugins.items():
            if name == plugin_name and isinstance(mod, types.ModuleType):
                return mod.build(dataset_info, **kwargs)
        raise ValueError(f"Custom model plugin not found: {plugin_name}")
    else:
        raise ValueError(f"Unknown model: {model_name}")
