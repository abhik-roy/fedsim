import torch.nn as nn
import torchvision.models as models


class DenseNet121(nn.Module):
    """DenseNet-121 adapted for small images (28x28 or 32x32) and variable num_classes.

    DenseNet has ~7M parameters and dense connections that make it robust
    to vanishing gradients — a realistic model for FL experimentation.
    """

    def __init__(self, input_channels: int = 3, num_classes: int = 10, image_size: int = 32):
        super().__init__()
        self.densenet = models.densenet121(weights=None, num_classes=num_classes)

        # Adapt first conv for small images: smaller kernel, no aggressive downsampling
        self.densenet.features.conv0 = nn.Conv2d(
            input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        # Remove the max pool that's too aggressive for 28x28/32x32 inputs
        self.densenet.features.pool0 = nn.Identity()

    def forward(self, x):
        return self.densenet(x)
