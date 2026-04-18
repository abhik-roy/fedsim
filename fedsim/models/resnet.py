import torch.nn as nn
import torchvision.models as models


class ResNet18(nn.Module):
    """ResNet-18 adapted for small images (28x28 or 32x32) and variable num_classes."""

    def __init__(self, input_channels: int = 3, num_classes: int = 10, image_size: int = 32):
        super().__init__()
        self.resnet = models.resnet18(weights=None, num_classes=num_classes)

        # Adapt first conv for small images: smaller kernel, no aggressive downsampling
        self.resnet.conv1 = nn.Conv2d(
            input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        # Remove the max pool that halves spatial dims early (too aggressive for 28x28/32x32)
        self.resnet.maxpool = nn.Identity()

    def forward(self, x):
        return self.resnet(x)
