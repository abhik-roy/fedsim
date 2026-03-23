import torch.nn as nn


class SimpleCNN(nn.Module):
    """CNN with batch normalization and dropout for improved accuracy.

    Architecture:
      Conv(32) → BN → ReLU → Conv(64) → BN → ReLU → Pool
      Conv(128) → BN → ReLU → Conv(128) → BN → ReLU → Pool
      FC(256) → Dropout → FC(num_classes)

    Suitable for 28x28 (MNIST/FEMNIST) and 32x32 (CIFAR) inputs.
    ~200K-500K params depending on num_classes.
    """

    def __init__(self, input_channels: int = 1, num_classes: int = 10, image_size: int = 28):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
        )
        reduced_size = image_size // 4
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * reduced_size * reduced_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
