import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_size: int = 784, num_classes: int = 10, hidden_size: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        return self.network(x)
