import torch
import torch.nn as nn

class ResNet(nn.Module):
    def __init__(self, num_channels=1, num_features=64, num_residual_blocks=5):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, num_features, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.res_blocks = nn.Sequential(*[
            nn.Sequential(
                nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
            ) for _ in range(num_residual_blocks)
        ])
        self.conv2 = nn.Conv2d(num_features, num_channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.res_blocks(out) + out
        out = self.conv2(out) + residual
        return out