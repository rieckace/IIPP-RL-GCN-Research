"""
network.py

Deep Q-Network architecture for the evacuation environment.
Uses a Convolutional Neural Network (CNN) with Adaptive Max Pooling to allow
for dynamic grid sizes (Zero-Shot Generalization).
"""

import torch
import torch.nn as nn

class DQNetwork(nn.Module):
    """Feedforward Deep Q-Network for fixed 30x30 grids."""
    def __init__(self, in_channels: int = 8, action_size: int = 5):
        super().__init__()
        # 30x30 grid * 8 channels = 7200 inputs
        input_size = 30 * 30 * in_channels
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x arrives as (B, C, H, W). Flatten to (B, C*H*W)
        x = x.reshape(x.size(0), -1)
        return self.network(x)
