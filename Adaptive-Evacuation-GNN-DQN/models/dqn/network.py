"""
network.py

Deep Q-Network architecture for the evacuation environment.
Uses a Convolutional Neural Network (CNN) with Adaptive Max Pooling to allow
for dynamic grid sizes (Zero-Shot Generalization).
"""

import torch
import torch.nn as nn

class DQNetwork(nn.Module):
    """Convolutional Deep Q-Network.

    Architecture:
        Grid Input (B, 8, H, W)
        → Conv2d(32) → ReLU
        → Conv2d(64) → ReLU
        → Conv2d(64) → ReLU
        → AdaptiveMaxPool2d((2, 2))  [This enables scale-invariance!]
        → Flatten
        → Linear(256) → ReLU
        → Linear(Action Size)

    Adaptive pooling ensures that regardless of whether the input grid is
    10x10 (Office) or 26x26 (Mall), the linear layers always receive a 
    fixed-size flattened feature vector (64 * 2 * 2 = 256).
    """

    def __init__(
        self,
        in_channels: int = 8,
        action_size: int = 5,
    ) -> None:
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # Adaptive pooling scales any HxW feature map down to a fixed 2x2 grid
        self.pool = nn.AdaptiveMaxPool2d((2, 2))
        
        self.fc = nn.Sequential(
            nn.Linear(64 * 2 * 2, 256),
            nn.ReLU(),
            nn.Linear(256, action_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, in_channels, H, W).

        Returns:
            Q-values tensor of shape (batch_size, action_size).
        """
        x = self.conv(x)
        x = self.pool(x)
        x = x.reshape(x.size(0), -1) # Flatten (B, 64, 2, 2) -> (B, 256)
        return self.fc(x)
