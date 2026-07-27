"""
network.py

Deep Q-Network architecture for the evacuation environment.
Takes one-hot encoded grid observations and outputs Q-values for each action.
"""

from typing import List

import torch
import torch.nn as nn


class DQNetwork(nn.Module):
    """Feedforward Deep Q-Network.

    Architecture:
        one-hot grid input (800) → Dense(256) → ReLU
                                 → Dense(256) → ReLU
                                 → Dense(128) → ReLU
                                 → Dense(5)   → Q-values

    The input is a one-hot encoded grid: each of the 100 cells is
    represented as an 8-dimensional one-hot vector (one per CellType),
    giving 100 × 8 = 800 input features.
    """

    def __init__(
        self,
        input_size: int = 800,
        action_size: int = 5,
        hidden_layers: List[int] | None = None,
    ) -> None:
        """
        Args:
            input_size:    Dimension of the one-hot encoded input.
            action_size:   Number of discrete actions.
            hidden_layers: List of hidden layer sizes. Default: [256, 256, 128].
        """
        super().__init__()

        if hidden_layers is None:
            hidden_layers = [256, 256, 128]

        layers: list[nn.Module] = []
        prev_size = input_size

        for h_size in hidden_layers:
            layers.append(nn.Linear(prev_size, h_size))
            layers.append(nn.ReLU())
            prev_size = h_size

        layers.append(nn.Linear(prev_size, action_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_size).

        Returns:
            Q-values tensor of shape (batch_size, action_size).
        """
        return self.network(x)
