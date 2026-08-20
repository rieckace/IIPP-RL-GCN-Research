"""
target_network.py

Utilities for managing the target network in DQN.
The target network provides stable TD targets by being updated
less frequently than the online Q-network.
"""

import torch.nn as nn


def hard_update(target_net: nn.Module, source_net: nn.Module) -> None:
    """Copy all weights from source network to target network.

    Used for periodic full-weight synchronisation (standard DQN).

    Args:
        target_net: Network to receive the weights.
        source_net: Network to copy weights from.
    """
    target_net.load_state_dict(source_net.state_dict())


def soft_update(target_net: nn.Module, source_net: nn.Module, tau: float = 0.005) -> None:
    """Polyak-average weights from source into target network.

    θ_target = τ * θ_source + (1 - τ) * θ_target

    Provides smoother updates than hard copying. Useful for
    continuous-update variants and future extensions.

    Args:
        target_net: Network to be blended towards source.
        source_net: Network providing the new weight direction.
        tau:        Blending coefficient (0 = no update, 1 = hard copy).
    """
    for target_param, source_param in zip(
        target_net.parameters(), source_net.parameters()
    ):
        target_param.data.copy_(
            tau * source_param.data + (1.0 - tau) * target_param.data
        )
