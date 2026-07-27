"""
replay_buffer.py

Experience replay memory for DQN training.
Stores (state, action, reward, next_state, done) transitions
and provides uniform random sampling for mini-batch learning.
"""

import random
from collections import deque
from typing import Tuple

import numpy as np


class ReplayBuffer:
    """Circular experience replay buffer with uniform sampling.

    Breaks temporal correlations in training data by storing transitions
    and sampling random mini-batches, which is critical for stable DQN learning.
    """

    def __init__(self, capacity: int = 50_000) -> None:
        """
        Args:
            capacity: Maximum number of transitions to store.
                      When full, oldest transitions are discarded.
        """
        self.buffer: deque = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store a transition in the buffer.

        Args:
            state:      Current observation.
            action:     Action taken (integer).
            reward:     Reward received.
            next_state: Next observation after the action.
            done:       Whether the episode ended (terminated or truncated).
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """Sample a random mini-batch of transitions.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
            as NumPy arrays ready for tensor conversion.
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.int32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.int32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        """Return the current number of stored transitions."""
        return len(self.buffer)

    def is_ready(self, batch_size: int) -> bool:
        """Check if enough transitions are stored for sampling."""
        return len(self.buffer) >= batch_size
