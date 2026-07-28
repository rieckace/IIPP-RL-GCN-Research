"""
replay_buffer.py

Experience replay buffer optimized for PyTorch Geometric graphs.
"""

from collections import deque
import random
from typing import Dict, List, Tuple

import numpy as np
import torch

try:
    from torch_geometric.data import Data, Batch
except ImportError:
    pass  # Handled in network.py


class GraphReplayBuffer:
    """Circular experience replay buffer for graph observations.
    
    Instead of flat arrays, this stores PyG Data objects and 
    batches them using torch_geometric.data.Batch.from_data_list().
    """

    def __init__(self, capacity: int = 50000):
        """Initialize the buffer.
        
        Args:
            capacity: Maximum number of transitions to store.
        """
        self.buffer = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def is_ready(self, batch_size: int) -> bool:
        """Check if the buffer has enough samples for a batch."""
        return len(self.buffer) >= batch_size

    def push(
        self,
        obs: Dict[str, np.ndarray],
        action: int,
        reward: float,
        next_obs: Dict[str, np.ndarray],
        done: bool,
    ) -> None:
        """Store a transition in the buffer.
        
        Args:
            obs:      Graph observation dict (node_features, edge_index).
            action:   Action taken.
            reward:   Reward received.
            next_obs: Next graph observation dict.
            done:     Whether the episode ended.
        """
        # Convert raw dicts to PyG Data objects immediately to save overhead later
        data_obs = Data(
            x=torch.FloatTensor(obs["node_features"]),
            edge_index=torch.LongTensor(obs["edge_index"])
        )
        data_next_obs = Data(
            x=torch.FloatTensor(next_obs["node_features"]),
            edge_index=torch.LongTensor(next_obs["edge_index"])
        )

        self.buffer.append((data_obs, action, reward, data_next_obs, done))

    def sample(self, batch_size: int) -> Tuple[Any, torch.Tensor, torch.Tensor, Any, torch.Tensor]:
        """Sample a batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample.
            
        Returns:
            Tuple of:
            - batch_obs:       PyG Batch of states
            - batch_actions:   torch.Tensor of actions, shape (batch_size,)
            - batch_rewards:   torch.Tensor of rewards, shape (batch_size,)
            - batch_next_obs:  PyG Batch of next states
            - batch_dones:     torch.Tensor of booleans, shape (batch_size,)
        """
        transitions = random.sample(self.buffer, batch_size)
        
        obs_list, actions, rewards, next_obs_list, dones = zip(*transitions)

        # Batch the graph objects
        batch_obs = Batch.from_data_list(obs_list)
        batch_next_obs = Batch.from_data_list(next_obs_list)

        # Convert scalar values to tensors
        batch_actions = torch.LongTensor(actions)
        batch_rewards = torch.FloatTensor(rewards)
        batch_dones = torch.BoolTensor(dones)

        return batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones
