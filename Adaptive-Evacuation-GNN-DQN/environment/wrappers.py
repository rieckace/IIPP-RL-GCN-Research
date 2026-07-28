"""
wrappers.py

Gymnasium wrappers for the evacuation environment.
"""

from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np


class GraphObservationWrapper(gym.ObservationWrapper):
    """Converts the flat observation into a graph dict observation.
    
    The environment inherently supports graph observations via 
    env.get_graph_observation(), but wrapping it makes it fully
    compatible with standard RL loops that expect step() to
    return the intended format directly.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        
        # We define the observation space as a Dict space, though
        # PyTorch Geometric will consume the raw numpy arrays anyway.
        # This is mostly for compliance with the gym API.
        
        # Determine sizes from the unwrapped env
        rows = self.unwrapped.rows
        cols = self.unwrapped.cols
        num_nodes = rows * cols
        num_cell_types = 8  # 0 to 7
        
        self.observation_space = gym.spaces.Dict({
            "node_features": gym.spaces.Box(
                low=0.0, high=1.0, 
                shape=(num_nodes, num_cell_types), 
                dtype=np.float32
            ),
            "edge_index": gym.spaces.Box(
                low=0, high=num_nodes - 1,
                # Max edges = 4 * num_nodes (4-connected grid)
                shape=(2, num_nodes * 4), 
                dtype=np.int64
            )
        })

    def observation(self, obs: Any) -> Dict[str, np.ndarray]:
        """Convert the underlying flat observation to a graph observation."""
        return self.unwrapped.get_graph_observation()
