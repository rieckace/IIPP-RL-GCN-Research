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
        num_cell_types = 9  # 8 one-hot + 1 visit count
        
        self.observation_space = gym.spaces.Dict({
            "node_features": gym.spaces.Box(
                low=0.0, high=100.0, 
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


class HybridObservationWrapper(gym.ObservationWrapper):
    """Converts the flat observation into a 9-dim hybrid graph observation.
    
    At each step, it runs the A* planner to find the shortest path to the 
    exit (ignoring fire). It then appends this path as a binary flag to 
    the node features.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        
        # Import here to avoid circular dependencies if any
        from environment.heuristics import AStarPlanner
        self.planner = AStarPlanner
        
        rows = self.unwrapped.rows
        cols = self.unwrapped.cols
        num_nodes = rows * cols
        num_cell_types = 9  # 8 CellTypes + 1 A* Path Flag
        
        self.observation_space = gym.spaces.Dict({
            "node_features": gym.spaces.Box(
                low=0.0, high=1.0, 
                shape=(num_nodes, num_cell_types), 
                dtype=np.float32
            ),
            "edge_index": gym.spaces.Box(
                low=0, high=num_nodes - 1,
                shape=(2, num_nodes * 4), 
                dtype=np.int64
            )
        })

    def observation(self, obs: Any) -> Dict[str, np.ndarray]:
        env_state = self.unwrapped.state
        grid = self.unwrapped.grid
        
        # If agent is dead/removed, just use empty path
        if not env_state.agent_positions:
            optimal_path = []
        else:
            start = env_state.agent_positions[0]
            exits = env_state.exit_cells
            optimal_path = self.planner.compute_path(grid, start, exits)
            
        return env_state.to_hybrid_graph(grid, optimal_path)


class MARLGraphObservationWrapper(gym.ObservationWrapper):
    """Converts the flat observation into a graph dict for MARL agents.
    
    Identical to GraphObservationWrapper, but works safely with MARLEvacuationEnv.
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
        
        rows = getattr(self.unwrapped, 'rows', 10)
        cols = getattr(self.unwrapped, 'cols', 10)
        num_nodes = rows * cols
        num_cell_types = 8
        
        self.observation_space = gym.spaces.Dict({
            "node_features": gym.spaces.Box(
                low=0.0, high=1.0, 
                shape=(num_nodes, num_cell_types), 
                dtype=np.float32
            ),
            "edge_index": gym.spaces.Box(
                low=0, high=num_nodes - 1,
                shape=(2, num_nodes * 4), 
                dtype=np.int64
            )
        })

    def observation(self, obs: Any) -> Dict[str, np.ndarray]:
        # If the unwrapped env doesn't have state/grid, fallback
        if not hasattr(self.unwrapped, 'state'):
            return obs
        return self.unwrapped.state.to_graph(self.unwrapped.grid)
