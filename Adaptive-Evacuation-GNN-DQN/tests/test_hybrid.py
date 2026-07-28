import os
import sys
import unittest
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from environment.grid import Grid
from environment.constants import CellType
from environment.heuristics import AStarPlanner
from environment.evacuation_env import EvacuationEnv
from environment.wrappers import HybridObservationWrapper
from utils.config_loader import load_config

class TestHybridHeuristics(unittest.TestCase):

    def test_astar_planner_direct_path(self):
        # Create a simple 5x5 grid
        grid = Grid(5, 5)
        # Agent at (0, 0), exit at (4, 4)
        start = (0, 0)
        exits = {(4, 4)}
        
        path = AStarPlanner.compute_path(grid, start, exits)
        
        self.assertTrue(len(path) > 0)
        self.assertEqual(path[0], start)
        self.assertEqual(path[-1], (4, 4))
        # Manhattan distance from (0,0) to (4,4) is 8, so path length should be 9
        self.assertEqual(len(path), 9)

    def test_astar_planner_with_walls(self):
        grid = Grid(5, 5)
        # Block the direct path
        grid.set_cell(0, 1, CellType.WALL)
        grid.set_cell(1, 1, CellType.WALL)
        grid.set_cell(2, 1, CellType.WALL)
        
        start = (0, 0)
        exits = {(0, 2)}
        
        path = AStarPlanner.compute_path(grid, start, exits)
        
        # Path must go around the wall: (0,0)->(1,0)->(2,0)->(3,0)->(3,1)->(3,2)->(2,2)->(1,2)->(0,2)
        # There are multiple valid paths of the same length, just check it exists and reaches exit
        self.assertTrue(len(path) > 0)
        self.assertEqual(path[-1], (0, 2))
        
        # Ensure it didn't pass through a wall
        for r, c in path:
            self.assertNotEqual(grid.get_cell(r, c), CellType.WALL)

    def test_hybrid_wrapper_9_dim(self):
        config_path = os.path.join(PROJECT_ROOT, "configs", "default.yaml")
        env_config = load_config(config_path)
        
        base_env = EvacuationEnv(env_config)
        env = HybridObservationWrapper(base_env)
        
        obs, _ = env.reset()
        
        self.assertIn("node_features", obs)
        self.assertIn("edge_index", obs)
        
        # Check that feature dim is 9
        # Assuming 10x10 grid -> 100 nodes
        self.assertEqual(obs["node_features"].shape, (100, 9))
        
        # The 9th column is the A* path flag. At least some nodes should be 1.0 (the path)
        path_flags = obs["node_features"][:, 8]
        self.assertTrue(np.sum(path_flags) > 0)

if __name__ == "__main__":
    unittest.main()
