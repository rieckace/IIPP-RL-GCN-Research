import sys
import os
from pathlib import Path
import unittest
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from environment.grid import Grid
from environment.state import EnvironmentState
from environment.constants import CellType
from environment.make_env import make_env

class TestGraphTopology(unittest.TestCase):
    def test_normal_connectivity_without_walls(self):
        grid = Grid(2, 2)
        state = EnvironmentState()
        
        graph = state.to_graph(grid)
        edge_index = graph["edge_index"]
        
        # 2x2 grid has 4 nodes, 8 directed edges (4 undirected)
        self.assertEqual(edge_index.shape[1], 8)
        
        edges = set(zip(edge_index[0], edge_index[1]))
        expected_edges = {
            (0, 1), (1, 0),
            (0, 2), (2, 0),
            (1, 3), (3, 1),
            (2, 3), (3, 2)
        }
        self.assertEqual(edges, expected_edges)
        
    def test_wall_separation_no_cross_wall_path(self):
        # Grid: AGENT(0) | WALL(1) | EXIT(2)
        grid = Grid(1, 3)
        grid.set_cell(0, 0, CellType.AGENT)
        grid.set_cell(0, 1, CellType.WALL)
        grid.set_cell(0, 2, CellType.EXIT)
        
        state = EnvironmentState()
        graph = state.to_graph(grid)
        edge_index = graph["edge_index"]
        
        # There should be NO edges because 0-1 and 1-2 are blocked by the wall at 1
        self.assertEqual(edge_index.shape[1], 0)
        
    def test_agent_and_exit_node_identification(self):
        grid = Grid(1, 3)
        grid.set_cell(0, 0, CellType.AGENT)
        grid.set_cell(0, 1, CellType.WALL)
        grid.set_cell(0, 2, CellType.EXIT)
        
        state = EnvironmentState()
        graph = state.to_graph(grid)
        node_features = graph["node_features"]
        
        self.assertEqual(node_features[0, CellType.AGENT], 1.0)
        self.assertEqual(node_features[1, CellType.WALL], 1.0)
        self.assertEqual(node_features[2, CellType.EXIT], 1.0)
        
    def test_boundary_handling(self):
        grid = Grid(1, 1) # Just 1 cell
        state = EnvironmentState()
        graph = state.to_graph(grid)
        
        self.assertEqual(graph["edge_index"].shape[1], 0)
        
    def test_wall_separation_larger_grid(self):
        # 3x5 Grid
        # S S W S S
        # S S W S S
        # S S W S S
        grid = Grid(3, 5)
        for r in range(3):
            grid.set_cell(r, 2, CellType.WALL)
            
        state = EnvironmentState()
        graph = state.to_graph(grid)
        edge_index = graph["edge_index"]
        
        edges = set(zip(edge_index[0], edge_index[1]))
        
        for u, v in edges:
            u_r, u_c = u // 5, u % 5
            v_r, v_c = v // 5, v % 5
            self.assertNotEqual(u_c, 2)
            self.assertNotEqual(v_c, 2)
            
            if u_c < 2:
                self.assertTrue(v_c < 2)
            if u_c > 2:
                self.assertTrue(v_c > 2)
                
    def test_all_benchmark_map_sizes(self):
        maps = {
            "office": (10, 10),
            "apartment": (14, 14),
            "school": (18, 18),
            "hospital": (22, 22),
            "mall": (26, 26)
        }
        
        state = EnvironmentState()
        for map_name, (rows, cols) in maps.items():
            env = make_env(map_name)
            graph = state.to_graph(env.grid)
            
            self.assertEqual(graph["node_features"].shape[0], rows * cols)
            self.assertEqual(graph["node_features"].shape[1], 9)
            
            edge_index = graph["edge_index"]
            edges = set(zip(edge_index[0], edge_index[1]))
            for u, v in edges:
                u_r, u_c = u // cols, u % cols
                v_r, v_c = v // cols, v % cols
                
                self.assertNotEqual(env.grid.get_cell(u_r, u_c), CellType.WALL)
                self.assertNotEqual(env.grid.get_cell(v_r, v_c), CellType.WALL)

if __name__ == "__main__":
    unittest.main()
