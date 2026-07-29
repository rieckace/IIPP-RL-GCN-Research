import os
import sys
import unittest
import torch
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from environment.marl_env import MARLEvacuationEnv
from environment.constants import Action
from models.marl.network import MARL_GNNDQNetwork

class TestMARL(unittest.TestCase):

    def setUp(self):
        self.config = {
            "grid": {"rows": 5, "cols": 5},
            "map": {
                "agent_start": [[0, 0], [0, 1]], # 2 Agents
                "exits": [[4, 4]],
                "walls": [],
                "fire_sources": []
            },
            "dynamics": {"max_steps": 10},
            "rewards": {
                "exit_reached": 100.0,
                "fire_hit": -50.0,
                "smoke_step": -10.0,
                "wall_bump": -5.0,
                "normal_step": -1.0,
                "stay_penalty": -2.0,
                "team_bonus": 10.0
            }
        }
        self.env = MARLEvacuationEnv(self.config)

    def test_marl_env_initialization(self):
        obs, info = self.env.reset()
        self.assertEqual(self.env.num_agents, 2)
        self.assertEqual(len(info["agent_positions"]), 2)
        self.assertEqual(len(info["active_agents"]), 2)
        self.assertTrue(all(info["active_agents"]))

    def test_marl_collision_avoidance(self):
        self.env.reset()
        # Both agents try to move to [0, 1]. Agent 1 is already there, Agent 0 wants to go RIGHT.
        # This should result in a collision where Agent 0 stays at [0, 0].
        # Wait, if Agent 1 moves DOWN to [1, 1], then [0, 1] is free.
        
        # Scenario: Agent 0 moves RIGHT to [0, 1]. Agent 1 moves LEFT to [0, 0]. 
        # They swap positions.
        actions = [int(Action.RIGHT), int(Action.LEFT)]
        obs, rewards, term, trunc, info = self.env.step(actions)
        
        # Because they targeted each other's cells, pos_counts will be 1 each, 
        # but they swap. In our simple collision logic, they swap successfully.
        self.assertEqual(info["agent_positions"][0], (0, 1))
        self.assertEqual(info["agent_positions"][1], (0, 0))
        
        # Now let's try a strict collision. 
        # Move them to [0, 0] and [0, 2]
        self.env.state.agent_positions = [(0, 0), (0, 2)]
        self.env.active_agents = [True, True]
        self.env.state.sync_to_grid(self.env.grid)
        
        # Both try to move to [0, 1]
        actions = [int(Action.RIGHT), int(Action.LEFT)]
        obs, rewards, term, trunc, info = self.env.step(actions)
        
        # Should bounce back to original
        self.assertEqual(info["agent_positions"][0], (0, 0))
        self.assertEqual(info["agent_positions"][1], (0, 2))

    def test_marl_network_shape(self):
        net = MARL_GNNDQNetwork(input_dim=8, hidden_dim=32, num_actions=5)
        
        # Create a dummy graph with 25 nodes (5x5 grid)
        x = torch.rand((25, 8))
        
        # Fully connected dummy edges
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        
        # 3 Agents standing on nodes 0, 1, and 24
        agent_node_indices = torch.LongTensor([0, 1, 24])
        
        q_vals = net(x, edge_index, agent_node_indices)
        
        # Should output [3 agents, 5 actions]
        self.assertEqual(q_vals.shape, (3, 5))

if __name__ == "__main__":
    unittest.main()
