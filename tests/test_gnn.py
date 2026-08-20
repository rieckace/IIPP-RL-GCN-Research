import os
import sys
import unittest

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from environment.evacuation_env import EvacuationEnv
from environment.wrappers import GraphObservationWrapper
from utils.config_loader import load_config

try:
    from torch_geometric.data import Data, Batch
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

if HAS_PYG:
    from models.gnn.network import GNNDQNetwork
    from models.gnn.replay_buffer import GraphReplayBuffer
    from models.gnn.trainer import GNNDQNAgent

@unittest.skipIf(not HAS_PYG, "torch_geometric is required for GNN tests")
class TestGNN(unittest.TestCase):
    
    def setUp(self):
        config_path = os.path.join(PROJECT_ROOT, "configs", "default.yaml")
        self.env_config = load_config(config_path)
        self.gnn_config = load_config(os.path.join(PROJECT_ROOT, "configs", "gnn.yaml"), validate=False)

    def test_graph_wrapper(self):
        base_env = EvacuationEnv(self.env_config)
        env = GraphObservationWrapper(base_env)
        
        obs, _ = env.reset()
        
        self.assertIn("node_features", obs)
        self.assertIn("edge_index", obs)
        
        # For 10x10 grid, 100 nodes.
        # Node features should be (100, 9)
        self.assertEqual(obs["node_features"].shape, (100, 9))
        
        # Max edges for 4-connected 10x10 is roughly 400. Shape must be (2, E)
        self.assertEqual(obs["edge_index"].shape[0], 2)

    def test_gnn_network_forward(self):
        net = GNNDQNetwork(node_feature_dim=9, action_size=5, gcn_hidden_dims=[16], mlp_hidden_dims=[16])
        
        # Create dummy batch
        x1 = torch.rand(100, 9)
        edge_index1 = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        data1 = Data(x=x1, edge_index=edge_index1)
        
        x2 = torch.rand(100, 9)
        data2 = Data(x=x2, edge_index=edge_index1)
        
        batch = Batch.from_data_list([data1, data2])
        
        out = net(batch)
        # 2 graphs in batch, 5 actions
        self.assertEqual(out.shape, (2, 5))

    def test_graph_replay_buffer(self):
        buffer = GraphReplayBuffer(capacity=10)
        
        obs = {
            "node_features": np.zeros((10, 9), dtype=np.float32),
            "edge_index": np.zeros((2, 5), dtype=np.int64)
        }
        next_obs = obs.copy()
        
        buffer.push(obs, 0, 1.0, next_obs, False)
        buffer.push(obs, 1, -1.0, next_obs, True)
        
        self.assertEqual(len(buffer), 2)
        
        batch_obs, acts, rews, batch_next, dones = buffer.sample(2)
        
        # B = 2 -> total nodes = 20
        self.assertEqual(batch_obs.x.shape, (20, 9))
        self.assertEqual(acts.shape, (2,))
        self.assertEqual(rews.shape, (2,))
        self.assertEqual(dones.shape, (2,))

if __name__ == "__main__":
    unittest.main()
