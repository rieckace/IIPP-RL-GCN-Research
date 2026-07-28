"""
trainer.py

Hybrid Agent that uses the GNN but expects 9-dimensional node features
(8 cell types + 1 A* path flag).
"""

from models.gnn.trainer import GNNDQNAgent
from models.gnn.network import GNNDQNetwork
from models.dqn.target_network import hard_update

class HybridGNNDQNAgent(GNNDQNAgent):
    """Extends the GNN Agent to support Hybrid A* features."""
    
    def __init__(self, config: dict):
        # Temporarily call super to init base structures
        super().__init__(config)
        
        # Override the hardcoded node feature dimension
        self.node_feature_dim = 9
        
        net_cfg = config.get("network", {})
        gcn_dims = net_cfg.get("gcn_hidden_dims", [64, 64, 64])
        mlp_dims = net_cfg.get("mlp_hidden_dims", [128, 64])
        
        # Re-initialize the networks with dim=9
        self.q_network = GNNDQNetwork(
            node_feature_dim=self.node_feature_dim,
            action_size=self.action_size,
            gcn_hidden_dims=gcn_dims,
            mlp_hidden_dims=mlp_dims,
        ).to(self.device)

        self.target_network = GNNDQNetwork(
            node_feature_dim=self.node_feature_dim,
            action_size=self.action_size,
            gcn_hidden_dims=gcn_dims,
            mlp_hidden_dims=mlp_dims,
        ).to(self.device)

        hard_update(self.target_network, self.q_network)
        self.target_network.eval()
        
        # Re-initialize optimizer for new network parameters
        import torch.optim as optim
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
