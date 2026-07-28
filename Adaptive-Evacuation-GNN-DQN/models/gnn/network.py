"""
network.py

Graph Neural Network (GNN) for the evacuation environment.
Uses PyTorch Geometric (PyG) for message passing.
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GCNConv, global_mean_pool
except ImportError:
    # Fallback to allow tests/typing to pass without PyG, but will crash if used
    class GCNConv(nn.Module):
        def __init__(self, *args, **kwargs): super().__init__()
    def global_mean_pool(*args, **kwargs): return None
    print("Warning: torch_geometric is not installed.")


class GNNDQNetwork(nn.Module):
    """Graph Neural Network for Deep Q-Learning.

    Architecture:
        1. GCN Layers (Message Passing): Propagate spatial information
           across the grid.
        2. Global Pooling: Aggregate all node embeddings into a single
           graph-level embedding vector.
        3. MLP Head: Project the graph embedding to Q-values for each action.
    """

    def __init__(
        self,
        node_feature_dim: int = 8,  # One-hot cell types
        action_size: int = 5,
        gcn_hidden_dims: List[int] = [64, 64, 64],
        mlp_hidden_dims: List[int] = [128, 64],
    ):
        """Initialize the GNNDQNetwork.

        Args:
            node_feature_dim: Number of features per node.
            action_size:      Number of possible actions (output Q-values).
            gcn_hidden_dims:  List of hidden dimensions for GCN layers.
            mlp_hidden_dims:  List of hidden dimensions for the MLP head.
        """
        super().__init__()

        # --- Message Passing (GCN) Layers ---
        self.convs = nn.ModuleList()
        in_dim = node_feature_dim
        for hidden_dim in gcn_hidden_dims:
            self.convs.append(GCNConv(in_dim, hidden_dim))
            in_dim = hidden_dim

        self.gcn_out_dim = in_dim

        # --- MLP Head ---
        layers = []
        mlp_in_dim = self.gcn_out_dim
        for hidden_dim in mlp_hidden_dims:
            layers.append(nn.Linear(mlp_in_dim, hidden_dim))
            layers.append(nn.ReLU())
            mlp_in_dim = hidden_dim

        layers.append(nn.Linear(mlp_in_dim, action_size))
        self.mlp_head = nn.Sequential(*layers)

    def forward(self, batch: Any) -> torch.Tensor:
        """Forward pass for a batch of graphs.

        Args:
            batch: PyTorch Geometric Batch object containing:
                   - x: Node features (total_nodes, node_feature_dim)
                   - edge_index: Graph connectivity (2, total_edges)
                   - batch: Batch vector assigning nodes to graphs (total_nodes,)

        Returns:
            Q-values: torch.Tensor of shape (batch_size, action_size)
        """
        x, edge_index, batch_idx = batch.x, batch.edge_index, batch.batch

        # 1. Message Passing
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)

        # 2. Global Pooling (Node Embeddings -> Graph Embedding)
        x = global_mean_pool(x, batch_idx)

        # 3. MLP Head (Graph Embedding -> Q-Values)
        q_values = self.mlp_head(x)

        return q_values
