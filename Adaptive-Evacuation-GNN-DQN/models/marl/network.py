import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class MARL_GNNDQNetwork(nn.Module):
    """
    Multi-Agent Graph Neural Network for Q-Learning.
    
    Instead of globally pooling the graph into a single vector, this network
    extracts the specific node embedding for the exact coordinate where each
    agent is located. This allows multiple agents to share the same underlying
    graph state, but output unique Q-values based on their distinct spatial locations.
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_actions: int):
        super().__init__()
        
        # Message passing layers
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        
        # Shared MLP Head (applied to each agent's node embedding individually)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, agent_node_indices: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for Multi-Agent GNN.
        
        Args:
            x: Node features of shape (num_nodes, input_dim)
            edge_index: Graph connectivity of shape (2, num_edges)
            agent_node_indices: 1D Tensor of length (num_agents) containing the integer 
                                node IDs where each agent is currently standing.
                                
        Returns:
            q_values: Tensor of shape (num_agents, num_actions)
        """
        # Pass full graph through GCN layers
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.conv3(x, edge_index)
        x = torch.relu(x)
        
        # EXTRACT Node-Specific Embeddings for each agent
        # agent_embeddings shape: (num_agents, hidden_dim)
        agent_embeddings = x[agent_node_indices]
        
        # Compute Q-values independently for each agent using the shared MLP
        # q_values shape: (num_agents, num_actions)
        q_values = self.mlp(agent_embeddings)
        
        return q_values
