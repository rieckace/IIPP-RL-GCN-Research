import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GNNDQNetwork(nn.Module):
    """
    Hybrid GCN-DQN Architecture
    
    1. Processes graph using GCN layers.
    2. Extracts the specific node embedding where the agent is located.
    3. Passes that embedding through a standard DQN linear head to get Q-values.
    """
    def __init__(self, num_node_features, hidden_channels, num_actions):
        super(GNNDQNetwork, self).__init__()
        
        # --- Spatial Processing (GCN) ---
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        
        # --- Decision Making (DQN) ---
        self.fc1 = nn.Linear(hidden_channels, 128)
        self.fc2 = nn.Linear(128, num_actions)

    def forward(self, data):
        """
        data is a PyG Data object or Batch object containing:
        - data.x (Node features)
        - data.edge_index (Connectivity)
        - data.batch (Batch indices for multi-graph processing)
        """
        x, edge_index = data.x, data.edge_index

        # 1. GCN Layers: Aggregate neighborhood information
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        
        # 2. Agent-Centric Pooling
        # Assuming feature index 2 is the 'is_agent' boolean flag.
        # We find all nodes where the agent is currently located.
        agent_mask = data.x[:, 2] == 1.0
        
        # Extract the hidden embeddings purely for the agent's current nodes
        agent_embeddings = x[agent_mask]
        
        # Edge case safety: if for some reason the agent isn't found (should never happen),
        # fallback to global mean pooling.
        if agent_embeddings.size(0) == 0:
            from torch_geometric.nn import global_mean_pool
            agent_embeddings = global_mean_pool(x, data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=x.device))

        # 3. DQN Layers: Convert agent's spatial embedding into Action Q-Values
        q_vals = self.fc1(agent_embeddings)
        q_vals = F.relu(q_vals)
        q_vals = self.fc2(q_vals)

        return q_vals

if __name__ == "__main__":
    print("Testing GCN-DQN Architecture Initialization...")
    
    # 4 features, 64 hidden size, 4 actions (Up, Down, Left, Right)
    model = GNNDQNetwork(num_node_features=4, hidden_channels=64, num_actions=4)
    print(model)
    print("Architecture successfully compiled!")
