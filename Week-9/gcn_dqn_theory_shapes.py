import torch

print("=====================================================")
print("Week 9: Conceptual Tensor Flow of GCN to DQN")
print("=====================================================")

# 1. Simulate the Graph Input
num_nodes = 25 # e.g., a 5x5 Grid
num_features = 4 # [is_wall, is_fire, is_exit, is_agent]
print(f"\n[1] Graph Input:")
print(f"    Nodes: {num_nodes}")
print(f"    Features per Node: {num_features}")

# Create dummy node feature tensor
x = torch.rand((num_nodes, num_features))
print(f"    Input Tensor Shape (x): {x.shape}")

# 2. Simulate GCN Spatial Processing
gcn_hidden_size = 64
print(f"\n[2] GCN Layers (Message Passing):")
print(f"    Applying GCNConv(in={num_features}, out={gcn_hidden_size})...")

# Dummy GCN output (Normally requires edge_index, but simulating shape here)
gcn_out = torch.rand((num_nodes, gcn_hidden_size))
print(f"    GCN Output Tensor Shape: {gcn_out.shape}")
print("    Notice: We still have 25 nodes, but now each has 64 complex features!")

# 3. Simulate Agent-Centric Pooling
print(f"\n[3] Agent-Centric Pooling:")
print("    Extracting ONLY the embedding for the node where the agent is standing.")

# Simulating extracting 1 specific row (the agent's location)
agent_node_index = 12 # Let's say the agent is standing on node 12
agent_embedding = gcn_out[agent_node_index].unsqueeze(0) # Shape becomes (1, 64)
print(f"    Extracted Agent Embedding Shape: {agent_embedding.shape}")

# 4. Simulate DQN Decision Making
num_actions = 4 # Up, Down, Left, Right
print(f"\n[4] DQN Action Head:")
print(f"    Passing the (1, 64) embedding through Linear(64, {num_actions})")

# Dummy DQN output
dqn_out = torch.rand((1, num_actions))
print(f"    DQN Output Tensor (Q-Values) Shape: {dqn_out.shape}")
print(f"    Example Q-Values: {dqn_out.detach().numpy()}")

print("\nConclusion: The GCN handled the irregular graph size, and compressed it into")
print("a fixed-size vector (1, 64) so the DQN could choose an action without crashing!")
print("=====================================================")
