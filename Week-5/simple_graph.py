import torch
from torch_geometric.data import Data
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx

print("Building a simple graph with PyTorch Geometric...")

# 1. Define Node Features (x)
# Let's say we have 4 nodes, and each node has 2 features.
# For example: [is_computer, is_router]
node_features = torch.tensor([
    [1, 0], # Node 0: Computer
    [0, 1], # Node 1: Router
    [1, 0], # Node 2: Computer
    [1, 0]  # Node 3: Computer
], dtype=torch.float)

# 2. Define Edges (edge_index)
# We use COO (Coordinate Format) to define edges.
# The first row contains the source nodes, the second row contains the target nodes.
# Let's connect: 0 <-> 1, 1 <-> 2, 1 <-> 3
edge_index = torch.tensor([
    [0, 1, 1, 2, 1, 3], # Source nodes
    [1, 0, 2, 1, 3, 1]  # Target nodes (bidirectional edges)
], dtype=torch.long)

# 3. Create the Data Object
graph_data = Data(x=node_features, edge_index=edge_index)

# 4. Explore the Graph Properties
print("\n--- Graph Information ---")
print(f"Number of nodes: {graph_data.num_nodes}")
print(f"Number of edges: {graph_data.num_edges}")
print(f"Number of node features: {graph_data.num_node_features}")
print(f"Contains isolated nodes? {graph_data.has_isolated_nodes()}")
print(f"Contains self-loops? {graph_data.has_self_loops()}")
print(f"Is undirected? {graph_data.is_undirected()}")

# 5. Visualize the Graph using NetworkX
print("\nConverting to NetworkX for visualization...")
nx_graph = to_networkx(graph_data, to_undirected=True)

plt.figure(figsize=(6, 6))
pos = nx.spring_layout(nx_graph)
nx.draw(nx_graph, pos, with_labels=True, node_color='lightblue', 
        node_size=1500, font_size=16, font_weight='bold', edge_color='gray')

plt.title("My First PyG Graph")
plt.savefig("my_first_graph.png")
print("Visualization saved as 'my_first_graph.png'")
