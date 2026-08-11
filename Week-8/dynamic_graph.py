import torch
from torch_geometric.data import Data
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx

print("Converting 2D Grid to Graph...")

# Define a simple 3x3 layout
# 0 = Empty Corridor, 1 = Wall
grid = [
    [0, 0, 1],
    [1, 0, 0],
    [0, 0, 1]
]

rows = len(grid)
cols = len(grid[0])

# Lists to hold our graph components
node_features = []
source_nodes = []
target_nodes = []

# Mapping (row, col) to a unique node ID
# Since walls won't be nodes, we need a dynamic mapping
pos_to_id = {}
current_id = 0

# 1. Create Nodes
for r in range(rows):
    for c in range(cols):
        if grid[r][c] == 0: # Not a wall
            pos_to_id[(r, c)] = current_id
            current_id += 1
            # Feature: [r, c] (Just storing coordinates as basic features for now)
            node_features.append([float(r), float(c)])

# 2. Create Edges
directions = [(-1, 0), (1, 0), (0, -1), (0, 1)] # Up, Down, Left, Right

for r in range(rows):
    for c in range(cols):
        if grid[r][c] == 0:
            u = pos_to_id[(r, c)]
            
            # Check neighbors
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                
                # If neighbor is within bounds and is an empty corridor
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 0:
                    v = pos_to_id[(nr, nc)]
                    source_nodes.append(u)
                    target_nodes.append(v)

# 3. Construct PyG Data
x = torch.tensor(node_features, dtype=torch.float)
edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)

data = Data(x=x, edge_index=edge_index)

print(f"\nSuccessfully converted 3x3 grid into graph with {data.num_nodes} nodes and {data.num_edges} edges!")
print("Node Features (Coordinates):\n", data.x)
print("Edge Index:\n", data.edge_index)

# 4. Visualization
nx_graph = to_networkx(data, to_undirected=True)
plt.figure(figsize=(5, 5))

# We can use the actual coordinates from our feature matrix to draw the graph exactly like the grid!
pos = {i: (x[i][1].item(), -x[i][0].item()) for i in range(data.num_nodes)}

nx.draw(nx_graph, pos, with_labels=True, node_color='lightblue', 
        node_size=800, font_size=10, font_weight='bold', edge_color='black')
plt.title("Spatial Grid Converted to Graph")
plt.savefig("spatial_graph.png")
print("Graph topology saved as 'spatial_graph.png'")
