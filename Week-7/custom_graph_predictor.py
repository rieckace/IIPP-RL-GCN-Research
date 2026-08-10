import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx

print("Constructing Custom IoT Edge Network Graph...")

# 1. Custom Node Features [CPU_Capacity, Current_Bandwidth]
# 10 Nodes total. 
# Nodes 0 and 5 are powerful Edge Servers.
# Nodes 1,2,3,4 connect to Server 0.
# Nodes 6,7,8,9 connect to Server 5.
x = torch.tensor([
    [10.0, 100.0], # Node 0: Server
    [1.0,  5.0],   # Node 1: IoT
    [1.5,  2.0],   # Node 2: IoT
    [2.0,  15.0],  # Node 3: IoT
    [1.0,  4.0],   # Node 4: IoT
    [12.0, 120.0], # Node 5: Server
    [0.5,  1.0],   # Node 6: IoT
    [2.0,  10.0],  # Node 7: IoT
    [1.0,  8.0],   # Node 8: IoT
    [1.5,  6.0],   # Node 9: IoT
], dtype=torch.float)

# 2. Custom Edges (Bidirectional Hub-and-Spoke + Server Link)
edge_index = torch.tensor([
    # Source
    [0, 1, 0, 2, 0, 3, 0, 4,  5, 6, 5, 7, 5, 8, 5, 9,  0, 5],
    # Target
    [1, 0, 2, 0, 3, 0, 4, 0,  6, 5, 7, 5, 8, 5, 9, 5,  5, 0]
], dtype=torch.long)

# 3. Custom Labels (0: Idle, 1: Busy)
# We invent some ground truth for our network.
y = torch.tensor([1, 0, 0, 1, 0, 0, 0, 1, 1, 0], dtype=torch.long)

# 4. Training and Testing Masks
# We only "know" the status of 60% of the nodes. We want the GCN to predict the rest!
train_mask = torch.tensor([True, True, False, True, False, True, True, False, True, False])
test_mask = ~train_mask # The exact opposite

# Build PyG Data Object
data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, test_mask=test_mask)

# Define GCN Model
class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(2, 4) # 2 input features -> 4 hidden
        self.conv2 = GCNConv(4, 2) # 4 hidden -> 2 output classes (Idle/Busy)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

model = GCN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.02, weight_decay=5e-4)

print("\nTraining GCN on Custom Graph...")
model.train()
for epoch in range(100):
    optimizer.zero_grad()
    out = model(data)
    # ONLY calculate loss on nodes in the train_mask!
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 20 == 0:
        print(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")

# Evaluation
model.eval()
pred = model(data).argmax(dim=1)
correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
acc = int(correct) / int(data.test_mask.sum())

print(f"\n✅ Custom Graph Test Accuracy: {acc*100:.1f}%")
print("Predictions for all nodes:", pred.tolist())
print("Actual truth for all nodes:", data.y.tolist())

# Visualization
print("\nGenerating Graph Visualization...")
nx_graph = to_networkx(data, to_undirected=True)

color_map = []
for node in range(data.num_nodes):
    if pred[node].item() == 1:
        color_map.append('red') # Busy
    else:
        color_map.append('lightgreen') # Idle

plt.figure(figsize=(8, 6))
pos = nx.spring_layout(nx_graph)
nx.draw(nx_graph, pos, with_labels=True, node_color=color_map, 
        node_size=1200, font_size=12, font_weight='bold', edge_color='gray')

plt.title("IoT Edge Network Status Prediction (Red=Busy, Green=Idle)")
plt.savefig("iot_network_graph.png")
print("Visualization saved to 'iot_network_graph.png'")
