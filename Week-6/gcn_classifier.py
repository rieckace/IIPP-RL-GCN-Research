import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

print("Loading Cora Dataset...")
# Download and load the Cora dataset
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

print(f'Dataset: {dataset}:')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Number of features: {dataset.num_node_features}')
print(f'Number of classes: {dataset.num_classes}')

# Define the GCN architecture
class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        torch.manual_seed(12345)
        # First GCN layer transforms features to hidden channels
        self.conv1 = GCNConv(dataset.num_node_features, hidden_channels)
        # Second GCN layer transforms hidden channels to class scores
        self.conv2 = GCNConv(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index):
        # 1st layer: Message passing + ReLU + Dropout
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        
        # 2nd layer: Message passing
        x = self.conv2(x, edge_index)
        return x

model = GCN(hidden_channels=16)
print("\nModel Architecture:")
print(model)

# Training setup
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

def train():
    model.train()
    optimizer.zero_grad()
    # Forward pass
    out = model(data.x, data.edge_index)
    # Compute loss only on training nodes
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    # Backward pass
    loss.backward()
    optimizer.step()
    return loss.item()

def test():
    model.eval()
    out = model(data.x, data.edge_index)
    # Use the class with the highest probability
    pred = out.argmax(dim=1)
    # Check against ground-truth labels for test nodes
    test_correct = pred[data.test_mask] == data.y[data.test_mask]
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum())
    return test_acc

print("\nTraining GCN...")
for epoch in range(1, 201):
    loss = train()
    if epoch % 20 == 0:
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

test_acc = test()
print(f'\nFinal Test Accuracy: {test_acc:.4f}')

# Visualization of embeddings using t-SNE
print("\nExtracting embeddings and generating visualization...")
model.eval()
out = model.conv1(data.x, data.edge_index) # Get output of hidden layer
z = out.detach().cpu().numpy()

tsne = TSNE(n_components=2, random_state=42)
z_tsne = tsne.fit_transform(z)

plt.figure(figsize=(10, 10))
classes = data.y.numpy()
scatter = plt.scatter(z_tsne[:, 0], z_tsne[:, 1], c=classes, cmap="Set2", s=50, alpha=0.8)
plt.legend(handles=scatter.legend_elements()[0], labels=[f"Class {i}" for i in range(dataset.num_classes)])
plt.title("GCN Hidden Layer Embeddings (t-SNE)")
plt.axis('off')
plt.savefig("cora_tsne_embeddings.png")
print("Visualization saved to 'cora_tsne_embeddings.png'")
