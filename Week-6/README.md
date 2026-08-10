# 🧠 Week 6 - Building Your First Graph Convolutional Network (GCN)

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![PyTorch Geometric](https://img.shields.io/badge/PyTorch_Geometric-2.3+-green.svg)
![Internship](https://img.shields.io/badge/INEC%20Lab-Yuan%20Ze%20University-orange.svg)

This week, we implement our very first **Graph Convolutional Network (GCN)** using PyTorch Geometric. Our goal is to train a GCN to classify nodes in a famous benchmark dataset (Cora).

---

## 📖 Theory: How does a GCN learn from graphs?

In a standard Neural Network (like the ones we used for DQN), data is treated as independent rows. But in a graph, nodes are connected. A GCN takes advantage of this by passing messages along the edges.

### 1. Message Passing
Every node looks at its direct neighbors and gathers their features. 
- If you want to guess what kind of academic paper a node represents, looking at the papers it cites (its neighbors) gives you a massive clue!

### 2. Aggregation
The node combines its own features with the features of its neighbors (usually by taking the sum or average).

### 3. Update (Neural Network Layer)
This combined information is passed through a neural network layer (with weights and an activation function like ReLU) to produce the node's new feature representation.

Mathematically, this is the GCN layer formula introduced by Kipf & Welling (2017):
`H^(l+1) = ReLU( D^(-1/2) A D^(-1/2) H^(l) W^(l) )`
- `H` = Node features
- `A` = Adjacency matrix (edges)
- `W` = Learnable weights

---

## 💻 Implementation: Node Classification

In `gcn_classifier.py`, we use the **Cora Dataset**, which is the "Hello World" of Graph Machine Learning.
- **Nodes (2708):** Scientific publications.
- **Edges (5429):** Citation links between publications.
- **Node Features (1433):** Bag-of-words indicating the presence of specific words in the publication.
- **Task:** Predict which of the 7 classes (topics) each publication belongs to.

### The Architecture
```python
class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        # 1st GCN layer (Feature reduction)
        self.conv1 = GCNConv(num_features, 16)
        # 2nd GCN layer (Classification)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # Message passing + ReLU activation
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        
        # Final message passing layer
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
```

---

## 📊 Results & Visualization

When you run the script, the GCN trains for 200 epochs using the Adam optimizer.
Because it uses the graph structure (who cites who) rather than just the text features alone, it achieves an impressive test accuracy (~81%).

The script also extracts the 16-dimensional hidden representations learned by the GCN and uses **t-SNE** to squash them into 2D space. It saves this visualization as `cora_tsne_embeddings.png`. You will visually see how the GCN brilliantly grouped similar papers together into distinct clusters!

### How to Run
```bash
run_week.cmd
```

---

## 🔭 What's Next?
Now that we know how to use GCNs on standard datasets, next week we will create our **own custom graph** simulating devices connected to servers, and run GCN predictions on it!

```
Week 1-4  ✅  RL Basics & Deep Q-Networks
Week 5    ✅  Learn About Graphs & PyTorch Geometric
Week 6    ✅  Build Your First GCN                  ← You are here
Week 7-8  🔜  Use Your Own Graph
Week 9-12 🔜  Combine DQN + GCN
```

---
## 👨‍💻 Author
**Rikesh Yadav**  
Research Intern — International Internship Pilot Program (IIPP)  
**Laboratory:** Intelligent Networks and Edge-Cloud Computing (INEC Lab)  
**Institution:** Yuan Ze University, Taoyuan, Taiwan  
