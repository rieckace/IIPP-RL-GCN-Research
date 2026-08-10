# 🕸️ Week 7 - Building and Predicting on a Custom Graph

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![PyTorch Geometric](https://img.shields.io/badge/PyTorch_Geometric-2.3+-green.svg)
![Internship](https://img.shields.io/badge/INEC%20Lab-Yuan%20Ze%20University-orange.svg)

This week, we bridge the gap between academic datasets (like Cora from Week 6) and real-world engineering problems. We will build our **own custom network graph** from scratch and train a GCN to make predictions on it!

This is a critical milestone because in our final IoT evacuation project, we won't be using a pre-made dataset—we will be converting building floorplans into custom graphs dynamically.

---

## 📖 The Scenario: IoT Edge Network

Imagine an IoT Edge Computing scenario (relevant to INEC Lab research):
- We have 10 devices (nodes).
- Some nodes are **Edge Servers** (powerful), others are **IoT Devices** (weak).
- They are connected in a topology where IoT devices communicate through central Edge Servers.

### The Prediction Task
We want to predict the **Status** of each node: is it `0 (Idle)` or `1 (Busy)`?
- We simulate some node features: `[CPU_Power, Current_Bandwidth]`.
- We assign fake ground-truth labels for training.
- We mask some nodes to simulate that we only know the status of *some* devices, and we want the GCN to predict the status of the *unknown* devices based on the network structure!

---

## 💻 Implementation: Custom Graph Construction

In `custom_graph_predictor.py`, we do this manually:

### 1. Constructing the Edges (Topology)
We define the `edge_index` to create a hub-and-spoke model where IoT devices connect to Edge Servers.
```python
# Node 0 and Node 5 are Edge Servers. Others are IoT devices.
edge_index = torch.tensor([
    [0, 1, 0, 2, 0, 3, 0, 4, 5, 6, 5, 7, 5, 8, 5, 9, 0, 5],
    [1, 0, 2, 0, 3, 0, 4, 0, 6, 5, 7, 5, 8, 5, 9, 5, 5, 0]
], dtype=torch.long)
```

### 2. Feature Engineering
```python
# [CPU_Capacity, Current_Bandwidth]
x = torch.tensor([
    [10.0, 100.0],  # Node 0 (Server - Strong)
    [1.0,  10.0],   # Node 1 (IoT)
    # ...
])
```

### 3. The Masking Technique
To train on a single graph (Transductive Learning), we use **masks**:
- `train_mask`: Tells the loss function which nodes it is allowed to look at the ground truth for.
- `test_mask`: Used only at the very end to evaluate how well the GCN guessed the hidden nodes.

---

## 📊 Results & Visualization

When you run the script, the GCN will train on the known nodes and attempt to classify the hidden nodes as `Idle` or `Busy`.
It will also generate `iot_network_graph.png`, showing the physical topology of our simulated edge network, color-coded by the GCN's predictions!

### How to Run
```bash
run_week.cmd
```

---

## 🔭 What's Next?
We have now fully mastered both standard Deep Q-Networks (DQN) and Graph Convolutional Networks (GCN). 
Over the next few weeks (Week 9-12), we will finally merge them together to build the **GCN-DQN**, where the GCN acts as the "eyes" of the DQN, allowing it to navigate dynamic building graphs!

```
Week 1-4  ✅  RL Basics & Deep Q-Networks
Week 5-6  ✅  PyTorch Geometric & GCN classification
Week 7-8  ✅  Use Your Own Custom Graph         ← You are here
Week 9-12 🔜  Combine DQN + GCN
```

---
## 👨‍💻 Author
**Rikesh Yadav**  
Research Intern — International Internship Pilot Program (IIPP)  
**Laboratory:** Intelligent Networks and Edge-Cloud Computing (INEC Lab)  
**Institution:** Yuan Ze University, Taoyuan, Taiwan  
