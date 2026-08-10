# 🌐 Week 5 - Introduction to Graphs & PyTorch Geometric

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![PyTorch Geometric](https://img.shields.io/badge/PyTorch_Geometric-2.3+-green.svg)
![Internship](https://img.shields.io/badge/INEC%20Lab-Yuan%20Ze%20University-orange.svg)

This week, we officially transition from standard Reinforcement Learning into the world of **Graph Neural Networks (GNNs)**. Before we can combine DQN and GNN, we first need to understand how to represent data as graphs!

---

## 📖 Theory: What is a Graph?

A graph is a data structure used to model relationships between objects. It consists of two main components:
1. **Nodes (Vertices - $V$):** The entities or objects. In a social network, nodes are people. In our future evacuation project, nodes will be physical locations or rooms.
2. **Edges ($E$):** The connections between nodes. In a social network, an edge means two people are friends. In our project, an edge means two rooms are connected by a door or hallway.

Mathematically, a graph is represented as $G = (V, E)$.

### Why use Graphs instead of Grids (Images)?
Standard Neural Networks (like CNNs) expect fixed-size grids (like a 10x10 matrix of pixels). But real-world data like building layouts, internet networks, or molecular structures are irregular. Graphs can represent any irregular shape naturally!

---

## 🛠️ Tool: PyTorch Geometric (PyG)

**PyTorch Geometric** is an extension library for PyTorch specifically designed to build and train Graph Neural Networks.

### Installation Task
To get started, we need to install PyG. It requires specific versions matching your CUDA and PyTorch installation. 

```bash
pip install torch_geometric
# You may also need the optional dependencies depending on your setup:
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

---

## 💻 Implementation: Making a Simple Graph

In PyTorch Geometric, a graph is represented using the `Data` object. We need to define two things:
1. `x`: The node features (what information does each node hold?).
2. `edge_index`: The connections between nodes (represented in Coordinate Format / COO).

### Task: Build a Simple Graph
In `simple_graph.py`, we construct a very basic 4-node graph:
```text
Node 0 (Feature: [1, 0]) --- Node 1 (Feature: [0, 1])
   |                            |
Node 2 (Feature: [1, 1]) --- Node 3 (Feature: [0, 0])
```

The code demonstrates how to:
1. Create the node feature tensor `x`.
2. Create the connectivity tensor `edge_index`.
3. Construct the PyG `Data` object.
4. Visualize the graph using `networkx` and `matplotlib`.

### How to Run
```bash
run_week.cmd
```
This will output the graph properties to the console and save a visualization as `my_first_graph.png`.

---

## 🔭 What's Next?
```
Week 1-4  ✅  Q-Learning, Basic DQN, Improved DQN, LunarLander
Week 5    ✅  Learn About Graphs & PyTorch Geometric    ← You are here
Week 6    🔜  Build Your First GCN
Week 7-8  🔜  Use Your Own Graph
Week 9-12 🔜  Combine DQN + GCN
```

---
## 👨‍💻 Author
**Rikesh Yadav**  
Research Intern — International Internship Pilot Program (IIPP)  
**Laboratory:** Intelligent Networks and Edge-Cloud Computing (INEC Lab)  
**Institution:** Yuan Ze University, Taoyuan, Taiwan  
