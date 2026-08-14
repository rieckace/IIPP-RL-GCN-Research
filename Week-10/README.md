# 🏗️ Week 10 - Building the GCN-DQN Model

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![PyTorch Geometric](https://img.shields.io/badge/PyTorch_Geometric-2.3+-green.svg)
![Internship](https://img.shields.io/badge/INEC%20Lab-Yuan%20Ze%20University-orange.svg)

This week, we translate the conceptual block diagram from Week 9 into actual PyTorch code. We will build the hybrid neural network class that combines PyTorch Geometric (`GCNConv`) with standard PyTorch (`Linear`) layers.

---

## 💻 Implementation: The Hybrid Architecture

In `gcn_dqn_model.py`, we construct the `GNNDQNetwork` class. This is a massive step towards our final project!

### Key Design Pattern: Agent-Centric Pooling
When a GCN processes a graph, it outputs a hidden vector for *every single node* in the graph. If we have a 100-node graph, we get 100 vectors. But the DQN only needs *one* vector to make a decision!

How do we compress 100 vectors into 1?
- We could average them (Global Mean Pool). But that loses the agent's specific location data.
- **Our Solution:** We filter the outputs to grab *only* the vector corresponding to the node where the agent is currently standing! We know where the agent is because one of our node features is an `is_agent` boolean flag.

```python
# Pseudo-code for Agent-Centric Pooling
node_embeddings = self.gcn_layers(x, edge_index)

# Find the node where the agent is standing (Feature column index 2)
agent_node_mask = x[:, 2] == 1.0

# Extract just that one row!
agent_embedding = node_embeddings[agent_node_mask]

# Pass to DQN
q_values = self.dqn_layers(agent_embedding)
```

---

## ▶️ How to Run
```bash
run_week.cmd
```
This prints the Week 10 GCN-DQN model definition and architecture summary.

## 🔭 What's Next?
Now that the architecture is fully coded, we need to test if it can actually learn. Next week, we will hook this model up to a Replay Buffer and train it on a mini-environment!

```
Week 1-9  ✅  RL Basics, GCNs, and Hybrid Theory
Week 10   ✅  Build GCN-DQN Model                    ← You are here
Week 11   🔜  Train and Test
Week 12   🔜  Final Mini Project (Adaptive Evacuation!)
```

---
## 👨‍💻 Author
**Rikesh Yadav**  
Research Intern — International Internship Pilot Program (IIPP)  
**Laboratory:** Intelligent Networks and Edge-Cloud Computing (INEC Lab)  
**Institution:** Yuan Ze University, Taoyuan, Taiwan  
