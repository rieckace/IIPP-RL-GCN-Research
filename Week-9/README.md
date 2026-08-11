# 🔗 Week 9 - The Big Idea: Combining GCN with DQN

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Internship](https://img.shields.io/badge/INEC%20Lab-Yuan%20Ze%20University-orange.svg)

This week marks the beginning of the final phase of the internship! We are finally combining the two major concepts we've studied: **Deep Q-Networks (DQN)** from Weeks 1-4 and **Graph Convolutional Networks (GCN)** from Weeks 5-8.

---

## 📖 Theory: Why combine them?

As we saw in Week 4, a standard DQN uses simple Linear layers (or Convolutional Neural Networks) to process its state. This works fine for a video game with a fixed screen size (like CartPole or LunarLander).

But what if we want to build an AI that navigates through buildings?
- An Office is 10x10.
- A Mall is 30x30.
If we train a standard DQN on the Office, its neural network expects exactly 100 inputs. If we drop that same trained agent into the Mall (900 inputs), the network crashes. It cannot **generalize** to new sizes!

### The Solution: GCN to the Rescue!
GCNs process nodes and edges, regardless of how many there are.
1. We use a **GCN** as the "eyes" of the agent. It looks at the building graph and processes the environmental context (hazards, walls, corridors) into a fixed-size latent vector (e.g., 64 numbers).
2. We pass that fixed-size vector into the **DQN**. The DQN no longer needs to worry about the size of the building; it just looks at the 64 numbers and decides whether to move Up, Down, Left, or Right!

---

## 🎨 Task: Block Diagram (GCN $\rightarrow$ DQN)

Here is the architectural diagram of our hybrid model:

```text
======================================================================
                        GCN-DQN ARCHITECTURE
======================================================================

[ 1. RAW GRAPH DATA ]
   |
   |-- Nodes (Tiles, Agent Position, Hazards)
   |-- Edges (Valid walking paths)
   V

[ 2. GCN LAYERS ] (Spatial Processing)
   |
   |-- GCNConv Layer 1: Gathers local neighbor context
   |-- ReLU
   |-- GCNConv Layer 2: Gathers wider building context
   V

[ 3. AGENT-CENTRIC POOLING ]
   |
   |-- We extract only the output vector for the specific 
   |   node where our Agent is currently standing!
   |   (e.g., a [64] dimensional vector)
   V

[ 4. DQN LAYERS ] (Decision Making)
   |
   |-- Linear(64, 128)
   |-- ReLU
   |-- Linear(128, 4) --> Outputs Q-Values for [UP, DOWN, LEFT, RIGHT]
   V
   
[ 5. ACTION EXECUTION ]
======================================================================
```

## ▶️ How to Run
```bash
run_week.cmd
```
This prints the conceptual GCN-to-DQN tensor flow demonstration in the terminal.

---

## 🔭 What's Next?
Now that we have the architecture mapped out conceptually, next week we will actually write the PyTorch code for this hybrid GCN-DQN model!

```
Week 1-8  ✅  RL Basics & GCNs
Week 9    ✅  Learn the Idea: Combining DQN and GCN   ← You are here
Week 10   🔜  Build GCN-DQN Model
Week 11   🔜  Train and Test
Week 12   🔜  Final Mini Project (Adaptive Evacuation!)
```

---
## 👨‍💻 Author
**Rikesh Yadav**  
Research Intern — International Internship Pilot Program (IIPP)  
**Laboratory:** Intelligent Networks and Edge-Cloud Computing (INEC Lab)  
**Institution:** Yuan Ze University, Taoyuan, Taiwan  
