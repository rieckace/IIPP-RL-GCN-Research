# Research Knowledge Map

A structured overview of concepts and progress for **IIPP-RL-GCN Research**.

## Legend

- ✅ Complete
- ⬜ In Progress / Pending

---

## 1) Machine Learning Foundations

- ✅ Supervised Learning
- ✅ Unsupervised Learning
- ✅ Reinforcement Learning

## 2) Reinforcement Learning Fundamentals

- ✅ Agent
- ✅ Environment
- ✅ State
- ✅ Action
- ✅ Reward
- ✅ Policy
- ✅ Episode
- ✅ Time Step

## 3) Q-Learning

- ✅ Bellman Equation
- ✅ Q-Table
- ✅ Learning Rate (α)
- ✅ Discount Factor (γ)
- ✅ Exploration vs. Exploitation

## 4) Implementations

- ✅ GridWorld
- ✅ DQN CartPole-v1

## 5) Deep Reinforcement Learning

- ✅ Why Q-Learning Fails
- ✅ Curse of Dimensionality
- ✅ DQN Motivation

## 6) PyTorch Foundations

- ✅ Python Lists
- ✅ NumPy Arrays
- ✅ PyTorch Tensors
- ✅ Tensor Dimensionality
- ✅ `torch.tensor()`

## 7) Adaptive Evacuation Environment (Phase 1)

- ✅ Gymnasium-compatible `EvacuationEnv`
- ✅ 10×10 building grid with walls, exits, corridors
- ✅ Dynamic fire spread (probabilistic, no burn-out)
- ✅ Smoke propagation (Manhattan radius)
- ✅ Multi-objective reward function (6 event types)
- ✅ GNN-ready graph state (one-hot nodes + COO edges)
- ✅ ANSI terminal renderer
- ✅ YAML configuration system
- ✅ Unit test suite (46 tests passing)

## 8) Research Paper Progress

- ✅ Paper V1 Reviewed
- ⬜ Strengthen AI Methodology
- ⬜ Improve Experimental Evaluation

---

## 5. Next Actions / Active Focus
*   **Action:** Build the GNN-DQN Architecture.
*   **Rationale:** Standard DQN fails at spatial generalization on raw grids (requires exact coordinate memory). A GNN will mathematically map the grid structure, allowing the agent to spawn anywhere and instantly find the exit.
*   **Dependencies:** `environment/graph_utils.py` needs to be built to feed PyG Graphs to `models/gnn/network.py`.

## 6. Architecture & Rewards Updates
*   **Dense Reward (A* Distance):** The environment now uses A* shortest-path distance (rather than naive Manhattan distance) for dense rewards, preventing the agent from getting stuck behind walls.
*   **Revisit Penalty:** Added a 9th observation channel tracking cell visit counts. Revisiting cells carries heavy negative penalties to eliminate looping behavior.
