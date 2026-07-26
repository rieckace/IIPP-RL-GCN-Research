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

## Next Actions

1. **Phase 2:** Build DQN agent with Experience Replay, Target Network, and training loop for the evacuation environment.
2. **Phase 3:** Implement GNN feature extractor (GCN/GAT) using `get_graph_observation()`.
3. **Phase 4:** Fuse GNN encoder with DQN into hybrid GNN-DQN architecture.
4. **Phase 5:** Baselines (A*, random, vanilla DQN), ablation studies, statistical evaluation.
5. **Phase 6:** Draft research paper with methodology, experiments, and results.
