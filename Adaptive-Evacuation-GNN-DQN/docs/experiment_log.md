# Experiment Log: Phase 3 & 4 Validations

## Experiment 1: DQN vs GNN Comparison
**Date**: July 28, 2026
**Hypothesis**: A Graph Neural Network (GNN) will generalize to new grid sizes natively, whereas a standard Flat-MLP DQN will not.
**Methodology**:
- Trained both a DQN and a GNN on a fixed 10x10 grid with 5% fire spawn probability.
- Training budget: 1500 episodes.
- Early Stopping: 90% success rate across 100 trailing episodes.

**Results**:
- **Baseline DQN**: Stopped at Episode 443. (Fast convergence via memorization).
- **GNN-DQN**: Stopped at Episode 938. (Slower convergence due to message-passing complexity).
- **Transfer Test**: Evaluated both models on an unseen 15x15 grid.
  - DQN: Crashed (Tensor dimension mismatch: Expected 800, got 1800).
  - GNN: 100% Success Rate (Zero-Shot Transfer).

**Conclusion**: The GNN successfully learns the mechanical rules of the environment and is agnostic to building size.

---

## Experiment 2: Hybrid GNN-A* Efficacy
**Date**: July 28, 2026
**Hypothesis**: Providing the GNN with a heuristic shortest-path "hint" will drastically reduce the number of episodes required to learn the environment.
**Methodology**:
- Implemented `AStarPlanner` to calculate the shortest path to the exit, ignoring the fire.
- Added a 9th node feature flag (1.0 if on optimal path, 0.0 otherwise).
- Trained the Hybrid GNN-A* on the same 10x10 grid.

**Results**:
- **Pure GNN**: 938 Episodes to converge.
- **Hybrid GNN-A***: 691 Episodes to converge.

**Conclusion**: The Hybrid approach reduced training time by **26.3%**. The agent leverages the A* heuristic to know *where* the exit is from Episode 1, allowing the RL network to dedicate all its capacity to learning *how to navigate around the fire*.

---

## Experiment 3: Multi-Agent Coordination (MARL)
**Date**: July 28, 2026
**Hypothesis**: By sharing a single GNN and extracting Node-Specific Embeddings for each agent's coordinate, multiple agents can navigate the grid simultaneously and avoid bottlenecks using a shared cooperative "Team Bonus" reward.
**Methodology**:
- Upgraded the GNN to extract node features at exact agent coordinates rather than using Global Mean Pooling.
- Environment updated with collision logic (agents cannot occupy the same cell).
- Team Bonus (+20) awarded to all active agents when one agent successfully escapes.
- Trained 3 agents simultaneously on a 10x10 grid for 1500 episodes.

**Results**:
- **Success Rate**: 0.0%
- Agents successfully learned basic movement but repeatedly fell into deadlocks (blocking each other in corridors).

**Conclusion**: While Node-Specific Extractors allow multiple agents to run through a shared GNN, implicit spatial awareness is insufficient for complex coordination. Without explicit communication channels (e.g., passing messages directly between agents), they cannot resolve bottlenecks effectively.
