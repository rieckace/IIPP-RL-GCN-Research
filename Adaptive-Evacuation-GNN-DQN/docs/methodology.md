# Research Methodology

## Objective
To develop an adaptive, robust pathfinding agent capable of navigating dynamic hazardous environments (e.g., spreading fire and smoke) using Deep Reinforcement Learning.

## Architecture Evolution
1. **Baseline DQN**: Standard Deep Q-Network relying on a flattened grid representation. Serves as a baseline to demonstrate the limitations of fixed-size architectures.
2. **GNN-DQN**: Replaces the standard MLP with a Graph Neural Network (GCNConv). The grid is treated as a graph where each cell is a node. This allows for Zero-Shot Transfer to building layouts of any size without retraining.
3. **Hybrid GNN-A***: Integrates traditional pathfinding heuristics. A* computes the shortest Manhattan path to the exit, and this path is injected into the GNN node features as a binary flag. The GNN uses this as a "hint" to dramatically speed up convergence.
4. **MARL GNN**: Introduces Node-Specific Feature Extractors. Instead of compressing the environment using Global Mean Pooling, multiple agents share a single GNN and extract output Q-values strictly from the specific node coordinate they are currently standing on.

## Environment Dynamics
- **Fire Spread**: Modeled using Cellular Automata. Fire spreads probabilistically to adjacent non-wall cells.
- **Smoke**: Generated automatically in a Manhattan radius around active fire cells.
- **Rewards**:
  - Reaching Exit: +100
  - Hitting Fire: -100
  - Step Penalty: -1
  - Hitting Wall: -5
