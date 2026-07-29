# Adaptive Evacuation GNN-DQN: Project Plan

## Phase 1: Environment and Core Grid System ✅
- **Goal**: Create a custom, Gymnasium-compatible grid environment for fire evacuation.
- **Features**: Procedural 10x10 grids, cellular automata fire and smoke spreading dynamics, multi-objective reward structure, and real-time ANSI terminal rendering.
- **Status**: Completed.

## Phase 2: Baseline Deep Q-Network (DQN) ✅
- **Goal**: Implement a standard RL agent to serve as a performance baseline.
- **Features**: Double DQN architecture with Replay Buffer and Epsilon-Greedy exploration. Network relies on a flattened 800-dimension vector of the grid.
- **Status**: Completed. Converged in 443 episodes on 10x10 grids but failed to transfer to larger grids.

## Phase 3: Graph Neural Network (GNN-DQN) ✅
- **Goal**: Upgrade the agent's architecture to understand spatial structure via graphs.
- **Features**: PyTorch Geometric (PyG) integration. State converted into `node_features` and `edge_index`. Uses 3 `GCNConv` layers for message passing.
- **Status**: Completed. Converged in 938 episodes and achieved 100% Zero-Shot Transfer to unseen 15x15 grids.

## Phase 4: Hybrid Heuristic Integration (A* + GNN) ✅
- **Goal**: Speed up RL convergence by injecting classical pathfinding hints.
- **Features**: Dynamically calculates the shortest path (A*) to the exit at every step, ignoring fire. Appends this optimal path to the GNN's node features as a 9th dimension.
- **Status**: Completed. Cut training time by 26% (Converged in 691 episodes) while retaining perfect size generalization.

## Phase 5: Multi-Agent Systems (MARL) ✅
- **Goal**: Introduce multiple agents evacuating simultaneously.
- **Features**: Subclassed `MARLEvacuationEnv` with collision avoidance logic and team-based cooperative rewards. Replaced the GNN Global Mean Pooling with Node-Specific Embeddings, allowing multiple agents to act independently based on their exact physical coordinate.
- **Status**: Completed. Training yielded a 0% success rate, proving that implicit spatial awareness is insufficient for multi-agent coordination in bottlenecks, highlighting the necessity for explicit communication networks.

## Phase 6: Front-End Visualization (Proposed) ⏳
- **Goal**: Build a graphical user interface (GUI) or web application to visualize the evacuation process, moving away from terminal ANSI characters.
