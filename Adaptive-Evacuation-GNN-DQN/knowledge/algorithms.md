# Algorithms & Architectures Used

## 1. Deep Q-Network (DQN)
Used as the Baseline Phase 2 model.
- Uses a standard Multi-Layer Perceptron (MLP) with dimensions: `[256, 256, 128]`.
- Implements Double Q-Learning to prevent Q-value overestimation.
- Replay Buffer of size 50,000 for experience replay.

## 2. Graph Neural Network (GNN)
Used as the advanced Phase 3 model.
- Uses PyTorch Geometric (PyG).
- Treats the grid as a graph, where each cell is a Node and adjacent cells are connected by Edges.
- Graph Convolutional Network (`GCNConv`): 3 layers of message passing `[64, 64, 64]`.
- Global Mean Pooling aggregates the node features into a single graph embedding.
- MLP Head `[128, 64]` processes the embedding to output 5 Q-values (actions).

## 3. A* Search (Heuristic)
Used in the Phase 4 Hybrid model.
- Calculates the absolute shortest Manhattan-distance path from the agent's current position to the nearest exit.
- Treats walls as impassable, but ignores fire and smoke.
- Output is fed as a binary flag feature into the GNN's observation.

## 4. Cellular Automata
Used in Phase 1 Environment generation.
- Spreads fire probabilistically from burning cells to adjacent empty cells at each time step.
