# Active Research Project Scope

## Active system

**Adaptive Evacuation GCN-Double-DQN**

The active project is a single-agent indoor evacuation system with:

- configurable grid-based building layouts;
- dynamic fire and smoke propagation;
- 5 discrete actions: up, down, left, right, stay;
- graph conversion using 4-connected adjacency;
- 9 node features: 8 one-hot cell types + visit count;
- GCN encoder;
- agent-centric node representation;
- Double-DQN learning;
- experience replay and target network;
- hazard-aware progress reward using A* path distance;
- multi-scale benchmark layouts.

## Retained baselines

- CNN-based Double-DQN.
- A* / hazard-aware A* reference.
- Random baseline where useful for sanity checks.

## Removed from active project

- Q-learning tutorial implementations.
- CartPole/LunarLander learning exercises.
- standalone GNN tutorials.
- Hybrid GNN-A* feature-injection implementation.
- Multi-Agent Reinforcement Learning implementation.
- MARL environment and tests.
- Web dashboard/frontend.
- internship-week folders.
- duplicate placeholder scripts.
- obsolete experiment logs that mix historical and current configurations.

These implementations belong to the development history, not the active research artifact.

## Important distinction

A* remains in the active project because it is needed for the hazard-aware reference/progress calculation and for classical baseline experiments. This does **not** make A* the learned evacuation policy.
