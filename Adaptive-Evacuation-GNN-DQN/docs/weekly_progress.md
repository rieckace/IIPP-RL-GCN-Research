# Weekly Progress Tracker

## Week 1
- **Focus**: RL Fundamentals
- **Tasks Completed**: Reviewed Q-Learning, basic MDPs, and implemented standard GridWorld environments.

## Week 2
- **Focus**: Deep Q-Networks (DQN)
- **Tasks Completed**: Transitioned from tabular methods to Deep RL. Researched DQN architectures and drafted the initial sections of the Overleaf research manuscript.

## Week 3
- **Focus**: Training Stability & Hyperparameters
- **Tasks Completed**: Implemented CartPole-v1. Experimented with epsilon decay, target network updates, and learning rates to stabilize training.

## Week 4
- **Focus**: Mid-Term Report & Project Setup
- **Tasks Completed**: Finalized and submitted the Mid-Term report. Laid out the software architecture for the primary research project (Adaptive Evacuation GNN-DQN).

## Week 5
- **Focus**: Phase 1 - Environment Engineering
- **Tasks Completed**: Built `EvacuationEnv` using the Gymnasium API. Implemented cellular automata logic for dynamic fire and smoke spreading. Added ANSI terminal rendering.

## Week 6
- **Focus**: Phase 2 & 3 - DQN to GNN Transition
- **Tasks Completed**: Built and trained the Baseline Double DQN. Discovered its limitation with fixed grid sizes. Integrated `torch_geometric` and built the `GNNDQNetwork` to allow for zero-shot transferability to larger grids (e.g., 15x15).

## Week 7
- **Focus**: Phase 4 - Hybrid Heuristic System & Experiments
- **Tasks Completed**: Conducted full 1500-episode experiments comparing DQN vs GNN. Implemented the `AStarPlanner` to feed shortest-path hints into the GNN's node features. The Hybrid GNN-A* successfully cut training time by 26%. All code and documentation finalized.

## Week 8
- **Focus**: Phase 5 - Multi-Agent Systems (MARL)
- **Tasks Completed**: Overhauled the environment to support multiple simultaneous agents and collision handling. Upgraded the GNN to extract node-specific embeddings (bypassing global pooling). Ran a 3-agent experiment which yielded a 0% success rate, successfully demonstrating that explicit communication networks are required to solve multi-agent bottleneck deadlocks.
