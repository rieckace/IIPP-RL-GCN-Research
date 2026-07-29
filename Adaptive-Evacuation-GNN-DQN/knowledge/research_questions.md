# Research Questions

1. **Size Generalization (Transferability)**
   *Can Graph Neural Networks (GNNs) effectively eliminate the strict size constraints of traditional Flat-MLP Reinforcement Learning agents in spatial navigation tasks?*
   - **Answered**: Yes. The GNN-DQN successfully achieved 100% zero-shot transfer from a 10x10 training environment to a 15x15 evaluation environment.

2. **Training Efficiency (Hybridization)**
   *Does providing a traditional algorithmic heuristic (like A* shortest path) as an input feature to a Deep RL agent significantly accelerate convergence compared to pure exploration?*
   - **Answered**: Yes. The Hybrid GNN-A* converged 26% faster (691 episodes vs 938 episodes) than the pure GNN.

3. **Dynamic Hazard Avoidance**
   *Can an RL agent learn to preemptively route around cellular automata-based fire spreading without knowing the exact probabilistic spread formula?*
   - **Answered**: Yes. The message-passing nature of the GNN allowed the agent to "feel" the fire approaching from adjacent nodes and successfully route around it to reach the exit safely.

4. **Multi-Agent Coordination (MARL)**
   *Is implicit spatial awareness sufficient for multiple agents to coordinate and avoid deadlocks in narrow corridors?*
   - **Answered**: No. The Phase 5 Multi-Agent experiment yielded a 0% success rate. Even with cooperative "Team Bonus" rewards and node-specific embeddings, agents physically blocked each other. This proves that explicit inter-agent communication architectures are required for bottleneck resolution.
