# Future Work (Phase 5 & 6)

## Multi-Agent Systems
The current implementation only supports a single agent. A critical expansion for realistic evacuation modeling is Multi-Agent reinforcement learning (MARL).
- **Challenge**: Agents may bottleneck in narrow corridors.
- **Solution**: Upgrade the GNN to include agent-to-agent communication via edge features, allowing them to coordinate paths and avoid congestion.

## Dynamic Obstacles
Currently, walls are static. In a real fire, structural collapses can create new obstacles dynamically.
- **Goal**: Implement collapsing ceilings or debris that blocks paths mid-episode, testing the A* + GNN's ability to recalculate and adapt on the fly.

## Visual Interface
The ANSI terminal renderer is sufficient for training, but less ideal for presentation.
- **Goal**: Build a Python `pygame` renderer or a Web UI (React/Next.js) to display the environment graphically in real-time.
