# Future Work (Phase 5 & 6)

## Explicit Communication Networks (CommNet)
In Phase 5, we discovered that multiple agents fail to coordinate in bottlenecks using only implicit spatial awareness (0% success rate). 
- **Challenge**: Agents must negotiate who goes first in narrow corridors to prevent gridlock.
- **Solution**: Implement a CommNet or TarMAC (Targeted Multi-Agent Communication) architecture, where agents can pass continuous vector messages directly to each other's hidden states before deciding their actions.

## Dynamic Obstacles
Currently, walls are static. In a real fire, structural collapses can create new obstacles dynamically.
- **Goal**: Implement collapsing ceilings or debris that blocks paths mid-episode, testing the A* + GNN's ability to recalculate and adapt on the fly.

## Visual Interface
The ANSI terminal renderer is sufficient for training, but less ideal for presentation.
- **Goal**: Build a Python `pygame` renderer or a Web UI (React/Next.js) to display the environment graphically in real-time.
