# Adaptive Evacuation — GCN-Double-DQN Research Project

This repository is the cleaned research codebase derived from the internship implementation. It contains only the components required for the active single-agent evacuation research program.

## Research focus

Graph-structured reinforcement learning for adaptive indoor evacuation under dynamic fire/smoke hazards.

## Active models

- CNN-Double-DQN baseline
- GCN-Double-DQN proposed baseline
- A* / hazard-aware A* reference

Hybrid GNN-A* and MARL implementations are intentionally excluded from the active codebase.

## Repository structure

```text
Adaptive-Evacuation-GNN-DQN/
├── baselines/              Classical reference agents
├── configs/                Active experiment configurations
├── environment/            Grid, hazards, rewards, maps interface
├── maps/                   Office, Apartment, School, Hospital, Mall
├── models/
│   ├── common/
│   ├── dqn/                CNN-DQN baseline
│   └── gnn/                GCN-Double-DQN model
├── evaluation/             Evaluation and statistics
├── training/               Training/evaluation entry points
├── visualization/          Research figures and trajectory views
├── tests/                  Automated tests
├── research/               New experiment infrastructure
├── docs/                   Research plan and implementation audit
└── paper/manuscript/       IEEE LaTeX manuscript
```

## Important

The existing internship results are not automatically treated as final evidence for new claims. New experiments will be registered and reproduced under controlled conditions before being used in the manuscript.

See:
- `docs/RESEARCH_UPGRADE_PLAN.md`
- `docs/IMPLEMENTATION_AUDIT.md`
- `docs/EXPERIMENT_REGISTER.md`
- `docs/PROJECT_SCOPE.md`
