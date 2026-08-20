# Implementation Audit — Baseline Before Research Upgrade

## Verified from current repository

### Action space
5 actions:
- UP
- DOWN
- LEFT
- RIGHT
- STAY

### Graph
- Node ordering: row-major.
- One node per grid cell.
- 4-connected adjacency.
- Bidirectional edges.

### Node features
9 dimensions:
1. EMPTY
2. WALL
3. AGENT
4. EXIT
5. OBSTACLE
6. SMOKE
7. FIRE
8. SENSOR
9. visit count

The feature dimensions are therefore 8 one-hot cell-type channels plus one visit-count channel.

### GCN-DQN
Current configured network:
- GCN hidden dimensions: [64, 64, 64, 64, 64]
- ReLU after each GCN layer
- agent-node extraction using the AGENT feature
- MLP: 128 → 64 → 5
- Adam optimizer
- Huber loss
- gradient clipping at 1.0
- Double-DQN target calculation
- replay buffer: 50,000
- batch size: 64
- gamma: 0.90
- learning rate: 0.001
- epsilon: 1.0 → minimum 0.01 with multiplicative decay 0.998
- hard target update every 10 episodes

### Environment
- Fire spreads probabilistically to adjacent EMPTY/SMOKE cells.
- Smoke is recomputed around fire cells using a Manhattan-radius rule.
- Episodes terminate at exit or fire contact.
- Episodes truncate at a map-dependent maximum step count.

### Reward
The current reward implementation combines:
- exit reward;
- fire penalty;
- smoke penalty;
- wall-bump penalty;
- normal movement penalty;
- stay penalty;
- revisit penalties;
- the reward function contains code for progress shaping based on the change in A* path distance to the nearest exit, but the current `EvacuationEnv` constructor does not pass `exit_progress_scale` into `RewardConfig`; therefore the current `make_env()` execution path disables that term. See `PHASE_MINUS1_1_AUDIT.md`.

## Known inconsistencies requiring resolution before publication

1. Historical manuscript says 3 GCN layers; current config uses 5.
2. Historical manuscript says 8 movement actions; current implementation has 5.
3. Historical manuscript describes additional distance node features that are not in the current graph state.
4. Current training script samples all five maps; therefore existing results must not automatically be described as zero-shot transfer from only Office/School.
5. Current evaluation uses a best checkpoint trained by a historical configuration; exact provenance must be reconstructed.
6. Path efficiency is implemented as an A*-reference ratio for successful episodes and must be documented exactly.
7. Some tests in the original project assumed 8 node features and required correction.
8. The current project contains obsolete hybrid/MARL code and historical documentation that can create ambiguity.

## Rule
No manuscript statement becomes authoritative until it is supported by the final code, final experiment record, or a cited external source.
