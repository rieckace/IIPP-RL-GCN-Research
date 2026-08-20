# Phase -1.1 — Repository Integrity & Provenance Audit

Date: 2026-08-20
Status: COMPLETE — baseline blocked from new research experiments until E0 resolves the listed provenance/configuration issues.

## 1. Repository scope audit

The cleaned repository contains the active research components only:

- single-agent evacuation environment;
- five benchmark map definitions;
- CNN-Double-DQN baseline;
- GCN-Double-DQN model;
- A*/hazard-aware A* utilities;
- training/evaluation utilities;
- tests;
- visualization utilities relevant to research;
- active LaTeX manuscript skeleton;
- research planning/audit documents.

Removed internship/tutorial systems are not present in the active tree. Hybrid GNN-A* and MARL are excluded.

## 2. Verified current GCN model

Checkpoint inspection of `checkpoints/gnn/best_model.pt` and `final_model.pt` confirms:

- input feature dimension: 9;
- five GCN layers, each outputting 64 dimensions;
- MLP dimensions: 64 -> 128 -> 64 -> 5;
- five action outputs;
- agent-node extraction rather than global pooling;
- Adam optimizer;
- Huber loss;
- gradient clipping at 1.0;
- Double-DQN target calculation;
- replay buffer capacity: 50,000;
- batch size: 64;
- gamma: 0.90;
- configured learning rate: 0.001;
- target update every 10 episodes.

The checkpoint state dictionaries confirm the five-layer, 9-input, 5-output architecture directly.

## 3. Verified graph/state representation

Current graph conversion uses:

- one node per grid cell;
- row-major node ordering;
- 4-connected bidirectional edges;
- eight one-hot cell-type channels;
- one visit-count channel;
- total node feature dimension: 9.

The eight cell types are EMPTY, WALL, AGENT, EXIT, OBSTACLE, SMOKE, FIRE, SENSOR.

There are no current Path Distance or Agent Distance node features in the graph conversion.

## 4. Verified action space

Five actions are implemented:

0. UP
1. DOWN
2. LEFT
3. RIGHT
4. STAY

No diagonal actions are implemented.

## 5. Critical reward/configuration discrepancy

`environment/make_env.py` sets `exit_progress_scale: 1.0` in its configuration dictionary.

However, `EvacuationEnv.__init__()` constructs `RewardConfig` without passing `exit_progress_scale`. The dataclass default is 0.0.

Therefore, in the current execution path through `make_env()`, the A*-distance progress term in `compute_reward()` is effectively DISABLED unless this constructor path is changed.

This is a publication-blocking discrepancy.

Do NOT describe the current baseline as using A*-distance reward shaping until E0 verifies the historical run and the intended/current configuration is explicitly fixed and rerun.

## 6. Critical training reproducibility issue

`training/train_gnn.py` accepts a seed and passes episode numbers to `env.reset(seed=episode)`, but map selection is performed using the global `random.choice()` without first seeding the Python random generator from the configured training seed.

Therefore the current training script does not provide full run-level reproducibility from the declared `training.seed` alone.

This must be fixed before multi-seed experiments.

## 7. Current training split

The current GCN training script samples uniformly from:

- office
- apartment
- school
- hospital
- mall

for each episode.

Therefore the current training script is NOT an unseen-layout/zero-shot protocol.

Any historical zero-shot claim must be traced to a separate, identifiable run/configuration before it can be used.

## 8. Current map dimensions

The current map implementations return:

- Office: 10x10
- Apartment: 14x14
- School: 18x18
- Hospital: 22x22
- Mall: 26x26

This conflicts with older documentation that describes the Mall as 30x30.

The active manuscript must use the dimensions verified from the map code for any new experiment. Historical 30x30 claims require provenance verification.

## 9. Historical evaluation artifact

`results/historical/internship_evaluation_results.csv` contains the reported 100-episode results:

- Office: 100.0%
- Apartment: 89.0%
- School: 72.0%
- Hospital: 77.0%
- Mall: 35.0%

with seeds documented as 2000–2099.

These values are preserved as historical evidence, not yet declared reproducible under the current code.

## 10. Historical training artifact

`results/historical/gnn_training_metrics.csv` contains exactly 1,500 episode rows.

The final recorded epsilon is approximately 0.0496, despite the configured minimum being 0.01. This is consistent with 1,500 multiplicative decay steps at 0.998 without reaching the floor.

This should be described as the observed training schedule rather than claiming that epsilon reached 0.01 during the 1,500-episode run.

## 11. Current test status

The test suite cannot currently be executed in the audit container because `gymnasium` and `torch_geometric` are not installed in the environment used for this audit.

This is an environment/dependency limitation, not evidence of test failure.

E0 must establish the declared environment from `requirements.txt`/`environment.yml`, install dependencies, and run the full test suite.

## 12. Git/provenance status

The cleaned repository is distributed as a research snapshot and does not retain the original repository's Git metadata.

A fresh Git repository should be initialized for the cleaned research artifact before E0 so every new experiment can record an exact commit.

## 13. Publication blockers before E0 completion

1. Resolve the `exit_progress_scale` constructor discrepancy.
2. Reconstruct which configuration/checkpoint generated the historical results.
3. Verify current map dimensions, especially Mall.
4. Establish deterministic multi-seed training.
5. Install declared dependencies and run tests.
6. Verify path-efficiency calculation and its handling of unsuccessful episodes.
7. Verify CNN baseline input representation so the CNN-vs-GCN comparison is controlled.
8. Only then begin new comparative experiments.

## 14. Research decision

The repository is considered structurally clean and correctly scoped.

No architecture improvement is authorized yet.

The next phase is E0 reproducibility/provenance, not model modification.
