# Research Upgrade Plan — Adaptive Evacuation GCN-DQN

## 1. Research objective

Transform the internship implementation into a reproducible research study on **graph-based deep reinforcement learning for adaptive indoor evacuation under dynamic hazards**.

The paper will focus on a single-agent GCN-based Double DQN policy. Classical path planning is retained as a reference/baseline, while CNN-DQN is retained as the primary learned baseline. Hybrid GNN-A* and MARL are removed from the active research codebase.

## 2. Core research question

> Does graph-structured state representation improve the robustness and generalization of reinforcement-learning evacuation policies when building topology, spatial scale, and hazard conditions change?

## 3. Secondary research questions

RQ1. Does a GCN-based policy outperform a CNN-based DQN under the same environment, reward, training budget, and evaluation protocol?

RQ2. Does graph-based representation retain performance better under unseen building layouts or larger grid sizes?

RQ3. How much does hazard-aware reward shaping contribute to evacuation success, hazard exposure, and path efficiency?

RQ4. What architectural factors affect scaling, especially GCN depth and agent-centric node extraction?

RQ5. What failure modes emerge as the environment becomes larger or more topologically complex?

## 4. Required model set

### M0 — Classical reference
- Hazard-aware A*.
- Optional ordinary A* only where needed to demonstrate why static shortest-path routing can fail under dynamic hazards.

### M1 — CNN-Double-DQN baseline
- Same environment.
- Same action space.
- Same reward function.
- Same training budget.
- Same seeds/evaluation protocol.
- CNN operates on the grid representation.

### M2 — Current GCN-Double-DQN
- 4-connected grid graph.
- 9 node features: 8 one-hot cell types + visit count.
- 5 GCN layers in the current configuration, 64 hidden units each.
- Agent-node embedding.
- MLP Q-value head: 128 → 64 → 5.
- Double-DQN target.

### M3 — Improved model(s), only if experiments justify them
Candidate changes:
- Dueling Double DQN.
- GAT instead of or alongside GCN.
- Agent-centric + global graph context.
- Better graph masking for non-navigable cells.
- Improved hazard representation.

Do not add M3 components simply to make the architecture look more complex. Each change must address an observed limitation and be experimentally evaluated.

## 5. Controlled experimental principles

Every comparison must control:
- environment maps;
- reward function;
- action space;
- episode budget;
- evaluation episodes;
- random seeds;
- checkpoint-selection rule;
- termination/truncation rules;
- hazard dynamics;
- start-position distribution.

Each final experiment should use multiple independent seeds where computationally feasible and report mean ± standard deviation. Success-rate confidence intervals should also be retained for the final evaluation.

## 6. Experiment matrix

### E0 — Reproduce current result
Reproduce the five-map GCN-DQN evaluation before changing the model.

### E1 — CNN vs GCN on matched conditions
Compare M1 and M2 on identical training/evaluation conditions.

Metrics:
- success rate;
- average steps;
- average successful steps;
- average reward;
- hazard events;
- path efficiency;
- training convergence.

### E2 — Seen vs unseen topology
Train on a declared subset of layouts and test on layouts excluded from training.

The exact split must be fixed before running the experiment. The current repository trains by randomly sampling all five maps, so existing results must not be described as zero-shot transfer without reconstructing the exact historical run.

### E3 — Scale shift
Train on smaller grids/layouts and evaluate on progressively larger layouts.

Primary analysis:
- performance retention versus node count;
- success degradation;
- step growth;
- hazard-event growth.

### E4 — Hazard shift
Train under one fire-spread regime and evaluate under controlled alternative regimes.

Candidate regimes should be documented before execution.

### E5 — Reward ablation
Compare the current reward with a reduced reward that removes the A*-distance progress component while keeping all other terms identical.

Purpose: determine whether the hazard-aware/progress shaping mechanism materially contributes to learning.

### E6 — Graph representation ablation
Candidate comparisons:
- current graph representation;
- graph with navigation-aware edge masking;
- agent-centric representation versus a carefully defined graph-level pooling alternative.

### E7 — GCN depth ablation
Compare a small set of depths, for example 2, 3, and 5 layers, with all other settings fixed.

Do not assume deeper is better.

### E8 — Improved architecture
Only after E1–E7 identify a clear weakness should an improved architecture be introduced.

## 7. Metrics

### Primary
- Success rate.

### Safety
- Hazard events per episode.
- Fire-caught episodes.
- Smoke exposures/steps, if retained as a clearly defined metric.

### Efficiency
- Average steps per episode.
- Average successful steps.
- Path efficiency, using the exact implementation-defined formula.

### Learning
- Episode reward.
- Rolling success rate.
- Loss.
- Episodes to predefined convergence criterion, if a criterion is fixed in advance.

### Generalization
- Seen-layout performance.
- Unseen-layout performance.
- Performance retention under scale/topology shift.

## 8. Statistical protocol

For each learned model:
- use the same seed set across compared methods;
- report per-seed results before aggregation;
- report mean ± standard deviation;
- use confidence intervals for proportions where appropriate;
- use paired comparisons when the same episode seeds are used across methods;
- avoid claiming statistical superiority unless the test and assumptions are appropriate.

## 9. Failure analysis

The Mall result is retained as a scientific failure case rather than removed.

Investigate:
- long-horizon credit assignment;
- graph message-passing depth/receptive field;
- hazard growth;
- route branching;
- repeated-state behavior;
- insufficient training diversity;
- action-space limitations;
- graph construction around walls/hazards.

The goal is to explain *why* performance degrades rather than simply report that it degrades.

## 10. Required figures

F1. System architecture.

F2. Grid-to-graph representation with node features and 4-connected edges.

F3. Dynamic hazard scenario showing ordinary shortest path versus hazard-constrained route.

F4. Training curves for CNN-DQN and GCN-DQN.

F5. Success rate versus environment scale/complexity.

F6. Hazard events versus environment scale.

F7. Path efficiency comparison.

F8. Representative successful and failed trajectories in the largest environment.

F9. Ablation summary.

Only figures supported by actual experiments will enter the paper.

## 11. Manuscript contribution target

The paper should not claim novelty merely from combining GCN and DQN. The intended contribution is an empirical and methodological study of graph-structured RL for adaptive evacuation, emphasizing:

1. topology-aware graph representation;
2. learned action selection using a graph-based Double DQN;
3. explicit treatment of dynamic hazards in the environment and reward/reference planning;
4. controlled comparison with a grid-based CNN-DQN baseline;
5. evaluation under topology/scale/hazard shifts;
6. systematic analysis of scaling limitations and failure modes.

The final contribution wording will be revised after experiments and literature review.

## 12. Manuscript order

1. Title
2. Abstract
3. Introduction
4. Related Work
5. Problem Formulation
6. Proposed Method
7. Experimental Methodology
8. Results
9. Discussion
10. Limitations
11. Conclusion

## 13. Publication gate

The manuscript is not considered submission-ready until:
- implementation and manuscript agree;
- all historical results have provenance;
- baseline comparisons are controlled;
- final experiments use documented seeds;
- no unsupported zero-shot/generalization claims remain;
- all equations match code;
- all figures can be regenerated;
- all numerical tables come from stored result files;
- limitations are explicitly discussed;
- related work has been updated against current literature.
