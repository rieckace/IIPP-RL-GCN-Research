# Phase E0 — Baseline Reproducibility & Provenance

## Status

**In progress.** Structural fixes are applied; numerical reproduction requires the declared Python dependencies and a runtime capable of executing Gymnasium/PyTorch Geometric.

## Baseline artifacts

- GCN checkpoint: `checkpoints/gnn/best_model.pt`
- GCN final checkpoint: `checkpoints/gnn/final_model.pt`
- Historical evaluation: `results/historical/internship_evaluation_results.csv`
- Historical training log: `results/historical/gnn_training_metrics.csv`

## Verified artifact hashes

- `best_model.pt`: `3190a863f278529f81fa4256d42ad1b97938a532d5aea25b56327dca58e9268e`
- `final_model.pt`: `79f57ac5c0ace7823d3e528ff9c015c66e4db13b7f262ed3068992c1cc39fff1`
- `internship_evaluation_results.csv`: `25d998474d730ac5af942995497adf18de229baf43241087825b51c2b28452b7`
- `gnn_training_metrics.csv`: `559129c6d349f76d9874616fe6b8f36a94e40d2ff9091942e346521af4cabc2f`

## Historical results

The historical CSV reports 100 deterministic evaluation episodes per map with seed range 2000–2099:

- Office: 100.0%
- Apartment: 89.0%
- School: 72.0%
- Hospital: 77.0%
- Mall: 35.0%

These remain **historical results** until the current locked code reproduces them under an explicitly recorded configuration.

## E0 fixes applied

1. `EvacuationEnv` now passes `exit_progress_scale` and `team_bonus` from configuration into `RewardConfig`.
2. Training now seeds Python, NumPy, and PyTorch and uses an isolated seeded RNG for map selection.
3. Evaluation uses the shared seed utility.
4. Training output now reports the verified current range 10x10 to 26x26 rather than the stale 30x30 description.
5. Added `scripts/e0_verify.py` for structural/runtime verification.

## Remaining E0 gates

- Install `requirements.txt` in the execution environment.
- Run `python -m pytest -q`.
- Run `python scripts/e0_verify.py`.
- Run the historical 100-episode GCN evaluation with the identified checkpoint.
- Compare the newly generated CSV with the historical artifact.
- If results differ, preserve both and investigate configuration/provenance rather than overwriting the historical record.

## Important interpretation rule

The historical result table is not yet treated as a reproducible benchmark of the current code. The checkpoint filename alone does not establish the exact training configuration that produced it.
