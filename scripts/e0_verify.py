"""Phase E0 verification for the locked Adaptive Evacuation research baseline.

Run from the repository root after installing requirements.txt.
This script performs structural/runtime checks only; it does not train a model.
"""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import torch

from environment.make_env import make_env
from environment.wrappers import GraphObservationWrapper
from models.gnn.trainer import GNNDQNAgent
from utils.config_loader import load_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "gnn.yaml"
CHECKPOINT = ROOT / "checkpoints" / "gnn" / "best_model.pt"
HIST = ROOT / "results" / "historical" / "internship_evaluation_results.csv"

EXPECTED_DIMS = {
    "office": (10, 10),
    "apartment": (14, 14),
    "school": (18, 18),
    "hospital": (22, 22),
    "mall": (26, 26),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    print("Phase E0 baseline verification")
    print("=" * 60)

    # Map dimensions + reward configuration + graph shape.
    for name, expected in EXPECTED_DIMS.items():
        env = make_env(name)
        actual = (env.rows, env.cols)
        assert actual == expected, f"{name}: expected {expected}, got {actual}"
        assert env.reward_config.exit_progress_scale == 1.0, (
            f"{name}: exit_progress_scale is "
            f"{env.reward_config.exit_progress_scale}, expected 1.0"
        )
        wrapped = GraphObservationWrapper(env)
        obs, _ = wrapped.reset(seed=1234)
        assert obs["node_features"].shape == (expected[0] * expected[1], 9)
        assert obs["edge_index"].shape[0] == 2
        wrapped.close()
        print(f"PASS {name}: {actual}, reward shaping=1.0, node features=9")

    config = load_config(str(CONFIG), validate=False)
    agent = GNNDQNAgent(config)
    agent.load_checkpoint(str(CHECKPOINT))
    assert agent.node_feature_dim == 9
    assert agent.action_size == 5
    assert len(agent.q_network.gcn_layers) == 5
    print("PASS checkpoint: 9 input features, 5 GCN layers, 5 actions")

    # Historical artifact integrity.
    with HIST.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 5
    print("PASS historical evaluation artifact: 5 environment rows")

    print(f"best_model.sha256 = {sha256(CHECKPOINT)}")
    print(f"historical_csv.sha256 = {sha256(HIST)}")
    print("\nE0 structural verification PASSED.")
    print("Run the full test suite and the 100-episode evaluation separately before declaring numerical reproduction complete.")


if __name__ == "__main__":
    main()
