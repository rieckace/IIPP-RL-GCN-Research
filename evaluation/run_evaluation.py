"""Deterministic multi-map evaluation for the active GCN-Double-DQN model."""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from environment.constants import CellType
from environment.heuristics import AStarPlanner
from environment.make_env import make_env
from environment.wrappers import GraphObservationWrapper
from models.gnn.trainer import GNNDQNAgent
from utils.config_loader import load_config
from utils.seed import seed_everything

MAPS = ["office", "apartment", "school", "hospital", "mall"]


def evaluate_map(agent: GNNDQNAgent, map_name: str, episodes: int, seed_start: int) -> dict:
    base_env = make_env(map_name)
    base_env.randomize_agent_start = True
    env = GraphObservationWrapper(base_env)

    successes = 0
    total_steps = []
    successful_steps = []
    total_rewards = []
    total_hazards = 0
    path_efficiencies = []

    for offset in range(episodes):
        seed = seed_start + offset
        seed_everything(seed)

        obs, _ = env.reset(seed=seed)
        start_pos = base_env.state.agent_positions[0]

        exits = {
            (r, c)
            for r in range(base_env.grid.rows)
            for c in range(base_env.grid.cols)
            if base_env.grid.get_cell(r, c) == CellType.EXIT
        }
        reference_path = AStarPlanner.compute_path(base_env.grid, start_pos, exits)
        reference_length = len(reference_path) - 1 if reference_path else 0

        reward_sum = 0.0
        steps = 0
        hazards = 0

        while True:
            action = agent.act(obs, explore=False)
            obs, reward, terminated, truncated, info = env.step(action)
            reward_sum += reward
            steps += 1

            reason = info.get("reason", "")
            if reason in {"in_smoke", "hit_fire", "fire_caught_agent"}:
                hazards += 1

            if terminated or truncated:
                break

        total_steps.append(steps)
        total_rewards.append(reward_sum)
        total_hazards += hazards

        if info.get("reason") == "reached_exit":
            successes += 1
            successful_steps.append(steps)
            if reference_length > 0:
                path_efficiencies.append(reference_length / steps)

    env.close()

    return {
        "environment": map_name,
        "episodes": episodes,
        "successful_episodes": successes,
        "failed_episodes": episodes - successes,
        "success_rate": successes / episodes,
        "avg_steps": float(np.mean(total_steps)),
        "avg_steps_success": float(np.mean(successful_steps)) if successful_steps else 0.0,
        "avg_reward": float(np.mean(total_rewards)),
        "total_hazard_events": total_hazards,
        "avg_hazard_events": total_hazards / episodes,
        "path_efficiency": float(np.mean(path_efficiencies)) if path_efficiencies else float("nan"),
        "seed_start": seed_start,
        "seed_end": seed_start + episodes - 1,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "gnn.yaml"))
    parser.add_argument("--checkpoint", default=str(PROJECT_ROOT / "checkpoints" / "gnn" / "best_model.pt"))
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=2000)
    parser.add_argument("--maps", nargs="+", default=MAPS, choices=MAPS)
    parser.add_argument("--output", default=str(PROJECT_ROOT / "results" / "tables" / "gnn_evaluation.csv"))
    args = parser.parse_args()

    config = load_config(args.config, validate=False)
    agent = GNNDQNAgent(config)
    agent.load_checkpoint(args.checkpoint)
    agent.epsilon = 0.0
    agent.q_network.eval()

    results = [
        evaluate_map(agent, name, args.episodes, args.seed_start)
        for name in args.maps
    ]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(results[0].keys())
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    print("environment | success | avg steps | avg reward | hazards/ep | path efficiency")
    for r in results:
        pe = "N/A" if np.isnan(r["path_efficiency"]) else f"{100*r['path_efficiency']:.2f}%"
        print(
            f"{r['environment']:10} | {100*r['success_rate']:6.1f}% | "
            f"{r['avg_steps']:9.2f} | {r['avg_reward']:10.2f} | "
            f"{r['avg_hazard_events']:10.2f} | {pe}"
        )

    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
