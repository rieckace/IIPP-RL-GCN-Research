"""Command-line entry point for the active evacuation research codebase."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adaptive Evacuation research project")
    parser.add_argument(
        "--mode",
        choices=["train-dqn", "train-gnn", "evaluate-gnn", "demo-dqn"],
        required=True,
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from utils.config_loader import load_config

    if args.mode == "train-dqn":
        from training.train_dqn import train
        config_path = args.config or str(PROJECT_ROOT / "configs" / "dqn.yaml")
        config = load_config(config_path, validate=False)
        train(config, num_episodes=args.episodes, seed=args.seed)

    elif args.mode == "train-gnn":
        from training.train_gnn import train
        config_path = args.config or str(PROJECT_ROOT / "configs" / "gnn.yaml")
        config = load_config(config_path, validate=False)
        train(config, num_episodes=args.episodes, seed=args.seed)

    elif args.mode == "evaluate-gnn":
        import subprocess
        command = [sys.executable, str(PROJECT_ROOT / "evaluation" / "run_evaluation.py")]
        if args.config:
            command += ["--config", args.config]
        if args.checkpoint:
            command += ["--checkpoint", args.checkpoint]
        if args.episodes:
            command += ["--episodes", str(args.episodes)]
        if args.seed is not None:
            command += ["--seed-start", str(args.seed)]
        subprocess.run(command, check=True)

    elif args.mode == "demo-dqn":
        from scripts.demo_trained import run_demo
        config_path = args.config or str(PROJECT_ROOT / "configs" / "dqn.yaml")
        checkpoint = args.checkpoint or str(PROJECT_ROOT / "checkpoints" / "dqn" / "best_model.pt")
        run_demo(config_path, checkpoint, args.seed or 42)


if __name__ == "__main__":
    main()
