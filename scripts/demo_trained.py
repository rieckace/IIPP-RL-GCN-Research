"""Terminal demo for a trained CNN-DQN baseline."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from environment.evacuation_env import EvacuationEnv
from models.dqn.trainer import DQNAgent
from utils.config_loader import load_config


def run_demo(config_path: str, checkpoint: str, seed: int = 42, delay: float = 0.3) -> None:
    config = load_config(config_path, validate=False)
    env_config_path = config.get("environment", {}).get("config_path", "configs/default.yaml")
    if not os.path.isabs(env_config_path):
        env_config_path = str(PROJECT_ROOT / env_config_path)
    env_config = load_config(env_config_path, validate=False)

    env = EvacuationEnv(env_config, render_mode="human")
    agent = DQNAgent(config)
    agent.load_checkpoint(checkpoint)
    agent.epsilon = 0.0

    obs, _ = env.reset(seed=seed)
    total_reward = 0.0
    steps = 0
    while True:
        action = agent.act(obs, explore=False)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        time.sleep(delay)
        if terminated or truncated:
            break

    print(f"Outcome: {info['reason']}; steps={steps}; reward={total_reward:+.2f}")
    env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "dqn.yaml"))
    parser.add_argument("--checkpoint", default=str(PROJECT_ROOT / "checkpoints" / "dqn" / "best_model.pt"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--delay", type=float, default=0.3)
    args = parser.parse_args()
    run_demo(args.config, args.checkpoint, args.seed, args.delay)


if __name__ == "__main__":
    main()
