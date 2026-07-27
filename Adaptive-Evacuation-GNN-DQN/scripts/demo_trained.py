"""
demo_trained.py

Load a trained DQN checkpoint and run a visual episode in the terminal.

Usage:
    cd Adaptive-Evacuation-GNN-DQN
    python scripts/demo_trained.py
    python scripts/demo_trained.py --checkpoint checkpoints/dqn/best_model.pt --delay 0.3
"""

import argparse
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils.config_loader import load_config
from environment.evacuation_env import EvacuationEnv
from models.dqn.trainer import DQNAgent
from environment.constants import ACTION_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Demo a trained DQN agent.")
    parser.add_argument("--checkpoint", type=str,
                        default=os.path.join(PROJECT_ROOT, "checkpoints", "dqn", "best_model.pt"),
                        help="Path to trained model checkpoint.")
    parser.add_argument("--config", type=str,
                        default=os.path.join(PROJECT_ROOT, "configs", "dqn.yaml"),
                        help="Path to DQN config YAML.")
    parser.add_argument("--delay", type=float, default=0.3,
                        help="Seconds between rendered frames.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Environment seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load configs
    dqn_config = load_config(args.config, validate=False)
    env_config_path = dqn_config.get("environment", {}).get("config_path", "configs/default.yaml")
    if not os.path.isabs(env_config_path):
        env_config_path = os.path.join(PROJECT_ROOT, env_config_path)
    env_config = load_config(env_config_path)

    # Create env and agent
    env = EvacuationEnv(env_config, render_mode="human")
    agent = DQNAgent(dqn_config)

    if os.path.exists(args.checkpoint):
        agent.load_checkpoint(args.checkpoint)
        agent.epsilon = 0.0  # Pure greedy
        print(f"[OK] Loaded trained model: {args.checkpoint}")
    else:
        print(f"[!] Checkpoint not found: {args.checkpoint}")
        print("   Running with untrained agent (random actions).")
        agent.epsilon = 1.0

    time.sleep(1)

    # Run episode
    obs, info = env.reset(seed=args.seed)
    time.sleep(args.delay)

    total_reward = 0.0
    step = 0

    while True:
        action = agent.act(obs, explore=False)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step += 1
        time.sleep(args.delay)

        if terminated or truncated:
            break

    # Summary
    print("=" * 50)
    print(f"  EPISODE COMPLETE")
    print(f"  Steps:         {step}")
    print(f"  Total Reward:  {total_reward:+.1f}")
    print(f"  Outcome:       {info['reason']}")
    print(f"  Fire cells:    {info['fire_count']}")
    print(f"  Smoke cells:   {info['smoke_count']}")
    print("=" * 50)

    env.close()


if __name__ == "__main__":
    main()
