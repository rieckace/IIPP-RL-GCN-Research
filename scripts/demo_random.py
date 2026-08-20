"""
demo_random.py

Run a single episode of the evacuation environment with a random agent.
Renders each step to the terminal so you can watch the agent navigate
while fire spreads dynamically.

Usage:
    cd Adaptive-Evacuation-GNN-DQN
    python scripts/demo_random.py
    python scripts/demo_random.py --config configs/default.yaml --delay 0.3
"""

import argparse
import os
import sys
import time

# Add project root to path so 'environment' and 'utils' are importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils.config_loader import load_config
from environment.evacuation_env import EvacuationEnv
from environment.constants import ACTION_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a random-agent demo of the Evacuation Environment."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(PROJECT_ROOT, "configs", "default.yaml"),
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.25,
        help="Seconds to wait between rendered frames.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--no-pause",
        dest="pause_on_terminal",
        action="store_false",
        help="Exit immediately after a terminal state instead of waiting.",
    )
    parser.set_defaults(pause_on_terminal=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # --- Load config ---
    config = load_config(args.config)
    print(f"Loaded config from: {args.config}")
    print(f"Grid: {config['grid']['rows']}×{config['grid']['cols']}")
    print(f"Max steps: {config['dynamics']['max_steps']}")
    print(f"Fire spread probability: {config['dynamics']['fire_spread_probability']}")
    print()
    time.sleep(1)

    # --- Create environment ---
    env = EvacuationEnv(config, render_mode="human")
    obs, info = env.reset(seed=args.seed)
    time.sleep(args.delay)

    # --- Run episode ---
    total_reward = 0.0
    step = 0

    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step += 1
        time.sleep(args.delay)

        if terminated or truncated:
            break

    # --- Episode summary ---
    print("=" * 50)
    print(f"  EPISODE COMPLETE")
    print(f"  Steps:        {step}")
    print(f"  Total Reward:  {total_reward:+.1f}")
    print(f"  Outcome:       {info['reason']}")
    print(f"  Fire cells:    {info['fire_count']}")
    print(f"  Smoke cells:   {info['smoke_count']}")
    if info.get("reason") == "reached_exit":
        print("  Status:        SUCCESS - agent reached the exit")
    elif info.get("reason") == "hit_fire":
        print("  Status:        FAILURE - agent was caught by fire")
    elif info.get("reason") == "max_steps_exceeded":
        print("  Status:        FAILURE - max steps exceeded")
    print("=" * 50)

    if args.pause_on_terminal:
        input("Press Enter to close the demo and keep the final frame visible...")

    env.close()


if __name__ == "__main__":
    main()
