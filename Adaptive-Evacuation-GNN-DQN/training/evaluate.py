"""
evaluate.py

Post-training evaluation script. Loads a trained DQN checkpoint,
runs greedy episodes, and reports performance statistics.

Usage:
    cd Adaptive-Evacuation-GNN-DQN
    python training/evaluate.py --checkpoint checkpoints/dqn/best_model.pt
"""

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils.config_loader import load_config
from environment.evacuation_env import EvacuationEnv
from models.dqn.trainer import DQNAgent
from models.dqn.inference import evaluate_agent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained DQN agent")
    parser.add_argument("--checkpoint", type=str,
                        default=os.path.join(PROJECT_ROOT, "checkpoints", "dqn", "best_model.pt"),
                        help="Path to trained model checkpoint.")
    parser.add_argument("--config", type=str,
                        default=os.path.join(PROJECT_ROOT, "configs", "dqn.yaml"),
                        help="Path to DQN config YAML.")
    parser.add_argument("--episodes", type=int, default=100,
                        help="Number of evaluation episodes.")
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
    env = EvacuationEnv(env_config)
    agent = DQNAgent(dqn_config)
    agent.load_checkpoint(args.checkpoint)
    agent.epsilon = 0.0  # Pure greedy evaluation

    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"Evaluating over {args.episodes} episodes (greedy)...\n")

    # Run evaluation
    results = evaluate_agent(env, agent, num_episodes=args.episodes)

    # Print results
    print("=" * 55)
    print("  EVALUATION RESULTS")
    print("=" * 55)
    print(f"  Episodes:       {results['num_episodes']}")
    print(f"  Avg Reward:     {results['avg_reward']:+.1f} ± {results['std_reward']:.1f}")
    print(f"  Avg Steps:      {results['avg_steps']:.1f}")
    print(f"  Success Rate:   {results['success_rate']*100:.1f}%")
    print()
    print("  Outcome Breakdown:")
    for reason, count in sorted(results["outcomes"].items(), key=lambda x: -x[1]):
        pct = count / results["num_episodes"] * 100
        print(f"    {reason:<25s}  {count:>4d}  ({pct:5.1f}%)")
    print("=" * 55)

    env.close()


if __name__ == "__main__":
    main()
