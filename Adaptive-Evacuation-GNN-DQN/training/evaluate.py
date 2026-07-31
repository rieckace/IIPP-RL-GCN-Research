"""
evaluate.py

Post-training evaluation script. Loads a trained DQN checkpoint,
runs greedy episodes, and reports performance statistics.

Usage:
    cd Adaptive-Evacuation-GNN-DQN
    python training/evaluate.py --checkpoint checkpoints/dqn/best_model.pt
"""

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils.config_loader import load_config
from environment.evacuation_env import EvacuationEnv
from models.dqn.trainer import DQNAgent
from models.dqn.inference import evaluate_agent
from evaluation.statistics import summarize_outcomes


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
    ci_low, ci_high = results.get("success_rate_ci95", (0.0, 0.0))
    print(f"  Success Rate:   {results['success_rate']*100:.1f}%")
    print(f"  95% CI:         [{ci_low*100:.1f}%, {ci_high*100:.1f}%]")
    print(f"  Success Count:   {results.get('success_count', 0)}")
    print(f"  Failure Count:   {results.get('failure_count', 0)}")
    print()
    print("  Outcome Breakdown:")
    outcome_percentages = summarize_outcomes(results["outcomes"], results["num_episodes"])
    for reason, count in sorted(results["outcomes"].items(), key=lambda x: -x[1]):
        pct = outcome_percentages.get(reason, 0.0) * 100
        print(f"    {reason:<25s}  {count:>4d}  ({pct:5.1f}%)")
    print("=" * 55)

    summary_path = os.path.join(PROJECT_ROOT, "results", "logs", "dqn_evaluation_summary.json")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "episodes": results["num_episodes"],
                "avg_reward": float(results["avg_reward"]),
                "std_reward": float(results["std_reward"]),
                "avg_steps": float(results["avg_steps"]),
                "success_rate": float(results["success_rate"]),
                "success_rate_ci95": [float(ci_low), float(ci_high)],
                "success_count": int(results.get("success_count", 0)),
                "failure_count": int(results.get("failure_count", 0)),
                "outcomes": results["outcomes"],
            },
            f,
            indent=2,
        )
    print(f"\nSaved evaluation summary to: {summary_path}")

    env.close()


if __name__ == "__main__":
    main()
