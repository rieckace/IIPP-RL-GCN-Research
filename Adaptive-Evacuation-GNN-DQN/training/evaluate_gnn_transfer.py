"""
evaluate_gnn_transfer.py

Evaluates the trained GNN agent on a variable-sized environment (e.g., 15x15)
to test the transferability of graph representations.
"""

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils.config_loader import load_config
from environment.evacuation_env import EvacuationEnv
from environment.wrappers import GraphObservationWrapper
from models.gnn.trainer import GNNDQNAgent
from evaluation.metrics import TrainingMetrics

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate GNN transferability on new grid sizes.")
    parser.add_argument("--checkpoint", type=str,
                        default=os.path.join(PROJECT_ROOT, "checkpoints", "gnn", "best_model.pt"),
                        help="Path to trained GNN model checkpoint.")
    parser.add_argument("--config", type=str,
                        default=os.path.join(PROJECT_ROOT, "configs", "gnn.yaml"),
                        help="Path to GNN config YAML.")
    parser.add_argument("--rows", type=int, default=15, help="Number of rows for transfer env.")
    parser.add_argument("--cols", type=int, default=15, help="Number of cols for transfer env.")
    parser.add_argument("--episodes", type=int, default=100, help="Evaluation episodes.")
    return parser.parse_args()

def main():
    args = parse_args()

    # Load base configs
    gnn_config = load_config(args.config, validate=False)
    env_config_path = gnn_config.get("environment", {}).get("config_path", "configs/default.yaml")
    if not os.path.isabs(env_config_path):
        env_config_path = os.path.join(PROJECT_ROOT, env_config_path)
    env_config = load_config(env_config_path)

    # Override environment dimensions for transfer
    env_config["grid"]["rows"] = args.rows
    env_config["grid"]["cols"] = args.cols
    # Expand wall boundaries or agent start logically, but default.yaml logic in env handles arbitrary sizes.
    # We will just let the env initialize on a larger grid with the same config structure.

    base_env = EvacuationEnv(env_config)
    env = GraphObservationWrapper(base_env)

    agent = GNNDQNAgent(gnn_config)
    
    if os.path.exists(args.checkpoint):
        agent.load_checkpoint(args.checkpoint)
        print(f"[OK] Loaded checkpoint: {args.checkpoint}")
    else:
        print(f"[!] Checkpoint not found: {args.checkpoint}. Evaluating with random weights.")
    
    agent.epsilon = 0.0  # Greedy eval

    print(f"Evaluating GNN on {args.rows}x{args.cols} grid for {args.episodes} episodes...")

    successes = 0
    total_rewards = 0
    outcomes = {}

    for ep in range(args.episodes):
        obs, info = env.reset(seed=1000 + ep)
        ep_reward = 0
        while True:
            action = agent.act(obs, explore=False)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            if terminated or truncated:
                break
        
        total_rewards += ep_reward
        reason = info.get("reason", "unknown")
        outcomes[reason] = outcomes.get(reason, 0) + 1
        if reason == "reached_exit":
            successes += 1

    print("\n" + "="*50)
    print(f"  TRANSFER EVALUATION RESULTS ({args.rows}x{args.cols})")
    print("="*50)
    print(f"  Avg Reward:    {total_rewards / args.episodes:+.1f}")
    print(f"  Success Rate:  {(successes / args.episodes) * 100:.1f}%")
    print("\n  Outcomes:")
    for reason, count in outcomes.items():
        print(f"    {reason:<20}: {count} ({(count / args.episodes) * 100:.1f}%)")
    print("="*50)

if __name__ == "__main__":
    main()
