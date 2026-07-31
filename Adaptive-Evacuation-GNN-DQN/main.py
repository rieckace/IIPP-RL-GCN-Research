"""
main.py

Unified entry point for the Adaptive Evacuation GNN-DQN project.

Usage:
    python main.py --mode train   --config configs/dqn.yaml
    python main.py --mode evaluate --checkpoint checkpoints/dqn/best_model.pt
    python main.py --mode demo     --checkpoint checkpoints/dqn/best_model.pt
"""

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adaptive Evacuation GNN-DQN — Unified Entry Point"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "evaluate", "demo", "train-gnn", "eval-gnn", "train-hybrid", "train-marl"],
        default="train",
        help="Execution mode.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(PROJECT_ROOT, "configs", "dqn.yaml"),
        help="Path to config YAML.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join(PROJECT_ROOT, "checkpoints", "dqn", "best_model.pt"),
        help="Path to model checkpoint (for evaluate/demo modes).",
    )
    parser.add_argument("--episodes", type=int, default=None, help="Override episode count.")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed.")
    parser.add_argument("--delay", type=float, default=0.3, help="Frame delay for demo mode.")
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

    if args.mode == "train":
        from utils.config_loader import load_config
        from training.train_dqn import train

        config = load_config(args.config, validate=False)
        train(config, num_episodes=args.episodes, seed=args.seed)
        print(f"[OK] Training DQN completed.")

    elif args.mode == "train-gnn":
        from utils.config_loader import load_config
        from training.train_gnn import train as train_gnn
        config = load_config(args.config, validate=False)
        print("Starting GNN-DQN Training...")
        train_gnn(config, num_episodes=args.episodes, seed=args.seed)
        print(f"[OK] Training GNN-DQN completed.")
        
    elif args.mode == "eval-gnn":
        print("To evaluate transferability, run:")
        print("python training/evaluate_gnn_transfer.py --rows 15 --cols 15")

    elif args.mode == "train-hybrid":
        from utils.config_loader import load_config
        from training.train_hybrid import train as train_hybrid
        config = load_config(args.config, validate=False)
        print("Starting Hybrid GNN-A* Training...")
        train_hybrid(config, num_episodes=args.episodes, seed=args.seed)
        print(f"[OK] Training Hybrid GNN-A* completed.")

    elif args.mode == "train-marl":
        from utils.config_loader import load_config
        from training.train_marl import train as train_marl
        config = load_config(args.config, validate=False)
        print("Starting Multi-Agent GNN Training...")
        train_marl(config, num_episodes=args.episodes, seed=args.seed)
        print(f"[OK] Training MARL completed.")

    elif args.mode == "evaluate":
        from utils.config_loader import load_config
        from environment.evacuation_env import EvacuationEnv
        from models.dqn.trainer import DQNAgent
        from models.dqn.inference import evaluate_agent

        dqn_config = load_config(args.config, validate=False)
        env_config_path = dqn_config.get("environment", {}).get("config_path", "configs/default.yaml")
        if not os.path.isabs(env_config_path):
            env_config_path = os.path.join(PROJECT_ROOT, env_config_path)
        env_config = load_config(env_config_path)

        env = EvacuationEnv(env_config)
        agent = DQNAgent(dqn_config)
        agent.load_checkpoint(args.checkpoint)
        agent.epsilon = 0.0

        episodes = args.episodes or 100
        results = evaluate_agent(env, agent, num_episodes=episodes)

        print(f"\nEvaluation over {episodes} episodes:")
        print(f"  Avg Reward:    {results['avg_reward']:+.1f}")
        print(f"  Success Rate:  {results['success_rate']*100:.1f}%")
        print(f"  Avg Steps:     {results['avg_steps']:.1f}")
        env.close()

    elif args.mode == "demo":
        import time
        from utils.config_loader import load_config
        from environment.evacuation_env import EvacuationEnv
        from models.dqn.trainer import DQNAgent

        dqn_config = load_config(args.config, validate=False)
        env_config_path = dqn_config.get("environment", {}).get("config_path", "configs/default.yaml")
        if not os.path.isabs(env_config_path):
            env_config_path = os.path.join(PROJECT_ROOT, env_config_path)
        env_config = load_config(env_config_path)

        env = EvacuationEnv(env_config, render_mode="human")
        agent = DQNAgent(dqn_config)

        if os.path.exists(args.checkpoint):
            agent.load_checkpoint(args.checkpoint)
            agent.epsilon = 0.0
            print(f"[OK] Loaded: {args.checkpoint}")
            print("[!] Note: this checkpoint was trained before the current fire-spread tuning.")
            print("    If success is unstable, retrain on the updated environment config.")
        else:
            print(f"[!] No checkpoint found. Using random agent.")
            agent.epsilon = 1.0

        seed = args.seed or 42
        obs, _ = env.reset(seed=seed)
        time.sleep(args.delay)

        total_reward = 0.0
        while True:
            action = agent.act(obs, explore=False)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            time.sleep(args.delay)
            if terminated or truncated:
                break

        print(f"\nResult: {info['reason']} | Reward: {total_reward:+.1f}")
        if info.get("reason") == "reached_exit":
            print("Status: SUCCESS - agent reached the exit")
        elif info.get("reason") == "hit_fire":
            print("Status: FAILURE - agent was caught by fire")
        elif info.get("reason") == "max_steps_exceeded":
            print("Status: FAILURE - max steps exceeded")

        if args.pause_on_terminal:
            input("Press Enter to close the demo and keep the final frame visible...")
        env.close()


if __name__ == "__main__":
    main()
