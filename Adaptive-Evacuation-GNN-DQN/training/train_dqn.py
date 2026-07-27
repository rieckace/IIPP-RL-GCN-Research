"""
train_dqn.py

Main training script for the DQN evacuation agent.
Handles the full training loop: environment interaction, learning,
logging, checkpointing, early stopping, and plot generation.

Usage:
    cd Adaptive-Evacuation-GNN-DQN
    python training/train_dqn.py
    python training/train_dqn.py --config configs/dqn.yaml --episodes 500 --seed 42
"""

import argparse
import os
import sys
import time

import numpy as np

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils.config_loader import load_config
from environment.evacuation_env import EvacuationEnv
from models.dqn.trainer import DQNAgent
from evaluation.metrics import TrainingMetrics
from visualization.training_plots import plot_training_curves


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DQN agent on Evacuation Environment")
    parser.add_argument("--config", type=str,
                        default=os.path.join(PROJECT_ROOT, "configs", "dqn.yaml"),
                        help="Path to DQN training config YAML.")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Override number of training episodes.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override random seed.")
    return parser.parse_args()


def train(config: dict, num_episodes: int | None = None, seed: int | None = None) -> TrainingMetrics:
    """Run the full DQN training loop.

    Args:
        config:       Merged config dict (DQN + environment).
        num_episodes: Override for number of episodes.
        seed:         Override for random seed.

    Returns:
        TrainingMetrics with all recorded episode data.
    """
    # --- Unpack config ---
    train_cfg = config.get("training", {})
    paths_cfg = config.get("paths", {})

    episodes = num_episodes or train_cfg.get("episodes", 1500)
    log_interval = train_cfg.get("log_interval", 10)
    checkpoint_interval = train_cfg.get("checkpoint_interval", 100)
    early_stop_sr = train_cfg.get("early_stop_success_rate", 0.90)
    early_stop_window = train_cfg.get("early_stop_window", 100)

    if seed is not None:
        config.setdefault("training", {})["seed"] = seed

    checkpoint_dir = paths_cfg.get("checkpoint_dir", "checkpoints/dqn")
    plots_dir = paths_cfg.get("plots_dir", "results/plots")
    logs_dir = paths_cfg.get("logs_dir", "results/logs")

    # --- Load environment config ---
    env_config_path = config.get("environment", {}).get("config_path", "configs/default.yaml")
    if not os.path.isabs(env_config_path):
        env_config_path = os.path.join(PROJECT_ROOT, env_config_path)
    env_config = load_config(env_config_path)

    # --- Create environment and agent ---
    env = EvacuationEnv(env_config)
    agent = DQNAgent(config)
    metrics = TrainingMetrics()

    # --- Training header ---
    print("=" * 60)
    print("  DQN Training - Adaptive Evacuation Environment")
    print("=" * 60)
    print(f"  Episodes:        {episodes}")
    print(f"  Grid:            {env.rows}x{env.cols}")
    print(f"  Network:         {config.get('network', {}).get('hidden_layers', [256, 256, 128])}")
    print(f"  Learning rate:   {agent.learning_rate}")
    print(f"  Batch size:      {agent.batch_size}")
    print(f"  Replay buffer:   {len(agent.memory)}/{agent.memory.buffer.maxlen}")
    print(f"  Device:          {agent.device}")
    print(f"  Double DQN:      Yes")
    print("=" * 60)
    print()

    start_time = time.time()
    best_success_rate = 0.0

    for episode in range(1, episodes + 1):
        # --- Run one episode ---
        obs, info = env.reset(seed=episode)
        total_reward = 0.0
        episode_loss = 0.0
        loss_count = 0
        steps = 0

        while True:
            action = agent.act(obs, explore=True)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.memory.push(obs, action, reward, next_obs, done)

            loss = agent.learn()
            if loss is not None:
                episode_loss += loss
                loss_count += 1

            obs = next_obs
            total_reward += reward
            steps += 1

            if done:
                break

        # --- Post-episode updates ---
        agent.decay_epsilon()

        if episode % agent.target_update_freq == 0:
            agent.update_target()

        # --- Record metrics ---
        avg_loss = episode_loss / loss_count if loss_count > 0 else 0.0
        success = info.get("reason", "") == "reached_exit"
        metrics.record(
            reward=total_reward,
            steps=steps,
            success=success,
            loss=avg_loss,
            epsilon=agent.epsilon,
            fire_count=info.get("fire_count", 0),
            reason=info.get("reason", ""),
        )

        # --- Logging ---
        if episode % log_interval == 0:
            rolling = metrics.rolling_average(window=50)
            elapsed = time.time() - start_time
            print(
                f"Episode {episode:>5}/{episodes}"
                f"  |  Reward: {total_reward:>+8.1f}"
                f"  |  Avg50: {rolling['avg_reward']:>+8.1f}"
                f"  |  SR: {rolling['success_rate']*100:>5.1f}%"
                f"  |  Loss: {avg_loss:>.4f}"
                f"  |  e: {agent.epsilon:.3f}"
                f"  |  {elapsed:>6.0f}s"
            )

        # --- Checkpointing ---
        if episode % checkpoint_interval == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_ep{episode}.pt")
            agent.save_checkpoint(ckpt_path)

        # --- Best model tracking ---
        rolling = metrics.rolling_average(window=early_stop_window)
        current_sr = rolling["success_rate"]
        if current_sr > best_success_rate and episode >= early_stop_window:
            best_success_rate = current_sr
            best_path = os.path.join(checkpoint_dir, "best_model.pt")
            agent.save_checkpoint(best_path)

        # --- Early stopping ---
        if current_sr >= early_stop_sr and episode >= early_stop_window:
            print(f"\n[OK] EARLY STOP at episode {episode}!")
            print(f"   Success rate: {current_sr*100:.1f}% >= {early_stop_sr*100:.1f}%")
            break

    # --- Save final checkpoint ---
    final_path = os.path.join(checkpoint_dir, "final_model.pt")
    agent.save_checkpoint(final_path)

    # --- Save metrics CSV ---
    csv_path = os.path.join(logs_dir, "training_metrics.csv")
    metrics.save_csv(csv_path)
    print(f"\n[+] Metrics saved to: {csv_path}")

    # --- Generate plots ---
    plot_path = os.path.join(plots_dir, "dqn_training.png")
    plot_training_curves(metrics.to_dict(), save_path=plot_path)

    # --- Training summary ---
    total_time = time.time() - start_time
    final_rolling = metrics.rolling_average(window=50)
    print()
    print("=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Episodes trained:   {len(metrics.rewards)}")
    print(f"  Total time:         {total_time:.1f}s")
    print(f"  Best success rate:  {best_success_rate*100:.1f}%")
    print(f"  Final avg reward:   {final_rolling['avg_reward']:+.1f}")
    print(f"  Final success rate: {final_rolling['success_rate']*100:.1f}%")
    print(f"  Final epsilon:      {agent.epsilon:.4f}")
    print(f"  Best model:         {os.path.join(checkpoint_dir, 'best_model.pt')}")
    print("=" * 60)

    env.close()
    return metrics


def main() -> None:
    args = parse_args()

    # Load DQN config
    dqn_config = load_config(args.config, validate=False)

    # Run training
    train(dqn_config, num_episodes=args.episodes, seed=args.seed)


if __name__ == "__main__":
    main()
