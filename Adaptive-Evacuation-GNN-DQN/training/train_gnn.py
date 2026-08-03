"""
train_gnn.py

Main training script for the GNN-DQN evacuation agent.
Uses the GraphObservationWrapper to feed PyG Data objects to the agent.
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
from environment.wrappers import GraphObservationWrapper
from models.gnn.trainer import GNNDQNAgent
from evaluation.metrics import TrainingMetrics
from visualization.training_plots import plot_training_curves


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GNN-DQN agent on Evacuation Environment")
    parser.add_argument("--config", type=str,
                        default=os.path.join(PROJECT_ROOT, "configs", "gnn.yaml"),
                        help="Path to GNN training config YAML.")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Override number of training episodes.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override random seed.")
    return parser.parse_args()


def train(config: dict, num_episodes: int | None = None, seed: int | None = None) -> TrainingMetrics:
    """Run the full GNN-DQN training loop."""
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

    checkpoint_dir = paths_cfg.get("checkpoint_dir", "checkpoints/gnn")
    if not os.path.isabs(checkpoint_dir):
        checkpoint_dir = os.path.join(PROJECT_ROOT, checkpoint_dir)
    checkpoint_dir = os.path.normpath(checkpoint_dir)
        
    plots_dir = paths_cfg.get("plots_dir", "results/plots")
    if not os.path.isabs(plots_dir):
        plots_dir = os.path.join(PROJECT_ROOT, plots_dir)
    plots_dir = os.path.normpath(plots_dir)
        
    logs_dir = paths_cfg.get("logs_dir", "results/logs")
    if not os.path.isabs(logs_dir):
        logs_dir = os.path.join(PROJECT_ROOT, logs_dir)
    logs_dir = os.path.normpath(logs_dir)

    # --- Load environment config ---
    env_config_path = config.get("environment", {}).get("config_path", "configs/default.yaml")
    if not os.path.isabs(env_config_path):
        env_config_path = os.path.join(PROJECT_ROOT, env_config_path)
    env_config = load_config(env_config_path)

    # --- Create environment and agent ---
    from environment.make_env import make_env
    import random
    training_maps = ["office", "apartment", "school", "hospital"]
    
    agent = GNNDQNAgent(config)
    metrics = TrainingMetrics()
 
    # --- Training header ---
    print("=" * 60)
    print("  GNN-DQN Training - Adaptive Evacuation Environment")
    print("=" * 60)
    print(f"  Episodes:        {episodes}")
    print(f"  Grid:            Multi-scale (10x10 to 18x18)")
    print(f"  GCN Layers:      {config.get('network', {}).get('gcn_hidden_dims', [64, 64, 64])}")
    print(f"  MLP Layers:      {config.get('network', {}).get('mlp_hidden_dims', [128, 64])}")
    print(f"  Learning rate:   {agent.learning_rate}")
    print(f"  Batch size:      {agent.batch_size}")
    print(f"  Device:          {agent.device}")
    print("=" * 60)
    print()
 
    start_time = time.time()
    best_success_rate = 0.0
 
    for episode in range(1, episodes + 1):
        # Select map randomly for this episode to learn scale and layout invariance
        map_name = random.choice(training_maps)
        base_env = make_env(map_name)
        base_env.randomize_agent_start = True  # Enable start state randomization for training generalization
        env = GraphObservationWrapper(base_env)
        
        # --- Run one episode ---
        obs, info = env.reset(seed=episode)
        total_reward = 0.0
        episode_loss = 0.0
        loss_count = 0
        steps = 0

        while True:
            action = agent.act(obs, explore=True)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated

            agent.memory.push(obs, action, reward, next_obs, done)

            loss = agent.learn()
            if loss is not None:
                episode_loss += loss
                loss_count += 1

            obs = next_obs
            total_reward += reward
            steps += 1

            if terminated or truncated:
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

    # --- Save metrics CSV and plots ---
    try:
        csv_path = os.path.join(logs_dir, "gnn_training_metrics.csv")
        metrics.save_csv(csv_path)
        print(f"\n[+] Metrics saved to: {csv_path}")

        plot_path = os.path.join(plots_dir, "gnn_training.png")
        plot_training_curves(metrics.to_dict(), save_path=plot_path, title="GNN-DQN Training")
    except Exception as e:
        print(f"\n[!] Warning: Failed to save metrics or plots: {e}")

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
    gnn_config = load_config(args.config, validate=False)
    train(gnn_config, num_episodes=args.episodes, seed=args.seed)


if __name__ == "__main__":
    main()
