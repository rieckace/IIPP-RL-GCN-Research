"""
train_hybrid.py

Main training script for the Hybrid GNN-A* agent.
Uses the HybridObservationWrapper to feed PyG Data objects with 9 features.
"""

import argparse
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils.config_loader import load_config
from environment.evacuation_env import EvacuationEnv
from environment.wrappers import HybridObservationWrapper
from models.hybrid.trainer import HybridGNNDQNAgent
from evaluation.metrics import TrainingMetrics
from visualization.training_plots import plot_training_curves

def train(config: dict, num_episodes: int | None = None, seed: int | None = None) -> TrainingMetrics:
    train_cfg = config.get("training", {})
    paths_cfg = config.get("paths", {})

    episodes = num_episodes or train_cfg.get("episodes", 1500)
    log_interval = train_cfg.get("log_interval", 10)
    checkpoint_interval = train_cfg.get("checkpoint_interval", 100)
    early_stop_sr = train_cfg.get("early_stop_success_rate", 0.90)
    early_stop_window = train_cfg.get("early_stop_window", 100)

    checkpoint_dir = paths_cfg.get("checkpoint_dir", "checkpoints/hybrid")
    plots_dir = paths_cfg.get("plots_dir", "results/plots")
    logs_dir = paths_cfg.get("logs_dir", "results/logs")

    env_config_path = config.get("environment", {}).get("config_path", "configs/default.yaml")
    if not os.path.isabs(env_config_path):
        env_config_path = os.path.join(PROJECT_ROOT, env_config_path)
    env_config = load_config(env_config_path)

    base_env = EvacuationEnv(env_config)
    env = HybridObservationWrapper(base_env)
    
    agent = HybridGNNDQNAgent(config)
    metrics = TrainingMetrics()

    print("=" * 60)
    print("  HYBRID GNN-A* Training - Adaptive Evacuation")
    print("=" * 60)
    print(f"  Episodes:        {episodes}")
    print(f"  Grid:            {base_env.rows}x{base_env.cols}")
    print(f"  Node Features:   {agent.node_feature_dim} (8 standard + 1 A*)")
    print("=" * 60)
    print()

    start_time = time.time()
    best_success_rate = 0.0

    for episode in range(1, episodes + 1):
        obs, info = env.reset(seed=(seed + episode if seed else episode))
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

        agent.decay_epsilon()
        if episode % agent.target_update_freq == 0:
            agent.update_target()

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

        if episode % checkpoint_interval == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_ep{episode}.pt")
            agent.save_checkpoint(ckpt_path)

        rolling = metrics.rolling_average(window=early_stop_window)
        current_sr = rolling["success_rate"]
        if current_sr > best_success_rate and episode >= early_stop_window:
            best_success_rate = current_sr
            best_path = os.path.join(checkpoint_dir, "best_model.pt")
            agent.save_checkpoint(best_path)

        if current_sr >= early_stop_sr and episode >= early_stop_window:
            print(f"\n[OK] EARLY STOP at episode {episode}!")
            print(f"   Success rate: {current_sr*100:.1f}% >= {early_stop_sr*100:.1f}%")
            break

    final_path = os.path.join(checkpoint_dir, "final_model.pt")
    agent.save_checkpoint(final_path)

    csv_path = os.path.join(logs_dir, "hybrid_training_metrics.csv")
    metrics.save_csv(csv_path)
    print(f"\n[+] Metrics saved to: {csv_path}")

    plot_path = os.path.join(plots_dir, "hybrid_training.png")
    plot_training_curves(metrics.to_dict(), save_path=plot_path, title="Hybrid GNN-A* Training")

    print(f"\n[OK] Training completed in {time.time() - start_time:.1f}s")
    env.close()
    return metrics
